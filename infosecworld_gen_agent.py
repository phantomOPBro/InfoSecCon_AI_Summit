import os
import socket
import json
from datetime import date, datetime
from typing import List, Union, Optional, Dict

import requests
from googleapiclient.discovery import build
import shodan
from censys.search import CensysHosts, SearchClient

from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

# Constants loaded from environment variables
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', '')
AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT', '')
EMBED_MODEL = os.environ.get('EMBED_MODEL', '')
EMBED_MODEL_DEPLOYMENT = os.environ.get('EMBED_MODEL_DEPLOYMENT', '')
CENSYS_API_ID = os.environ.get('CENSYS_API_ID', '')
CENSYS_API_SECRET = os.environ.get('CENSYS_API_SECRET', '')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
BASE_URL = os.environ.get('BASE_URL', '')
SHODAN_API_KEY = os.environ.get('SHODAN_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
CSE_ID = os.environ.get('CSE_ID', '')

# Initialize LLM and embedding models
llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=1.0,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-02-15-preview",
)

embed_model = AzureOpenAIEmbedding(
    model=EMBED_MODEL,
    deployment_name=EMBED_MODEL_DEPLOYMENT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=f"{AZURE_ENDPOINT}openai/deployments/{EMBED_MODEL}/embeddings?api-version=2023-05-15",
    api_version="2023-05-15",
)

Settings.llm = llm
Settings.embed_model = embed_model

# Load documents
reader = SimpleDirectoryReader(input_dir="path_to_your_policies_and_standards")
docs = reader.load_data()
print(f"Loaded {len(docs)} documents")

# Split documents into nodes
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(docs)

# Create summary and vector indexes
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# Create query engines for summary and vector searches
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

vector_query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [{"key": "page_label", "value": p} for p in []],  # Adjust page numbers as needed
        condition=FilterCondition.OR,
    ),
)

# Define tools
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to the documents in the knowledge base (company policies and standards).",
)


def vector_query(query: str, page_numbers: List[str]) -> str:
    metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR),
    )
    response = query_engine.query(query)
    return response


vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query,
    description="Useful for obtaining deeper, more specific information about documents in your knowledge base (company policies and standards).",
)


def search_github_code(search_string: str):
    headers = {
        'Authorization': f'token {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3+json',
    }
    params = {'q': search_string}
    response = requests.get(BASE_URL, headers=headers, params=params)
    if response.status_code == 200:
        results = []
        for item in response.json().get('items', []):
            result = {
                'repository': item['repository']['full_name'],
                'file': item['name'],
                'path': item['path'],
                'url': item['html_url'],
            }
            results.append(result)
        return results
    else:
        print(f"Error: {response.status_code}")
        return None


gh_tool = FunctionTool.from_defaults(
    fn=search_github_code,
    description="Useful for finding code or text snippets in GitHub repositories.",
)


def search_shodan(query="default_query"):
    """
    Searches Shodan for the given query and returns a report of IPs, ports, and vulnerabilities.
    """
    try:
        shodan_api = shodan.Shodan(SHODAN_API_KEY)
        results = shodan_api.search(query)

        def generate_report(results):
            report = []
            for result in results['matches']:
                ports = []
                vulnerabilities = result.get('vulns', [])
                if 'port' in result:
                    ports.append(result['port'])
                if ports and vulnerabilities:
                    host_info = {
                        'IP': result['ip_str'],
                        'Ports': list(set(ports)),
                        'Vulnerabilities': list(set(vulnerabilities)),
                    }
                    report.append(host_info)
            return report

        report = generate_report(results)
        return json.dumps(report, separators=(',', ':'))
    except shodan.APIError as e:
        return f"Error: {e}"


shodan_tool = FunctionTool.from_defaults(
    fn=search_shodan,
    description="Useful for finding useful information associated with company-owned IPs. Such as open ports, services, CVEs, etc.",
)


def censys_search_hosts(query: str, virtual_hosts: str = "EXCLUDE") -> List[dict]:
    """
    Searches the Censys host index based on a query string.
    Args:
        query (str): The search query using Censys Search Language.
        virtual_hosts (str): Mode for querying virtual hosts. Options are 'EXCLUDE', 'INCLUDE', 'ONLY'.
    Returns:
        List[dict]: A list of host details matching the search criteria.
    """
    h = CensysHosts()
    search = h.search(query, virtual_hosts=virtual_hosts)
    return list(search)


def censys_view_host(ip_address: str, at_time: Optional[Union[date, datetime, str]] = None) -> dict:
    """
    Retrieves detailed information about a specific host.
    Args:
        ip_address (str): The IP address of the host to view.
        at_time (Optional[Union[date, datetime, str]]): Specific timestamp to retrieve host data for historical view.
    Returns:
        dict: A dictionary containing detailed information about the host.
    """
    h = CensysHosts()
    host_details = h.view(ip_address, at_time=at_time)
    return host_details


def censys_aggregate_hosts(query: str, field: str, num_buckets: int = None, virtual_hosts: str = "EXCLUDE") -> dict:
    """
    Constructs a report using a query, an aggregation field, and the number of buckets to bin.
    Args:
        query (str): Search query using Censys Search Language.
        field (str): Field to aggregate the results by.
        num_buckets (int): Number of buckets to return in the aggregation.
        virtual_hosts (str): Option to include virtual hosts in the aggregation. Options are 'EXCLUDE', 'INCLUDE'.
    Returns:
        dict: A dictionary containing aggregated data.
    """
    client = SearchClient()
    report = client.v2.hosts.aggregate(
        query,
        field,
        num_buckets=num_buckets,
        virtual_hosts=virtual_hosts,
    )
    return report


def censys_bulk_view_hosts(ip_addresses: List[str]) -> dict:
    """
    Retrieves detailed information for multiple IP addresses in bulk.
    Args:
        ip_addresses (List[str]): A list of IP addresses.
    Returns:
        dict: A dictionary with IP addresses as keys and host details as values.
    """
    h = CensysHosts()
    hosts_details = h.bulk_view(ip_addresses)
    return hosts_details


# Define tools for Censys operations
censys_search_hosts_tool = FunctionTool.from_defaults(
    fn=censys_search_hosts,
    description="""Searches the Censys host index based on a query string using Censys Search Language. Supports pagination and filtering of virtual hosts.
Examples:
- (example.com) and services.port=`22`
- ((example.com)) and labels=`login-page`
- ((example.com)) and location.country=`United States`
""",
)

censys_view_host_tool = FunctionTool.from_defaults(
    fn=censys_view_host,
    description="Retrieves detailed information about a specific host based on its IP address. Supports viewing historical data if a timestamp is provided.",
)

censys_aggregate_hosts_tool = FunctionTool.from_defaults(
    fn=censys_aggregate_hosts,
    description="""Constructs a report by aggregating data over a specified field from the Censys host index, based on a given query. Only use the num_buckets parameter if the user explicitly provides it. If there is an error due to too many results returned, request the user to enter it. Supports aggregation including or excluding virtual hosts.
Examples of fields to aggregate on:
- location.country (e.g., United States)
- services.port (e.g., 443)
- labels (e.g., 'remote-access')
- services.service_name (e.g., 'SSH')
- services.software.vendor (e.g., 'Squid Cache')
""",
)

censys_bulk_view_hosts_tool = FunctionTool.from_defaults(
    fn=censys_bulk_view_hosts,
    description="Retrieves detailed information in bulk for multiple IP addresses from the Censys host index.",
)


def search_and_format_results(search_term: str, num_results: int = None):
    """
    Retrieves Google search results for a given query and returns a formatted list of results.
    """
    # Set up the Google Custom Search service
    service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
    # Perform the search
    res = service.cse().list(q=search_term, cx=CSE_ID, num=num_results).execute()
    # Extract results and format them
    results = res.get('items', [])
    formatted_results = []
    for result in results:
        title = result.get('title')
        link = result.get('link')
        snippet = result.get('snippet')
        # Create a compact formatted string for each result
        formatted_result = f"Title: {title} | Link: {link} | Snippet: {snippet}"
        formatted_results.append(formatted_result)
    # Join all formatted results into a single string with newline as a separator
    return '\n'.join(formatted_results)


google_custom_search_tool = FunctionTool.from_defaults(
    fn=search_and_format_results,
    description="""Retrieves Google search results for a given query and returns a formatted list of results with title, link, and snippet from the search results. Use standard Google Search syntax, including quotation marks for exact matches and AND/OR operators.
Examples:
- "secure login" AND "best practices"
- "data loss prevention" OR DLP
""",
)


def resolve_ip_hostname(input_value):
    """
    Given an IP address or hostname, resolves it to the corresponding hostname or IP address.
    """
    try:
        # Check if the input value is an IP address
        socket.inet_aton(input_value)
        is_ip = True
    except socket.error:
        is_ip = False
    try:
        if is_ip:
            # Resolve hostname from IP address
            return socket.gethostbyaddr(input_value)[0]
        else:
            # Resolve IP from hostname
            return socket.gethostbyname(input_value)
    except socket.gaierror as e:
        return f"Error resolving {'IP' if is_ip else 'hostname'}: {str(e)}"


ip_hostname_resolution_tool = FunctionTool.from_defaults(
    fn=resolve_ip_hostname,
    description="Given an IP address, this tool resolves the hostname, and vice versa.",
)

# Agent setup with improved system prompt
agent_worker = FunctionCallingAgentWorker.from_tools(
    [
        vector_query_tool,
        summary_tool,
        gh_tool,
        shodan_tool,
        censys_bulk_view_hosts_tool,
        censys_aggregate_hosts_tool,
        censys_view_host_tool,
        censys_search_hosts_tool,
        google_custom_search_tool,
        ip_hostname_resolution_tool,
    ],
    llm=llm,
    verbose=True,
    system_prompt=(
        "You are an AI assistant equipped with various tools to assist the user in answering their queries. "
        "You have tools to search code on GitHub, analyze publicly exposed services and ports using Censys and Shodan, "
        "perform Google Custom Searches, resolve IP addresses and hostnames, and access company policies and standards for context. "
        "Use these tools effectively to provide accurate and helpful responses while adhering to security and privacy guidelines. Ruminate on your findings and feel free to use more than one tool at a time, use as many as needed to achieve the task."
    ),
)

agent = AgentRunner(agent_worker)

# Example usage
response = agent.chat("What is our company password policy?")

response = agent.chat(
    "Are there any repositories of code I should look into that might contain our company's sensitive data?"
)
