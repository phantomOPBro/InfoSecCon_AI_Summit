# Harnessing AI to Detect Sensitive Data Exfiltration


Welcome to the repository for the **Harnessing AI to Detect Sensitive Data Exfiltration** talk presented at InfoSecWorld 2024 in Orlando, FLorida. This repository contains a PowerPoint presentation and three key code files demonstrating the use of AI techniques to prevent data loss.

## Repository Contents

- **PowerPoint Slide (coming soon, post conference):**  
  - `DLP_Presentation.pptx`: A PowerPoint file summarizing the key concepts, methodology, and use cases for AI in Data Loss Prevention (DLP). This slide deck is used during the talk to guide the audience through the practical applications of AI in securing sensitive information.

- **Code Files:**
  1. **Image CNN for DLP:**  
     - [infosecworld_cnn_image.py](https://github.com/phantomOPBro/InfoSecWorld_AI_Summit/blob/main/infosecworld_cnn_image.py): This file contains the code to build a Convolutional Neural Network (CNN) to analyze image data and perform binary classificaiton (0 or 1). 
   
  2. **Text-Based CNN for Log Analysis:**  
     - [infosecworld_cnn_log_analysis.py](https://github.com/phantomOPBro/InfoSecWorld_AI_Summit/blob/main/infosecworld_cnn_log_analysis.py): This file includes the code to build a CNN designed for analyzing logs using character tokenization. Again, this is a binary classification task. 
   
  3. **Generative AI Agent using Llama_Index:**  
     - [infosecworld_gen_agent.py](https://github.com/phantomOPBro/InfoSecWorld_AI_Summit/blob/main/infosecworld_gen_agent.py): This script uses the Llama_Index framework to create a generative AI agent capable of using tools to answer user questions. The agent can generate contextual responses and insights related to data security and risk detection. 

## Project Overview

This project showcases three distinct applications of AI in enhancing Data Loss Prevention capabilities:
1. **Image Analysis**: Utilizing CNNs to identify sensitive parts being sold on the deep/open web.
2. **Log Analysis**: Analyzing textual logs with CNNs to detect exfiltration of sensitive data.
3. **Generative AI for DLP**: Creating an AI-powered agent that helps automate responses and insights into potential data loss scenarios using the Llama_Index library.

## Setup and Usage

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/yourusername/dlp-ai-project.git
   cd dlp-ai-project
