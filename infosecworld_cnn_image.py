import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

"""We'll create a custom dataset class to load and preprocess our data:"""

class eBayDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.csv_file = csv_file
        self.transform = transform
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label

"""We'll define data transforms to resize images to 224x224 and normalize pixel values"""

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

"""Our CNN model will consist of several convolutional and pooling layers followed by fully connected layers"""

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # output layer (1 for IP, 0 otherwise)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""We'll define a training loop to train our model using the Adam optimizer and cross-entropy loss"""

def train_model(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

"""We'll tie everything together in our main function"""

def main():
    # Set device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    dataset = eBayDataset('train_data.csv', transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = CNNModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for epoch in range(10):
        train_model(model, device, loader, optimizer, epoch)

    # Save trained model
    torch.save(model.state_dict(), 'image_cnn_model.pth')

if __name__ == '__main__':
    main()