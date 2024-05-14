import json
import time

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import datetime

print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Create Dataset
classes = {"astilbe": 0, "bellflower": 1, "black_eyed_susan": 2, "calendula": 3, "california_poppy": 4, "carnation": 5,
           "common_daisy": 6, "coreopsis": 7, "daffodil": 8, "dandelion": 9, "iris": 10, "magnolia": 11, "rose": 12,
           "sunflower": 13, "tulip": 14, "water_lily": 15}


class FlowerDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.zeros(16)
        y_label[classes[self.annotations.iloc[index, 1]]] = 1

        if self.transform:
            image = self.transform(image)
        return (image, y_label)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((180, 180)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.RandomAffine(0, shear=10),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),


])

dataset = FlowerDataset(csv_file='data.csv', root_dir='', transform=transform)
train_set, valid_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.95),
                                                               len(dataset) - int(len(dataset) * 0.95)],
                                                     generator=torch.Generator().manual_seed(123))
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=32, shuffle=True)


# Create Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 96, 3)
        self.pool3 = nn.MaxPool2d(4, 4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(9600, 800)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(800, 16)

    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = self.pool3(F.relu(self.conv3(out)))
        out = self.flatten(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out


net = ConvNet()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=0.00001)


def load_model(model_folder_path):
    loaded = ConvNet()
    loaded = loaded.to(device)
    loaded.load_state_dict(torch.load(model_folder_path + "/model.pth"))
    loaded.eval()
    with open(model_folder_path + "/model_data.json", "r") as f:
        model_data = json.load(f)
    return loaded, model_data["train_losses"], model_data["train_correct_per_epoch"], model_data["valid_losses"], \
        model_data["valid_correct_per_epoch"]

def save_model(model_folder_path, train_losses, train_correct_per_epoch, valid_losses, valid_correct_per_epoch):
    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)
    torch.save(net.state_dict(), model_folder_path + "/model.pth")
    model_data = {
        "train_losses": train_losses,
        "train_correct_per_epoch": train_correct_per_epoch,
        "valid_losses": valid_losses,
        "valid_correct_per_epoch": valid_correct_per_epoch
    }
    with open(model_folder_path + "/model_data.json", "w") as f:
        json.dump(model_data, f)

# Train Model
def train():
    num_epochs = 100

    train_losses = []
    train_correct_per_epoch = []
    valid_losses = []
    valid_correct_per_epoch = []

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == torch.argmax(labels, 1)).sum().item()
            print(f'Batch {i}/{len(train_loader)}', end='\r')
        train_losses.append(total_loss)
        train_correct_per_epoch.append(correct)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {total_loss / len(train_set)}, Accuracy: {correct}/{len(train_set)} images correct')
        total_loss = 0
        correct = 0
        for i, (images, labels) in enumerate(valid_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Track the accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == torch.argmax(labels, 1)).sum().item()
            print(f'Batch {i}/{len(valid_loader)}', end='\r')
        valid_losses.append(total_loss)
        valid_correct_per_epoch.append(correct)
        print(
            f'Validation: Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {total_loss / len(valid_set)}, Accuracy: {correct}/{len(valid_set)} images correct')
        if correct >= 650:
            save_model(f"models/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", train_losses,
                       train_correct_per_epoch, valid_losses, valid_correct_per_epoch)
            print('Model Saved')