import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self.data_frame['Labels'].unique().tolist()  # Extract unique classes from the 'Labels' column

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx, 0])  # Assuming the first column contains image file names
        image = read_image(img_name)
        label = self.data_frame.iloc[idx, 7]  # Assuming the eighth column contains labels
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformation for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to your CSV files and image directory
train_csv_file = "../data/train_dataset.csv"
test_csv_file = "../data/test_dataset.csv"
img_dir = "../data/img"

# Create custom datasets
train_data = CustomImageDataset(csv_file=train_csv_file, img_dir=img_dir, transform=transform)
test_data = CustomImageDataset(csv_file=test_csv_file, img_dir=img_dir, transform=transform)

# Define DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Define the model
model = models.resnet18(pretrained=True)  # You can use other pretrained models like resnet50, etc.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device
model = model.to(device)

# Training the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    # Evaluate the model
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            running_loss += loss.item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'chest_xray_classifier.pth')