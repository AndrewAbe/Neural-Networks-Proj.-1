import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# Fetch and preprocess the dataset
ionosphere = fetch_ucirepo(id=52)
X = ionosphere.data.features.values  # Convert DataFrame to numpy array directly
y = ionosphere.data.targets
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y).ravel()  # Ensure y_encoded is a 1D array

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Convert the numpy arrays to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

class IonosphereNN(nn.Module):
    def __init__(self):
        super(IonosphereNN, self).__init__()
        self.fc1 = nn.Linear(34, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  

# Create the TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize model, loss function, and optimizer
model = IonosphereNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to keep track of the losses and accuracies
train_losses = []
test_losses = []
test_accuracies = []

num_epochs = 400

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)  # Multiply by batch size for total loss
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    # Testing phase
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            predicted = torch.round(torch.sigmoid(outputs))
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}')

# Plotting the training and test losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

# Plotting the test accuracy
plt.figure(figsize=(10, 5))
plt.plot(test_accuracies, label='Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.legend()
plt.show()
