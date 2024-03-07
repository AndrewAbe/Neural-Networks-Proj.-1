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

# Define the neural network
class IonosphereNN(nn.Module):
    def __init__(self):
        super(IonosphereNN, self).__init__()
        self.fc1 = nn.Linear(34, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output without activation, BCEWithLogitsLoss later

# Create the TensorDatasets and DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Initialize model, loss function, and optimizer
model = IonosphereNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predicted = torch.round(torch.sigmoid(outputs))
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')

"""old code-kinda works
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

class IonosphereNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(IonosphereNN, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(input_size, hidden_size1)
        # Define the second hidden layer
        self.hidden2 = nn.Linear(hidden_size1, hidden_size2)
        # Define the output layer
        self.output = nn.Linear(hidden_size2, output_size)
        
    def forward(self, x):
        # Pass the input through the first hidden layer, then apply a ReLU activation
        x = F.relu(self.hidden1(x))
        # Pass through the second hidden layer, then apply another ReLU activation
        x = F.relu(self.hidden2(x))
        # Pass through the output layer
        x = self.output(x)
        return x

#given from website-import start
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
ionosphere = fetch_ucirepo(id=52) 
  
# data (as pandas dataframes) 
X = ionosphere.data.features 
y = ionosphere.data.targets 
  
# metadata 
#print(ionosphere.metadata) 
  
# variable information 
#print(ionosphere.variables) 
#given from website-end
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

input_size = 34  # Number of input features
hidden_size1 = 64
hidden_size2 = 32
output_size = 1  # For binary classification

# Initialize the model
model = IonosphereNN(input_size, hidden_size1, hidden_size2, output_size)

# Fetch dataset
ionosphere = fetch_ucirepo(id=52)

# Data (as pandas dataframes)
X = ionosphere.data.features
y = ionosphere.data.targets

# Encode the 'Class' target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y = pd.DataFrame(y_encoded, columns=['Class'])  # Creates a new DataFrame with encoded labels

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network Definition
class IonosphereNN(nn.Module):
    def __init__(self):
        super(IonosphereNN, self).__init__()
        self.fc1 = nn.Linear(34, 64)  # Assuming 34 features in the dataset
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Using BCEWithLogitsLoss, so no sigmoid here
        return x

# Convert data to PyTorch tensors and create DataLoader
X_train_tensor = torch.FloatTensor(X_train.values)
y_train_tensor = torch.FloatTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test.values)
y_test_tensor = torch.FloatTensor(y_test.values)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize Model, Loss Function, and Optimizer
model = IonosphereNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the Model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predicted = torch.round(torch.sigmoid(outputs))
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
"""
