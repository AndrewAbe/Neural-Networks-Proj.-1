# ----Part 3: Classification----
# First Step is importing libraries/pip installing and fetching data from dataset
!pip install ucimlrepo
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
X = ionosphere.data.features.values
y = ionosphere.data.targets
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y).ravel()

#Split Data Set into Test and Training 
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#Create a Neural Network with two hidden layers with 64 and 32 neurons in the hidden layer and have RELU forward propegation.
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

#Create the tensorDatasets and dataLoaders, which initializes datasets for managing batches of the training and testing data and creates loaders that provide an iterable over these batches.
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).view(-1, 1))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test).view(-1, 1))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

#Initialize model, loss function, and ADAM optimizer with learning rate of 0.001. For output layer we used binary cross entropy as loss function. We're using sigmoid activation function for output layer.
model = IonosphereNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Initialize early stopping parameters
early_stopping_patience = 20
min_val_loss = np.inf
epochs_no_improve = 0

#Initialize lists to keep track of the losses and accuracies
train_losses = []
test_losses = []
test_accuracies = []

#Define the number of epochs and do the training and testing set on those two split data sets. Second part is the early stopping that stops the program from running when the results do not improve. Needs to be one big block of code or else it will not stop properly.
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
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

    #early stopping
    if test_loss < min_val_loss:
        min_val_loss = test_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stopping_patience:
            print('Early stopping!')
            break

#Printing accuracy and Graph Plotting with results
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
