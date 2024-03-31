#----Applying Linear Regression----
#Import Library and load the dataset

import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
response = requests.get(url)

# Split dataset into traning and test set
# Check if the request was successful
if response.status_code == 200:
    # Get the text content of the response
    data = response.text

    # Split the data into lines
    lines = data.split("\n")

    # Initialize lists to store features and target variable
    X = []
    y = []

    # Parse the data and extract features and target variable
    for line in lines[1:]:
        if line:
            values = line.strip().split("\t")
            X.append([float(values[2]), float(values[3])])
            y.append(float(values[10]))

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    print("Failed to fetch data from the URL.")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on Training Data and testing data
train_predictions = model.predict(X_train)
train_mse = mean_squared_error(y_train, train_predictions)
print("Training MSE:", train_mse)

test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Testing MSE:", test_mse)

