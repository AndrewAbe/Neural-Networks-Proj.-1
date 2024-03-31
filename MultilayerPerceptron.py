# ----Constructing a Multilayer Perceptron----
# Import libraries and define mathmatical functions that will be used in forward and back progegation

import math
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    if not isinstance(y_true, (list, tuple)):
        y_true = [y_true]
    if not isinstance(y_pred, (list, tuple)):
        y_pred = [y_pred]

    # Calculate squared differences
    squared_diffs = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]

    # Calculate the mean of squared differences
    return sum(squared_diffs) / len(squared_diffs)

#A neural network with:
#  - 2 inputs
#  - a hidden layer with 2 neurons (h1, h2)
#- another hidden layer with 2 neurons (h3, h4)
#  - an output layer with 1 neuron (o1)
# Need to intialize weights and biases. Create a forward and backpropagation function. Train and test the model with a early stopping. With our data we choose to do batch gradient descent with learning rate = 0.00001, epochs = 15000, batch size = 96.

class OurNeuralNetwork:
    def __init__(self):

       # Initialize weights
        self.w1 = random.gauss(0, 1)  # Mean=0, Standard Deviation=1
        self.w2 = random.gauss(0, 1)
        self.w3 = random.gauss(0, 1)
        self.w4 = random.gauss(0, 1)
        self.w5 = random.gauss(0, 1)
        self.w6 = random.gauss(0, 1)
        self.w7 = random.gauss(0, 1)
        self.w8 = random.gauss(0, 1)
        self.w9 = random.gauss(0, 1)
        self.w10 = random.gauss(0, 1)

        # Initialize biases
        self.b1 = random.gauss(0, 1)
        self.b2 = random.gauss(0, 1)
        self.b3 = random.gauss(0, 1)
        self.b4 = random.gauss(0, 1)
        self.b5 = random.gauss(0, 1)

    def feedforward(self, x):
        # Feedforward function
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        h3 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        h4 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b4)
        o1 = self.w9 * h3 + self.w10 * h4 + self.b5
        return o1

    def train(self, X_train, y_train, X_test, y_test):
        learn_rate = 0.00001
        epochs = 15000
        batch_size = 96  # Set the batch size
        data_array = X_train
        all_y_trues = y_train
        losses_train = []  # List to store losses
        losses_test = []   # List to store test losses
        best_test_loss = float('inf')  # Initialize the best test loss

        for epoch in range(epochs):
            epoch_loss_train = 0  # Initialize loss for this epoch
            epoch_loss_test = 0  # Initialize loss for this epoch

            # Shuffle the training data for each epoch
            # Generate a list of indices
            shuffled_indices = list(range(len(data_array)))

            # Shuffle the list of indices
            random.shuffle(shuffled_indices)
            # Initialize lists to store shuffled data
            data_array_shuffled = []
            all_y_trues_shuffled = []

            # Iterate over shuffled indices to create shuffled data
            for idx in shuffled_indices:
                data_array_shuffled.append(data_array[idx])
                all_y_trues_shuffled.append(all_y_trues[idx])

            # Iterate over batches
            for i in range(0, len(data_array_shuffled), batch_size):
                # Get batch
                batch_X = data_array_shuffled[i:i+batch_size]
                batch_y = all_y_trues_shuffled[i:i+batch_size]

                # Compute batch gradients
                d_L_d_w1_batch = 0
                d_L_d_w2_batch = 0
                d_L_d_w3_batch = 0
                d_L_d_w4_batch = 0
                d_L_d_w5_batch = 0
                d_L_d_w6_batch = 0
                d_L_d_w7_batch = 0
                d_L_d_w8_batch = 0
                d_L_d_w9_batch = 0
                d_L_d_w10_batch = 0
                # Update for other weights similarly
                d_L_d_b1_batch = 0
                d_L_d_b2_batch = 0
                d_L_d_b3_batch = 0
                d_L_d_b4_batch = 0
                d_L_d_b5_batch = 0
                # Update for other biases similarly

                # Iterate over samples in the batch to compute gradients
                for j in range(len(batch_X)):
                    x = batch_X[j]
                    y_true = batch_y[j]

                    # Convert x to a numpy array if it's not already
                    x = list(x)

                    # Feedforward
                    sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                    h1 = sigmoid(sum_h1)
                    sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                    h2 = sigmoid(sum_h2)
                    sum_h3 = self.w5 * h1 + self.w6 * h2 + self.b3
                    h3 = sigmoid(sum_h3)
                    sum_h4 = self.w7 * h1 + self.w8 * h2 + self.b4
                    h4 = sigmoid(sum_h4)
                    sum_o1 = self.w9 * h3 + self.w10 * h4 + self.b5
                    o1 = sum_o1
                    y_pred = o1
                    Loss = (y_true - y_pred) ** 2

                    # Backpropagation
                    d_L_d_ypred = -2 * (y_true - y_pred)
                    d_ypred_d_o1 = 1

                    # Neuron o1
                    d_o1_d_w9 = h3
                    d_o1_d_w10 = h4
                    d_o1_d_b5 = 1
                    d_o1_d_h3 = self.w9
                    d_o1_d_h4 = self.w10

                    # Neuron h3
                    d_h3_d_w5 = h1 * deriv_sigmoid(sum_h3)
                    d_h3_d_w6 = h2 * deriv_sigmoid(sum_h3)
                    d_h3_d_b3 = deriv_sigmoid(sum_h3)
                    d_h3_d_h1 = self.w5 * deriv_sigmoid(sum_h3)
                    d_h3_d_h2 = self.w6 * deriv_sigmoid(sum_h3)

                    # Neuron h4
                    d_h4_d_w7 = h1 * deriv_sigmoid(sum_h4)
                    d_h4_d_w8 = h2 * deriv_sigmoid(sum_h4)
                    d_h4_d_b4 = deriv_sigmoid(sum_h4)
                    d_h4_d_h1 = self.w7 * deriv_sigmoid(sum_h4)
                    d_h4_d_h2 = self.w8 * deriv_sigmoid(sum_h4)

                    # Neuron h1
                    d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                    d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                    d_h1_d_b1 = deriv_sigmoid(sum_h1)
                    d_h1_d_x1 = self.w1 * deriv_sigmoid(sum_h1)
                    d_h1_d_x2 = self.w2 * deriv_sigmoid(sum_h1)

                    # Neuron h2
                    d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                    d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                    d_h2_d_b2 = deriv_sigmoid(sum_h2)
                    d_h2_d_x1 = self.w3 * deriv_sigmoid(sum_h2)
                    d_h2_d_x2 = self.w4 * deriv_sigmoid(sum_h2)

                    # Update weights and biases
                    # Neuron o1
                    d_L_d_w9 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_w9
                    d_L_d_w10 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_w10
                    d_L_d_b5 =  d_L_d_ypred * d_ypred_d_o1 * d_o1_d_b5

                    # Neuron h3
                    d_L_d_w5 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h3 * d_h3_d_w5
                    d_L_d_w6 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h3 * d_h3_d_w6
                    d_L_d_b3 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h3 * d_h3_d_b3

                    # Neuron h4
                    d_L_d_w7 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h4 * d_h4_d_w7
                    d_L_d_w8 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h4 * d_h4_d_w8
                    d_L_d_b4 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h4 * d_h4_d_b4

                  # Neuron h1
                    d_o1_d_h1 = (d_o1_d_h3 * d_h3_d_h1) + (d_o1_d_h4 * d_h4_d_h1)
                    d_L_d_w1 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h1 * d_h1_d_w1
                    d_L_d_w2 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h1 * d_h1_d_w2
                    d_L_d_b1 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h1 * d_h1_d_b1

                    # Neuron h2
                    d_o1_d_h2 = (d_o1_d_h3 * d_h3_d_h2) + (d_o1_d_h4 * d_h4_d_h2)
                    d_L_d_w3 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h2 * d_h2_d_w3
                    d_L_d_w4 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h2 * d_h2_d_w4
                    d_L_d_b2 = d_L_d_ypred * d_ypred_d_o1 * d_o1_d_h2 * d_h2_d_b2

                    # Accumulate gradients for the batch
                    d_L_d_w1_batch += d_L_d_w1
                    d_L_d_w2_batch += d_L_d_w2
                    d_L_d_w3_batch += d_L_d_w3
                    d_L_d_w4_batch += d_L_d_w4
                    d_L_d_w5_batch += d_L_d_w5
                    d_L_d_w6_batch += d_L_d_w6
                    d_L_d_w7_batch += d_L_d_w7
                    d_L_d_w8_batch += d_L_d_w8
                    d_L_d_w9_batch += d_L_d_w9
                    d_L_d_w10_batch += d_L_d_w10
                    # Accumulate for other weights similarly
                    d_L_d_b1_batch += d_L_d_b1
                    d_L_d_b2_batch += d_L_d_b2
                    d_L_d_b3_batch += d_L_d_b3
                    d_L_d_b4_batch += d_L_d_b4
                    d_L_d_b5_batch += d_L_d_b5
                    # Accumulate for other biases similarly

                # Update weights and biases using accumulated gradients
                self.w1 -= learn_rate * (d_L_d_w1_batch / batch_size)
                self.w2 -= learn_rate * (d_L_d_w2_batch / batch_size)
                self.w3 -= learn_rate * (d_L_d_w3_batch / batch_size)
                self.w4 -= learn_rate * (d_L_d_w4_batch / batch_size)
                self.w5 -= learn_rate * (d_L_d_w5_batch / batch_size)
                self.w6 -= learn_rate * (d_L_d_w6_batch / batch_size)
                self.w7 -= learn_rate * (d_L_d_w7_batch / batch_size)
                self.w8 -= learn_rate * (d_L_d_w8_batch / batch_size)
                self.w9 -= learn_rate * (d_L_d_w9_batch / batch_size)
                self.w10 -= learn_rate * (d_L_d_w10_batch / batch_size)
                # Update for other weights similarly
                self.b1 -= learn_rate * (d_L_d_b1_batch / batch_size)
                self.b2 -= learn_rate * (d_L_d_b2_batch / batch_size)
                self.b3 -= learn_rate * (d_L_d_b3_batch / batch_size)
                self.b4 -= learn_rate * (d_L_d_b4_batch / batch_size)
                self.b5 -= learn_rate * (d_L_d_b5_batch / batch_size)
                # Update for other biases similarly

                # Accumulate loss for this batch
                epoch_loss_train += mse_loss(y_true, y_pred)

            # Calculate average loss for this epoch and store it
            epoch_loss_train /= (len(data_array) / batch_size)
            losses_train.append(epoch_loss_train)

            # Test the model
            predictions_test = []
            for i in range(len(X_test)):
                prediction = self.feedforward(X_test[i])
                predictions_test.append(prediction)

            epoch_loss_test = mse_loss(y_test, predictions_test)
            losses_test.append(epoch_loss_test)

            # Print or log the loss if needed
            if epoch % 10 == 0:
                print("Epoch %d: Train Loss: %.3f, Test Loss: %.3f" % (epoch, epoch_loss_train, epoch_loss_test))

            # Check for early stopping
            if epoch_loss_test < best_test_loss:
                best_test_loss = epoch_loss_test
            else:
                print("Early stopping at epoch %d" % epoch)
                break

        return losses_train, losses_test

#Since the dataset is the same as Part 1 linear regression, we only need to train the neural network.
# Train our neural network!
network = OurNeuralNetwork()

train_losses, test_losses = network.train(X_train, y_train, X_test, y_test)
# Plotting train and test loss over epochs
plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
plt.plot(range(len(test_losses)), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Losses over Epochs')
plt.legend()
plt.show()
