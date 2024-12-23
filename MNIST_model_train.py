from network.main import Network
from network.layers.dense import Dense  
from network.activation_functions import LeakyReLU
from network.loss_functions import CategoricalCrossEntropy
from utils import compute_classification_accuracy
import numpy as np
import pandas as pd
import pickle

''' Training a model to recognize handwritten digits from the MNIST dataset '''

# Load the training data
data = pd.read_csv('mnist_dataset/mnist_train.csv')

# Split data into features and labels
x_train = np.reshape(data.iloc[:, 1:].values / 255.0, (60000, 784, 1)) # Normalize pixel values
y_train = np.eye(10)[data.iloc[:, 0].values].reshape(60000, 10, 1)

# Define layers
layers = [
    Dense(784, 64),
    LeakyReLU(),
    Dense(64, 32),
    LeakyReLU(),
    Dense(32, 10)
]

# Create the network
network = Network(layers, CategoricalCrossEntropy)

# Define parameters
epochs = 25
learning_rate = 0.0025

# Train the network
for epoch in range(epochs):
    error = 0
    correct_predictions = 0
    
    for x, y in zip(x_train, y_train):
        error += network.backward(x, y, learning_rate)
        y_pred = network.forward(x)
        correct_predictions += compute_classification_accuracy(y, y_pred)
    
    accuracy = correct_predictions / len(x_train)
    print(f'Epoch {epoch}, Error: {error:.4f}, Accuracy: {accuracy * 100:.2f}%')

# Load the testing data
data = pd.read_csv('mnist_dataset/mnist_test.csv')

# Split data into features and labels
x_test = np.reshape(data.iloc[:, 1:].values / 255.0, (10000, 784, 1))
y_test = np.eye(10)[data.iloc[:, 0].values].reshape(10000, 10, 1)

# Calculate accuracy on the test set
correct_predictions = 0
for x, y in zip(x_test, y_test):
    y_pred = network.forward(x)
    correct_predictions += compute_classification_accuracy(y, y_pred)

accuracy = correct_predictions / 10000
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
with open('MNIST_MODEL.pkl', 'wb') as model_file:
    pickle.dump(network, model_file)