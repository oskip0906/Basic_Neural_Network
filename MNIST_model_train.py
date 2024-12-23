from network.main import Network
from network.model_engine import ModelEngine
from network.layers.dense import Dense  
from network.activation_functions import LeakyReLU
from network.loss_functions import CategoricalCrossEntropy
from network.utils import compute_classification_accuracy
import numpy as np
import pandas as pd

''' Training a model to recognize handwritten digits from the MNIST dataset '''

# Load the training data
data = pd.read_csv('mnist_dataset/mnist_train.csv')

# Split data into features and labels
x_train = np.reshape(data.iloc[:, 1:].values / 255.0, (60000, 784, 1)) # Normalize pixel values
y_train = np.eye(10)[data.iloc[:, 0].values].reshape(60000, 10, 1)

# Define custom layers
layers = [
    Dense(784, 64),
    LeakyReLU(),
    Dense(64, 32),
    LeakyReLU(),
    Dense(32, 10)
]

# Create the network
network = Network(layers, CategoricalCrossEntropy)

# Intialize model engine with custom parameters for training and evaluation
engine = ModelEngine(network, compute_classification_accuracy)

# Train the network
engine.train(x_train, y_train, epochs=25, learning_rate=0.005, log_filename='TRAINING_LOG.txt')

# Load the testing data
data = pd.read_csv('mnist_dataset/mnist_test.csv')

# Split data into features and labels
x_test = np.reshape(data.iloc[:, 1:].values / 255.0, (10000, 784, 1))
y_test = np.eye(10)[data.iloc[:, 0].values].reshape(10000, 10, 1)

# Evaluate the trained model
accuracy = engine.evaluate(x_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the trained model
engine.save('MNIST_MODEL.pkl')