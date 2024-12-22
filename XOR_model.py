from network import Network
from network_layers.dense import Dense  
from activation_functions import Sigmoid
from loss_functions import MeanSquaredError
import numpy as np

# This is the famous XOR problem which requires a non-linear model to solve!

# Define the training data
x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# Define layers
layers = [
    Dense(2, 64),
    Sigmoid(),
    Dense(64, 1),
    Sigmoid()
]

# Create the network (customizable layers and loss function)
network = Network(layers, MeanSquaredError)

# Define parameters
epochs = 10000
learning_rate = 0.1

# Train the network
for epoch in range(epochs):
    error = 0
    for x, y in zip(x_train, y_train):
        error += network.train(x, y, learning_rate)
    print(f'Epoch {epoch}, Error: {error}')

# Make a prediction
prediction = network.predict(np.reshape([[0, 1]], (2, 1)))  
# Should output a value very close to 1 (the right answer)
print(prediction)