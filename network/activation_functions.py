from network.layers.activation import Activation
import numpy as np

''' Activation functions '''

# Sigmoid
class Sigmoid(Activation):

    def __init__(self):
        super().__init__(self.sigmoid, self.sigmoid_derivative)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

# Tanh
class Tanh(Activation):

    def __init__(self):
        super().__init__(self.tanh, self.tanh_derivative)

    def tanh(self, x):
        return np.tanh(x)
        
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

# ReLU
class ReLU(Activation):
    def __init__(self):
        super().__init__(self.relu, self.relu_derivative)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

# Leaky ReLU
class LeakyReLU(Activation):

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        super().__init__(self.leaky_relu, self.leaky_relu_derivative)

    def leaky_relu(self, x):
        return np.maximum(self.alpha * x, x)

    def leaky_relu_derivative(self, x):
        return np.where(x > 0, 1, self.alpha)