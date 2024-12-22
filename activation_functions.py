from network_layers.activation import Activation
import numpy as np

# Sigmoid
class Sigmoid(Activation):

    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_derivative)

# Tanh
class Tanh(Activation):

    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_derivative = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_derivative)

# ReLU
class ReLU(Activation):

    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_derivative = lambda x: (x > 0)
        super().__init__(relu, relu_derivative)

# Leaky ReLU
class LeakyReLU(Activation):

    def __init__(self, alpha=0.01):
        leaky_relu = lambda x: np.maximum(alpha * x, x)
        leaky_relu_derivative = lambda x: (x > 0) + alpha * (x <= 0)
        super().__init__(leaky_relu, leaky_relu_derivative)