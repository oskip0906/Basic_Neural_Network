from network_layers.layer import Layer
import numpy as np

# Dense layer
class Dense(Layer):

    def __init__(self, input_size, output_size):
        # Initialize random weights and bias at the start
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    # output_gradient: derivative of the loss with respect to the output
    # Note: the output of this layer is the input to the next layer
    # learning_rate: rate for updating the weights and bias
    def backward(self, output_gradient, learning_rate):
        # Derivative of loss with respect to weights, bias, and input
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = output_gradient
        input_gradient = np.dot(self.weights.T, output_gradient)
        # Use gradient descent to update weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        # Derivative of loss with respect to input
        return input_gradient