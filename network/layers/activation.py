from network.layers.layer import Layer
import numpy as np

# Activation layer
class Activation(Layer):

    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    # output_gradient: derivative of the loss with respect to the output
    def backward(self, output_gradient, learning_rate):
        # Derivative of loss with respect to input
        # Simplify to element-wise multiplication of output gradient with input derivative
        return np.multiply(output_gradient, self.derivative(self.input))