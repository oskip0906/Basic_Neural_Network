# Fully customizable neural network
class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    # Forward pass (predict)
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # Backward pass (train)
    def backward(self, x, y, learning_rate):
        output = x
        # Forward pass to get the final output
        output = self.forward(output)
        # Calculate the loss and derivative of the loss
        loss = self.loss.compute(y, output)
        loss_derivative = self.loss.derivative(y, output)
        # Backward pass to update the weights and biases
        for layer in reversed(self.layers):
            loss_derivative = layer.backward(loss_derivative, learning_rate)
        return loss