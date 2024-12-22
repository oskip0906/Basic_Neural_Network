import numpy as np

# Mean Squared Error
class MeanSquaredError:

    def compute(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

# Mean Absolute Error
class MeanAbsoluteError:

    def compute(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivative(y_true, y_pred):
        return np.sign(y_pred - y_true) / np.size(y_true)

# Huber Loss
class HuberLoss:
    
    def __init__(self, delta=1.0):
        self.delta = delta

    def compute(self, y_true, y_pred):
        error = y_true - y_pred
        return np.mean(np.where(np.abs(error) < self.delta, 0.5 * error ** 2, self.delta * (np.abs(error) - 0.5 * self.delta)))

    def derivative(self, y_true, y_pred):
        error = y_true - y_pred
        return np.where(np.abs(error) < self.delta, error, self.delta * np.sign(error)) / np.size(y_true)