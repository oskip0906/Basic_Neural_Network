import numpy as np

''' Loss functions '''

# Mean Squared Error (Regression)
class MeanSquaredError:

    def compute(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

# Mean Absolute Error (Regression)
class MeanAbsoluteError:

    def compute(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def derivative(y_true, y_pred):
        return np.sign(y_pred - y_true) / np.size(y_true)
    
# Binary Cross Entropy (Binary Classification)
class BinaryCrossEntropy:

    def compute(y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def derivative(y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * np.size(y_true))

# Categorical Cross Entropy (Multiclass Classification)
class CategoricalCrossEntropy:

    def compute(y_true, y_pred):
        logits_stable = y_pred - np.max(y_pred, axis=0)
        log_probs = logits_stable - np.log(np.sum(np.exp(logits_stable), axis=0))
        return -np.mean(np.sum(y_true * log_probs, axis=0))
    
    def derivative(y_true, y_pred):
        logits_stable = y_pred - np.max(y_pred, axis=0)
        exp_logits = np.exp(logits_stable)
        return exp_logits / np.sum(exp_logits, axis=0) - y_true