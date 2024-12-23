import numpy as np

# Compute accuracy from true and predicted labels
# Used for classification tasks
def compute_classification_accuracy(y_true, y_pred):
    predicted_class = np.argmax(y_pred)
    true_class = np.argmax(y_true)
    return int(predicted_class.item() == true_class.item())