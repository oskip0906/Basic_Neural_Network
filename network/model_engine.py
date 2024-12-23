import pickle

class ModelEngine:

    def __init__(self, model=None, accuracy_fn=None):
        self.model = model
        self.accuracy_fn = accuracy_fn

    def train(self, features, labels, epochs, learning_rate, log_filename):
        # Train model for the specified number of epochs
        # Keep track of the error and accuracy
        with open(log_filename, 'w') as log_file:
            for epoch in range(epochs):
                error = 0
                correct_predictions = 0
                # Iterate over each sample
                for x, y in zip(features, labels):
                    error += self.model.backward(x, y, learning_rate)
                    y_pred = self.model.forward(x)
                    correct_predictions += self.accuracy_fn(y, y_pred)
                accuracy = correct_predictions / len(features)
                # Print statistics and write to log file
                log_message = f'Epoch {epoch + 1}, Error: {error:.4f}, Accuracy: {accuracy * 100:.2f}%\n'
                print(log_message.strip())
                log_file.write(log_message)

    # Calculate the accuracy on a testing set
    def evaluate(self, features, labels):
        correct_predictions = 0
        for x, y in zip(features, labels):
            y_pred = self.model.forward(x)
            correct_predictions += self.accuracy_fn(y, y_pred)
        return correct_predictions / len(features)

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)