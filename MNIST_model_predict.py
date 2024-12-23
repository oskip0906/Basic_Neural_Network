import pickle
import numpy as np
from PIL import Image

# Load the model
with open('MNIST_MODEL.pkl', 'rb') as file:
    model = pickle.load(file)

# Load image and predict the label
def predict_label(image_path):
    # Load image, convert it to grayscale, and resize to 28x28 pixels
    img = Image.open(image_path).convert('L').resize((28, 28))
    # Make sure the image is in the same shape as the expected model input
    img_array = np.reshape(np.array(img) / 255.0, (784, 1))
    # Predict the label
    prediction = model.forward(img_array)
    label = np.argmax(prediction, axis=0)
    return label[0]

# Random image of a handwritten digit
image_path = 'test_image.png'
predicted_label = predict_label(image_path)
print(f"The predicted digit is: {predicted_label}")