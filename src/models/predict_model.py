import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import argparse


# Define the function to load and preprocess the image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Define the function to make a prediction
def make_prediction(model, image_path, label_mapping):
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)

    # Make the prediction
    prediction = model.predict(img_array)

    # Decode the prediction
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = {v: k for k, v in label_mapping.items()}[predicted_label_index]

    return predicted_label


# Main function
def main(image_path):
    # Load the trained model
    model = load_model('Carmine400i70aDeploy.h5')

    # Define label mapping
    label_mapping = {
        "['normal']": 0,
        "['pneumonia']": 1  # Adjust as per your unique labels
    }

    # Make a prediction
    predicted_label = make_prediction(model, image_path, label_mapping)

    print(f'The predicted label for the image is: {predicted_label}')


# If the script is run directly, use argparse to handle the image path input
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the label of an input image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file to be predicted')
    args = parser.parse_args()

    main(args.image_path)