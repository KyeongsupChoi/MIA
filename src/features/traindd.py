
import pandas as pd

reduced = pd.read_csv('../visualization/mia.csv')

print(reduced.to_string())

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

# Data Preprocessing
image_dir = '../../data/raw/img'  # Directory where images are stored

# Load image paths and labels from the CSV file
image_paths = reduced['ImageID'].apply(lambda x: os.path.join(image_dir, x))
labels = reduced['Labels']

# Define label mapping
label_mapping = {
    "['normal']": 0,
    "['pneumonia']": 1  # Adjust as per your unique labels
}

# Convert labels to categorical format
labels = labels.apply(lambda x: label_mapping[x[0]] if isinstance(x, list) else label_mapping[
    x])  # Extract the label from the list and map it

valid_image_paths = image_paths[image_paths.apply(os.path.exists)]

# Load and preprocess images
X = []
y = []
for img_path, label in zip(valid_image_paths, labels):
    img = load_img(img_path, target_size=(224, 224))  # Assuming input size for the model is 224x224
    img = img_to_array(img)
    img = img / 255.0  # Normalize pixel values
    X.append(img)
    y.append(label)

X = np.array(X)
y = to_categorical(y, num_classes=len(label_mapping))  # Convert labels to one-hot encoded format

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




# Define ResNet model
def create_resnet_model(input_shape, num_classes):
    base_model = ResNet50(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False  # Freeze layers of the pre-trained model

    model = Sequential([
        base_model,
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model

# Define input shape and number of classes
input_shape = (224, 224, 3)  # Assuming input size for the model is 224x224 with 3 channels (RGB)
num_classes = 2  # Number of classes (normal and abnormal)

# Create the model
model = create_resnet_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# model.save('Carmine400i70a.h5')