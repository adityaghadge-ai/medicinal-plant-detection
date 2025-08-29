import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Paths
MODEL_PATH = "models/medicinal_mobilenet.h5"
LABELS_PATH = "models/class_labels.json"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)
    class_labels = {v: k for k, v in class_labels.items()}  # reverse mapping

# Image size (same as training)
IMG_SIZE = (128, 128)

def predict_plant(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    plant_name = class_labels[predicted_class]
    print(f"🌿 Predicted Plant: {plant_name} ({confidence:.2f}%)")
    return plant_name, confidence

# Example usage
if __name__ == "__main__":
    test_image = r"D:\EDI\medicinal-plant-detection\test_samples\4230.jpg"  # replace with your test image
    predict_plant(test_image)
