import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the trained model
MODEL_PATH = "medicinal_disease_model.h5"
model = load_model(MODEL_PATH)

# Path to dataset (to get class names)
DATASET_DIR = "disease_dataset_split/train"

# Get class names from dataset folder
class_names = sorted(os.listdir(DATASET_DIR))
print("Loaded classes:", class_names)

# Function to predict disease from image
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))   # resize to model input size
    img_array = image.img_to_array(img) / 255.0             # normalize
    img_array = np.expand_dims(img_array, axis=0)           # add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)

    return class_names[predicted_class], confidence

# Test with an image
if __name__ == "__main__":
    test_image = "test_img2/1a80b84d-1a5a-4e23-8deb-823ba928e29a___FREC_C.Rust 4431.JPG"  # 🔹 replace with your image path
    if os.path.exists(test_image):
        disease, conf = predict_disease(test_image)
        print(f"✅ Predicted: {disease} (confidence: {conf:.2f})")
    else:
        print("⚠️ Please place a test image as 'test_img2/1a80b84d-1a5a-4e23-8deb-823ba928e29a___FREC_C.Rust 4431.JPG' in the project folder")
