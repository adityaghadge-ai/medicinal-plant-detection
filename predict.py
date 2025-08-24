import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Disable oneDNN warnings (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths
MODEL_PATH = "models/plant_classifier.h5"

LABELS_PATH = "models/class_labels.json"
TEST_DIR = "test_samples"  # Folder where you put your test images

# Image size (must match what you used in train.py)
IMG_SIZE = (128, 128)

print("🔄 Loading model and labels...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# Invert mapping: {class_name: index} -> {index: class_name}
class_labels = {v: k for k, v in class_labels.items()}


def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    label = class_labels[pred_idx]
    conf = preds[0][pred_idx]
    return label, conf


# Loop through all test images
for img_file in os.listdir(TEST_DIR):
    img_path = os.path.join(TEST_DIR, img_file)
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        label, conf = predict_image(img_path)
        print(f"🖼 {img_file} → 🌿 {label} ({conf:.2f})")
