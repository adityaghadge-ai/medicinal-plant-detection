import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Disable oneDNN warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths
MODEL_PATH = "models/plant_classifier.h5"   # or medicinal_model.h5
LABELS_PATH = "models/class_labels.json"

# Load model and labels
print("🔄 Loading model and labels...")
model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# Reverse mapping {id: label}
id_to_label = {v: k for k, v in class_labels.items()}

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # adjust if different
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds))

    return id_to_label[pred_idx], confidence



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            label, conf = predict_image(filepath)
            return render_template("result.html", filename=file.filename, label=label, confidence=round(conf, 2))

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename="uploads/" + filename))


if __name__ == "__main__":
    app.run(debug=True)
