import os, json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ===============================
# Config
# ===============================
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ===============================
# Load Models
# ===============================
# Load plant identification model
plant_model = load_model("plant_model.h5")

# Load disease model (try best -> fallback to final)
if os.path.exists("best1_disease_model.h5"):
    print("✅ Loading best1_disease_model.h5")
    disease_model = load_model("best1_disease_model.h5")
elif os.path.exists("final1_disease_model.h5"):
    print("⚠️ best1 not found, loading final1_disease_model.h5")
    disease_model = load_model("final1_disease_model.h5")
else:
    raise FileNotFoundError("❌ No disease model found! Please add best1_disease_model.h5 or final1_disease_model.h5")

# ===============================
# Load JSON Files
# ===============================
with open("class1_labels.json") as f:
    plant_labels = {int(v): k for k, v in json.load(f).items()}

with open("models/disease_labels.json") as f:
    disease_labels = {int(v): k for k, v in json.load(f).items()}

with open("models/remedies.json") as f:
    remedies = json.load(f)

# ===============================
# Utility Functions
# ===============================
def prepare_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_plant(img_path):
    img_array = prepare_image(img_path)
    preds = plant_model.predict(img_array)
    idx = np.argmax(preds)
    return plant_labels[idx], float(np.max(preds))

def predict_disease(img_path):
    img_array = prepare_image(img_path)
    preds = disease_model.predict(img_array)
    idx = np.argmax(preds)
    disease = disease_labels[idx]
    confidence = float(np.max(preds))
    remedy = remedies.get(disease, "No remedy available.")
    return disease, confidence, remedy

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]
    task = request.form["task"]  # "plant" or "disease"

    if file.filename == "":
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        if task == "plant":
            label, confidence = predict_plant(filepath)
            return render_template("result_plant.html",
                                   filename=file.filename,
                                   label=label,
                                   confidence=confidence)

        elif task == "disease":
            disease, confidence, remedy = predict_disease(filepath)
            return render_template("result_disease.html",
                                   filename=file.filename,
                                   disease=disease,
                                   confidence=confidence,
                                   remedy=remedy)

# ===============================
# Image Display Route
# ===============================
@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

# ===============================
if __name__ == "__main__":
    app.run(debug=True)
