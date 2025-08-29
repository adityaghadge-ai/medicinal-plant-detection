import os
import time
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Quiet oneDNN chatter (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# ---------------- Paths ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

PLANT_MODEL_PATH   = os.path.join(MODEL_DIR, "medicinal_mobilenet.h5")      # medicinal plant id (128x128)
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "medicinal_disease_model.h5")  # PlantVillage disease (224x224)

PLANT_LABELS_PATH   = os.path.join(MODEL_DIR, "class_labels.json")   # {name: id}
DISEASE_LABELS_PATH = os.path.join(MODEL_DIR, "disease_labels.json") # {name: id}
REMEDIES_PATH       = os.path.join(MODEL_DIR, "remedies.json")       # {"Plant___Disease": "remedy ..."}

# ---------------- Load models ---------------- #
print("🔄 Loading models ...")
medicinal_model = load_model(PLANT_MODEL_PATH)
disease_model   = load_model(DISEASE_MODEL_PATH)

# ---------------- Load & invert labels ---------------- #
def load_and_invert(path):
    """
    Input JSON format is {label_name: id}. We invert it to {id:int -> label_name:str}.
    Handles ids that are numbers or numeric strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    id_to_label = {}
    for name, idx in d.items():
        try:
            id_to_label[int(idx)] = name
        except Exception:
            # Just in case idx is not numeric (rare), keep as-is
            id_to_label[idx] = name
    return d, id_to_label

plant_labels_map, plant_id_to_label       = load_and_invert(PLANT_LABELS_PATH)
disease_labels_map, disease_id_to_label   = load_and_invert(DISEASE_LABELS_PATH)

with open(REMEDIES_PATH, "r", encoding="utf-8") as f:
    remedies = json.load(f)

# A set of plants that the disease model knows (from PlantVillage names)
disease_plants_from_labels = set([name.split("___")[0] for name in disease_labels_map.keys()])

print(f"✅ Medicinal labels loaded ({len(plant_id_to_label)} classes)")
print(f"✅ Disease labels loaded ({len(disease_id_to_label)} classes)")

# ---------------- Preprocessing helpers ---------------- #
def prep_img_128(img_path):
    """For medicinal model (MobileNetV2 128)."""
    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0).astype("float32") / 255.0
    return x

def prep_img_224(img_path):
    """For PlantVillage disease model (224)."""
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0).astype("float32") / 255.0
    return x

# ---------------- Decision thresholds ---------------- #
DISEASE_CONF_GATE = 0.65  # if disease model ≥ this, show disease pipeline
MEDICINAL_MIN_SHOW = 0.50 # fallback minimum to show medicinal if disease is weak

# ---------------- Routes ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        f = request.files["file"]
        if not f or f.filename.strip() == "":
            return redirect(request.url)

        # Save with a unique name to avoid caching issues
        fname = secure_filename(f.filename)
        name, ext = os.path.splitext(fname)
        unique_name = f"{name}_{int(time.time())}{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        f.save(save_path)

        # ---- Run BOTH models with their correct input sizes ---- #
        # 1) Medicinal plant identification
        img128 = prep_img_128(save_path)
        med_pred = medicinal_model.predict(img128)[0]
        med_idx = int(np.argmax(med_pred))
        med_conf = float(med_pred[med_idx])
        plant_name = plant_id_to_label.get(med_idx, f"Unknown-{med_idx}")

        # 2) PlantVillage disease model
        img224 = prep_img_224(save_path)
        dis_pred = disease_model.predict(img224)[0]
        dis_idx = int(np.argmax(dis_pred))
        dis_conf = float(dis_pred[dis_idx])
        full_disease_label = disease_id_to_label.get(dis_idx, f"Unknown-{dis_idx}")

        # Parse "Tomato___Late_blight" → plant + disease
        if "___" in full_disease_label:
            dis_plant, dis_name = full_disease_label.split("___", 1)
        else:
            dis_plant, dis_name = full_disease_label, "Unknown"

        # Remedy lookup (flat map with full key)
        remedy_text = remedies.get(full_disease_label, "No remedy found.")

        # ---- Decision logic (auto-detect) ---- #
        # If disease model is confident, we trust it (fruit pipeline).
        # Otherwise, show medicinal identification and mark disease N/A.
        use_disease_pipeline = dis_conf >= DISEASE_CONF_GATE

        if use_disease_pipeline:
            # Fruit/disease path (Pipeline 2)
            return render_template(
                "result.html",
                filename=unique_name,
                plant_label=dis_plant,
                plant_conf=f"{dis_conf*100:.1f}%",
                disease_label=dis_name,
                disease_conf=f"{dis_conf*100:.1f}%",
                remedy=remedy_text
            )
        else:
            # Medicinal identification (Pipeline 1)
            # Only show this if the medicinal model isn't totally confused
            # (keeps UI sensible when both are weak).
            show_med = max(med_conf, MEDICINAL_MIN_SHOW)
            return render_template(
                "result.html",
                filename=unique_name,
                plant_label=plant_name,
                plant_conf=f"{med_conf*100:.1f}%",
                disease_label="Not applicable for medicinal plants (N/A)",
                disease_conf="N/A",
                remedy="Used directly as medicinal plant 🌿"
            )

    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    # Important: Flask reloader can load models twice; this is okay but noisy.
    app.run(debug=True)
