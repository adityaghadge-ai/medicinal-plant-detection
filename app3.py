from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Check if key is available
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please check your .env file.")
print("✅ GROQ_API_KEY found.")

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
# Chatbot Integration (Groq)
# ===============================
import os
from groq import Groq

# Load API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set. Please set it as an environment variable.")

client = Groq(api_key=GROQ_API_KEY)

conversation = []

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global conversation

    if request.method == "POST":
        user_message = request.form["message"]
        lang = request.form["lang"]

        # Add user message to conversation
        conversation.append({"sender": "user", "text": user_message})

        # Prepare messages for Groq API
        messages = [
            {
                "role": "system",
                "content": f"""
You are a multilingual farming assistant.
STRICT RULE: Reply only in {lang}. Never answer in English unless {lang} is English. Never answer in Marathi unless {lang} is Marathi. Never answer in Hindi unless {lang} is Hindi.
             
Guidelines:
- Natural, conversational tone (like ChatGPT/DeepSeek/Grok).
- 3–5 sentences max (concise but informative).
- No repetition of user’s question.
- Simple, farmer-friendly advice (avoid jargon).
- Use bullet points only when helpful.
- Always stay supportive, clear, and practical.
"""
            },
            *[
                {
                    "role": "user" if msg["sender"] == "user" else "assistant",
                    "content": msg["text"]
                }
                for msg in conversation
            ]
        ]

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",   # ✅ updated Groq model
                messages=messages,
                temperature=0.7
            )
            bot_reply = response.choices[0].message.content.strip()
        except Exception as e:
            bot_reply = f"⚠️ Error connecting to Groq API: {str(e)}"

        # Save bot reply
        conversation.append({"sender": "bot", "text": bot_reply})

        return render_template("chat.html", conversation=conversation, lang=request.form.get("lang", "en"))

    # For GET requests, just render the chat page
    return render_template("chat.html", conversation=conversation, lang=request.args.get("lang", "en"))


if __name__ == "__main__":
    app.run(debug=True)