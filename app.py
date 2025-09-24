from dotenv import load_dotenv
import os, json, re
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from groq import Groq

# =============================
# Load environment variables
# =============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set.")
client = Groq(api_key=GROQ_API_KEY)

# =============================
# Config
# =============================
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (224, 224)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =============================
# Load Models
# =============================
disease_model = load_model("best_disease_model.h5")  # retrained disease model

# =============================
# Load Label JSONs
# =============================
with open("models/disease_labels.json") as f:
    disease_labels = {int(v): k for k, v in json.load(f).items()}

with open("models/fertilizers.json") as f:
    fertilizer_data = json.load(f)

# =============================
# Helper Functions
# =============================
def prepare_image(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def format_to_bullets(text):
    """Convert numbered/dash-separated text into bullet points."""
    points = re.split(r"\d+\.\s+|[-*]\s+|\n", text)
    points = [p.strip() for p in points if p.strip()]
    if not points:
        return text
    return "<ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>"

def predict_disease(img_path, lang="english"):
    img_array = prepare_image(img_path)
    preds = disease_model.predict(img_array)
    idx = int(np.argmax(preds))
    disease = disease_labels[idx]
    confidence = float(np.max(preds))

    # Fertilizer suggestion
    fertilizer = fertilizer_data.get(disease, "No fertilizer recommendation available")

    # AI-powered remedy & preventive tips
    ai_remedy, ai_prevention = "Not available", "Not available"
    try:
        messages = [
            {"role": "system",
             "content": f"You are a multilingual agricultural assistant. Reply only in {lang}. Keep tone simple and farmer-friendly."},
            {"role": "user",
             "content": f"Disease detected: {disease}. Give two separate sections:\n"
                        f"1. Remedies (numbered list)\n"
                        f"2. Preventive Tips (numbered list)."}
        ]
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7
        )
        ai_text = response.choices[0].message.content.strip()

        # Split by "Preventive" word
        if "Preventive" in ai_text:
            parts = ai_text.split("Preventive")
            ai_remedy = format_to_bullets(parts[0].strip())
            ai_prevention = format_to_bullets("Preventive " + parts[1].strip())
        else:
            ai_remedy = format_to_bullets(ai_text)
    except Exception as e:
        ai_remedy = f"⚠️ Error: {str(e)}"

    return disease, confidence, fertilizer, ai_remedy, ai_prevention

# =============================
# Routes
# =============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    lang = request.form.get("lang", "english")

    if not file or file.filename == "":
        return redirect(request.url)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    disease, confidence, fertilizer, ai_remedy, ai_prevention = predict_disease(filepath, lang)

    return render_template("result_disease.html",
                           filename=file.filename,
                           disease=disease,
                           confidence=confidence,
                           fertilizer=fertilizer,
                           ai_remedy=ai_remedy,
                           ai_prevention=ai_prevention,
                           lang=lang)

@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

# =============================
# Chatbot
# =============================
conversation = []
@app.route("/chat", methods=["GET", "POST"])
def chat():
    global conversation
    lang = "marathi"
    if request.method == "POST":
        user_message = request.form["message"]
        conversation.append({"sender": "user", "text": user_message})

        system_prompt = "तू एक शेतकरी सहाय्यक चॅटबॉट आहेस. फक्त मराठीत उत्तर दे."
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": "user" if msg["sender"]=="user" else "assistant", "content": msg["text"]} for msg in conversation]

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7
            )
            bot_reply = response.choices[0].message.content.strip()
        except Exception as e:
            bot_reply = f"⚠️ Groq API Error: {str(e)}"

        conversation.append({"sender": "bot", "text": bot_reply})
        return render_template("chat.html", conversation=conversation, lang=lang)

    return render_template("chat.html", conversation=conversation, lang=lang)

# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=True)
