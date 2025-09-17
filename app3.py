from dotenv import load_dotenv
import os, json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from groq import Groq

# ===============================
# Load environment variables
# ===============================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set. Please check your .env file.")
print("✅ GROQ_API_KEY found.")

client = Groq(api_key=GROQ_API_KEY)

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
# Plant model
plant_model = load_model("plant_model.h5")

# Disease model
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

with open("models/fertilizers.json") as f:
    fertilizers = json.load(f)

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

def predict_disease(img_path, lang="english"):
    img_array = prepare_image(img_path)
    preds = disease_model.predict(img_array)
    idx = np.argmax(preds)
    disease = disease_labels[idx]
    confidence = float(np.max(preds))
    remedy = remedies.get(disease, "No remedy available.")
    fertilizer = fertilizers.get(disease, "No fertilizer recommendation available.")

    # ✅ AI-powered remedy + preventive tips
    ai_remedy = "AI remedy not available."
    ai_prevention = "AI preventive tips not available."

    try:
        messages = [
            {
                "role": "system",
                "content": f"""
You are a multilingual agricultural assistant.
Reply only in {lang}. 
Keep the tone simple, practical, and farmer-friendly.
Provide two sections:
1. **Remedy** → step-by-step treatment for the disease.
2. **Preventive Tips** → how farmers can avoid this disease in the future.
Keep it short (5-7 lines max each).
"""
            },
            {
                "role": "user",
                "content": f"Disease detected: {disease}. Suggest remedies and preventive tips for farmers."
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7
        )
        ai_text = response.choices[0].message.content.strip()

        # Split remedy & preventive tips
        if "Preventive" in ai_text:
            parts = ai_text.split("Preventive")
            ai_remedy = parts[0].strip()
            ai_prevention = "Preventive" + parts[1].strip()
        else:
            ai_remedy = ai_text

    except Exception as e:
        ai_remedy = f"⚠️ Error fetching AI remedy: {str(e)}"

    return disease, confidence, remedy, fertilizer, ai_remedy, ai_prevention

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
    lang = request.form.get("lang", "english")  # ✅ pass language

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
            disease, confidence, static_remedy, fertilizer, ai_remedy, ai_prevention = predict_disease(filepath, lang)
            return render_template("result_disease.html",
                                   filename=file.filename,
                                   disease=disease,
                                   confidence=confidence,
                                   static_remedy=static_remedy,
                                   fertilizer=fertilizer,
                                   ai_remedy=ai_remedy,
                                   ai_prevention=ai_prevention,
                                   lang=lang)

# ===============================
# Image Display Route
# ===============================
@app.route("/display/<filename>")
def display_image(filename):
    return redirect(url_for("static", filename="uploads/" + filename))

# ===============================
# Chatbot Integration (Groq)
# ===============================
conversation = []

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global conversation

    if request.method == "POST":
        user_message = request.form["message"]
        lang = request.form["lang"]

        # Add user message
        conversation.append({"sender": "user", "text": user_message})

        # System prompt
        system_prompt = f"""
You are a multilingual farming assistant.

STRICT RULES:
- Always reply in {lang}.
- If {lang} is "marathi", reply in मराठी (देवनागरी लिपी).
- If {lang} is "hindi", reply in हिन्दी (देवनागरी लिपी).
- If {lang} is "english", reply in plain English.
- Never mix English unless for common agri terms (e.g., NPK, fertilizer).

Guidelines:
- Natural, conversational tone (like ChatGPT/DeepSeek/Grok).
- Keep answers short: 3–5 sentences max.
- Do not repeat the user’s question.
- Farmer-friendly, simple advice.
- Use bullet points only when helpful.
- Always supportive and clear.
"""

        messages = [
            {"role": "system", "content": system_prompt},
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
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.7
            )
            bot_reply = response.choices[0].message.content.strip()
        except Exception as e:
            bot_reply = f"⚠️ Error connecting to Groq API: {str(e)}"

        # Save bot reply
        conversation.append({"sender": "bot", "text": bot_reply})

        return render_template("chat.html", conversation=conversation, lang=lang)

    return render_template("chat.html", conversation=conversation, lang=request.args.get("lang", "en"))


if __name__ == "__main__":
    app.run(debug=True)
