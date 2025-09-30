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
# Assuming these files exist in your project structure
try:
    disease_model = load_model("best_disease_model.h5")  # retrained disease model
except Exception as e:
    print(f"Warning: Could not load disease model: {e}")
    disease_model = None

# =============================
# Load Label JSONs
# =============================
try:
    with open("models/disease_labels.json") as f:
        disease_labels = {int(v): k for k, v in json.load(f).items()}
except Exception as e:
    print(f"Warning: Could not load disease labels: {e}")
    disease_labels = {}

try:
    with open("models/fertilizers.json") as f:
        fertilizer_data = json.load(f)
except Exception as e:
    print(f"Warning: Could not load fertilizers data: {e}")
    fertilizer_data = {}

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
    if not disease_model:
        return "Model Error", 0.0, "Model not loaded.", "Error: Disease model not loaded.", "Error: Disease model not loaded."
        
    img_array = prepare_image(img_path)
    preds = disease_model.predict(img_array)
    idx = int(np.argmax(preds))
    disease = disease_labels.get(idx, "Unknown Disease")
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
            # Handle both English and Marathi split (मराठीत 'प्रतिबंधक')
            split_keyword = "Preventive" if "Preventive" in ai_text else "प्रतिबंधक"
            parts = re.split(f"({split_keyword})", ai_text, 1, re.IGNORECASE)
            
            if len(parts) > 1:
                ai_remedy = format_to_bullets(parts[0].strip())
                # Re-add the keyword for formatting
                prevention_text = parts[1] + parts[2].strip() if len(parts) > 2 else parts[1]
                ai_prevention = format_to_bullets(prevention_text.strip())
            else:
                ai_remedy = format_to_bullets(ai_text)
                
        else:
            ai_remedy = format_to_bullets(ai_text)
            
    except Exception as e:
        ai_remedy = f"⚠️ Error fetching AI response: {str(e)}"
        ai_prevention = f"⚠️ Error fetching AI response: {str(e)}"


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
    # Ensure UPLOAD_FOLDER exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
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
# Global conversation state
conversation = [] 

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global conversation
    lang = "marathi"
    if request.method == "POST":
        user_message = request.form["message"]
        
        # Clear conversation if user types "clear" or similar
        if user_message.lower() in ["clear", "reset", "थांबा"]:
            conversation = []
            return render_template("chat.html", conversation=conversation, lang=lang)
            
        conversation.append({"sender": "user", "text": user_message})

        system_prompt = "तू एक शेतकरी सहाय्यक चॅटबॉट आहेस. फक्त मराठीत उत्तर दे. आपले उत्तर थोडक्यात आणि मुद्देसूद असावे."
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
            bot_reply = f"⚠️ Groq API Error: {str(e)}. कृपया पुन्हा प्रयत्न करा."

        conversation.append({"sender": "bot", "text": bot_reply})
        # The key is to pass the conversation state to the template
        return render_template("chat.html", conversation=conversation, lang=lang)

    # Initial GET request
    return render_template("chat.html", conversation=conversation, lang=lang)

# =============================
# Run App
# =============================
if __name__ == "__main__":
    # Ensure UPLOAD_FOLDER exists before running
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)