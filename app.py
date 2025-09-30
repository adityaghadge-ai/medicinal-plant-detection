from dotenv import load_dotenv
import os, json, re, requests
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from groq import Groq

# =============================
# Load environment variables
# =============================
# 🚨 Using dotenv_path='a.env' based on your file structure
load_dotenv(dotenv_path='a.env') 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not set.")
client = Groq(api_key=GROQ_API_KEY)

# ✅ Weather API key
OPENWEATHER_API = os.getenv("OPENWEATHER_API")
if not OPENWEATHER_API:
    print("⚠️ Warning: OPENWEATHER_API key not set in .env")

# =============================
# Config
# =============================
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (224, 224)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =============================
# Custom Jinja Filter (FIX FOR 'nl2br' ERROR)
# =============================
def nl2br_filter(s):
    """Converts newlines to HTML breaks."""
    if s is not None:
        # Also convert common numbered list patterns to unordered list for better formatting
        s = re.sub(r'(\d+\.\s+)', r'\n- ', s).strip()
        # Replace remaining newlines with HTML breaks
        return s.replace('\n', '<br>')
    return ''

app.jinja_env.filters['nl2br'] = nl2br_filter
# =============================

# =============================
# Load Models
# =============================
try:
    disease_model = load_model("best_disease_model.h5")
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

    fertilizer = fertilizer_data.get(disease, "No fertilizer recommendation available")

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

        if "Preventive" in ai_text or "प्रतिबंधक" in ai_text:
            split_keyword = "Preventive" if "Preventive" in ai_text else "प्रतिबंधक"
            parts = re.split(f"({split_keyword})", ai_text, 1, re.IGNORECASE)
            if len(parts) > 1:
                ai_remedy = format_to_bullets(parts[0].strip())
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


# New function to get AI-powered weather insights
def get_weather_insights(weather_data):
    if not GROQ_API_KEY:
        return "AI insights not available. GROQ_API_KEY not set."
    
    prompt = f"""
    You are an AI-powered agricultural weather assistant.
    Analyze the following weather data for a city in India and provide a farmer-friendly summary, an alert, and a couple of cultivation tips.
    The response must contain three distinct sections, clearly labeled:
    1. SUMMARY: (A brief, positive opening statement about the current weather.)
    2. ALERT: (A specific warning or caution for farmers based on the data, e.g., 'High humidity increases fungal risk,' or 'Prepare for light rainfall.')
    3. TIPS: (A short, numbered list of 2-3 actionable cultivation recommendations.)

    Weather Data:
    City: {weather_data['city']}
    Temperature: {weather_data['temperature']}°C (Feels like: {weather_data['feels_like']}°C)
    Humidity: {weather_data['humidity']}%
    Description: {weather_data['description']}
    Wind Speed: {weather_data['wind_speed']} m/s
    """

    try:
        messages = [
            {"role": "system", "content": "You are a farmer-friendly AI weather assistant. Keep the language simple."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error fetching AI weather insights: {str(e)}"


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
conversation = [] 

@app.route("/chat", methods=["GET", "POST"])
def chat():
    global conversation
    lang = "marathi"
    if request.method == "POST":
        user_message = request.form["message"]
        
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
        return render_template("chat.html", conversation=conversation, lang=lang)

    return render_template("chat.html", conversation=conversation, lang=lang)

# =============================
# 🌦 Weather Route (Updated for AI)
# =============================
@app.route("/weather", methods=["GET", "POST"]) 
def weather():
    if request.method == "POST":
        city = request.form.get("city")
    else: # Default city for GET request or initial load
        city = request.args.get("city", "Pune")
        
    if not OPENWEATHER_API:
        return render_template("weather.html", error="OPENWEATHER_API key not set in .env")

    # API call to OpenWeatherMap
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},in&appid={OPENWEATHER_API}&units=metric"
    
    try:
        res = requests.get(url)
        data = res.json()
        
        if data.get("cod") != 200:
            error_message = data.get("message", "Unable to fetch weather data. Check city name.")
            return render_template("weather.html", error=error_message)
        
        # Extract basic weather info
        weather_info = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
            "pressure": data["main"]["pressure"]
        }
        
        # Get AI Insights
        ai_insights = get_weather_insights(weather_info)

        # Render the results
        return render_template("weather.html", 
                               weather=weather_info, 
                               ai_insights=ai_insights)
    
    except Exception as e:
        # Catch unexpected errors during the process
        return render_template("weather.html", error=f"An unexpected error occurred: {str(e)}")

# =============================
# Run App
# =============================
if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)