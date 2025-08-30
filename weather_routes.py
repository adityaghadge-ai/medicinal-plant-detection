# weather_routes.py
import os
import requests
from flask import Blueprint, render_template, request

weather_bp = Blueprint("weather", __name__)

# ---------------- Config ---------------- #
API_KEY = os.getenv("OPENWEATHER_API_KEY", "82315474dfba22ddc9a9bc57f662876d")  # keep env var for safety
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# ---------------- Helpers ---------------- #
def get_weather(city):
    """Fetch weather info for a given city using OpenWeather API."""
    try:
        url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        return {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"]
        }
    except Exception as e:
        return {"error": str(e)}

def generate_advisory(weather_info):
    """Generate simple crop/plant advisory from weather conditions."""
    advisories = []
    if "error" in weather_info:
        advisories.append(weather_info["error"])
        return advisories

    temp = weather_info["temperature"]
    humidity = weather_info["humidity"]
    cond = weather_info["condition"].lower()

    if temp > 35:
        advisories.append("🔥 High temperature may stress crops, ensure proper irrigation.")
    elif temp < 15:
        advisories.append("❄️ Low temperature detected, protect plants from frost.")

    if humidity > 80:
        advisories.append("🌧️ High humidity may cause fungal diseases. Monitor closely.")
    elif humidity < 30:
        advisories.append("💧 Very low humidity, risk of dehydration in plants.")

    if "rain" in cond:
        advisories.append("☔ Rain expected, adjust irrigation accordingly.")
    elif "cloud" in cond:
        advisories.append("⛅ Cloudy weather may reduce photosynthesis.")

    if not advisories:
        advisories.append("✅ Weather conditions are favorable for crops.")

    return advisories

# ---------------- Routes ---------------- #
@weather_bp.route("/weather", methods=["GET", "POST"])
def weather():
    if request.method == "POST":
        city = request.form.get("city")
        if not city:
            return render_template("weather.html", error="Please enter a city name.")

        weather_info = get_weather(city)
        advisories = generate_advisory(weather_info)
        return render_template("weather_result.html", weather=weather_info, advisories=advisories)

    return render_template("weather.html")
