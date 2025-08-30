import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import requests

# Import your existing app2.py as a module
import app2  

# Reuse the app from app2 instead of making a new one
app = app2.app  

# ---------------- Weather Feature ---------------- #
API_KEY = "82315474dfba22ddc9a9bc57f662876d"  # replace with your API key

@app.route("/weather", methods=["GET", "POST"])
def weather():
    if request.method == "POST":
        city = request.form["city"]
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url).json()

        if response.get("cod") != 200:
            return render_template("weather_result.html", weather=None, advisory=[])

        weather_data = {
            "city": response["name"],
            "temperature": response["main"]["temp"],
            "humidity": response["main"]["humidity"],
            "condition": response["weather"][0]["description"].title(),
        }

        advisory = []
        temp = weather_data["temperature"]
        humidity = weather_data["humidity"]

        if temp > 35:
            advisory.append("⚠️ High temperature! Ensure regular irrigation.")
        elif temp < 15:
            advisory.append("❄️ Low temperature! Protect plants from frost.")

        if humidity > 80:
            advisory.append("🌧️ High humidity! Watch out for fungal diseases.")
        elif humidity < 30:
            advisory.append("💧 Low humidity! Provide extra watering.")

        if not advisory:
            advisory.append("✅ Weather conditions are normal for crops.")

        return render_template("weather_result.html", weather=weather_data, advisory=advisory)

    return render_template("weather.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)  # ✅ single server
