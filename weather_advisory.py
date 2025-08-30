import requests
import os

# 🔑 Get your API key from: https://home.openweathermap.org/api_keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "82315474dfba22ddc9a9bc57f662876d")

def get_weather(city: str, country_code: str = ""):
    """Fetch current weather data from OpenWeatherMap API"""
    query = f"{city},{country_code}" if country_code else city
    url = f"http://api.openweathermap.org/data/2.5/weather?q={query}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

    data = response.json()
    if data.get("cod") != 200:
        return {"error": data.get("message", "Unknown error")}

    return {
        "city": data.get("name", city),
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"]
    }


def crop_advisory(weather_info: dict):
    """Give simple crop advisory based on weather conditions"""
    if "error" in weather_info:
        return [weather_info["error"]]

    temp = weather_info["temperature"]
    humidity = weather_info["humidity"]
    condition = weather_info["condition"]

    advice = []

    # Temperature-based advice
    if temp > 35:
        advice.append("⚠️ High temperature detected! Ensure proper irrigation.")
    elif temp < 15:
        advice.append("❄️ Low temperature! Protect sensitive crops from frost.")

    # Humidity-based advice
    if humidity > 80:
        advice.append("🌧️ High humidity may cause fungal diseases. Monitor closely.")
    elif humidity < 30:
        advice.append("💧 Soil may dry quickly. Irrigation needed.")

    # Weather condition-based advice
    if "rain" in condition.lower():
        advice.append("🌦️ Rain expected. Avoid excess irrigation and apply fertilizer after rainfall.")
    elif "clear" in condition.lower():
        advice.append("☀️ Clear weather is good for most crops. Continue regular monitoring.")

    if not advice:
        advice.append("✅ Weather is favorable for crops today.")

    return advice


if __name__ == "__main__":
    city = input("Enter your city: ")
    weather = get_weather(city)
    print("Weather Info:", weather)
    print("Crop Advisory:")
    for line in crop_advisory(weather):
        print("-", line)
