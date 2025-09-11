import difflib
from deep_translator import GoogleTranslator

class SmartChatbot:
    def __init__(self, user_lang="en"):
        self.user_lang = user_lang  # Farmer’s chosen language

        # Knowledge Base
        self.responses = {
            "hello": "Hello! How can I help you with farming today?",
            "hi": "Hi! Do you want to ask about plant care or disease?",
            "bye": "Goodbye! Wishing you healthy crops 🌱",

            # General care
            "water": "Water your plants early morning or evening to reduce evaporation.",
            "fertilizer": "Use organic compost or balanced NPK fertilizer for healthy growth.",
            "pesticide": "Always use pesticides carefully. Try neem oil for organic pest control.",
            "soil": "Loamy soil with good drainage is best for most crops.",
            "sunlight": "Most plants need 6–8 hours of sunlight daily.",

            # Crops
            "tomato": "Tomato plants need 6–8 hours of sunlight and regular watering.",
            "potato": "Potatoes grow best in loose, well-drained soil.",
            "wheat": "Wheat requires cool weather and well-drained fertile soil.",
            "rice": "Rice needs standing water in fields and warm climate.",
            "onion": "Onions grow well in mild weather with loose soil.",
            "cotton": "Cotton requires a long frost-free period and black soil.",

            # Disease hints
            "leaf spot": "Leaf spot is often fungal. Remove infected leaves and spray fungicide.",
            "yellow leaves": "Yellowing can be due to nitrogen deficiency. Try adding urea or compost.",
            "powdery mildew": "Powdery mildew is fungal. Use sulfur-based fungicides or neem spray.",
            "root rot": "Root rot is caused by overwatering. Ensure good drainage and avoid waterlogging.",
            "aphids": "Aphids can be controlled with insecticidal soap or neem oil.",
            "whiteflies": "Whiteflies can be managed with yellow sticky traps and neem oil.",
            "blight": "Blight is a serious disease. Remove affected plants and use copper-based fungicides.",
        }

    def get_response(self, user_input):
        # Step 1: Translate farmer’s input → English
        query = GoogleTranslator(source=self.user_lang, target="en").translate(user_input).lower()

        # Step 2: Fuzzy match
        best_match = difflib.get_close_matches(query, self.responses.keys(), n=1, cutoff=0.6)
        if best_match:
            response = self.responses[best_match[0]]
        else:
            response = "Sorry, I don’t know the answer to that. Please try asking differently."

        # Step 3: Translate bot response → farmer’s chosen language
        final_response = GoogleTranslator(source="en", target=self.user_lang).translate(response)
        return final_response


if __name__ == "__main__":
    print("🌾 Farmer Chatbot (Multilingual + Fuzzy Matching). Type 'exit' to quit.")
    print("\nAvailable languages: en (English), hi (Hindi), mr (Marathi)")

    # Farmer selects language
    user_lang = input("Choose your language (default 'en'): ").strip().lower()
    if user_lang == "":
        user_lang = "en"

    bot = SmartChatbot(user_lang)

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye! 👋")
            break

        reply = bot.get_response(user_input)
        print("Bot:", reply)
