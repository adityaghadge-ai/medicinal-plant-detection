import os
import requests

class SmartChatbot:
    def __init__(self, user_lang="en"):
        self.user_lang = user_lang
        self.api_key = os.getenv("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")  # Set in system environment
        self.api_url = "https://api.groq.com/v1/chat/completions"

    def get_response(self, user_message: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # System prompt to make bot domain-specific
        system_prompt = f"""
        You are an expert farming assistant chatbot.
        Always reply in {self.user_lang} language.
        Provide answers about irrigation, fertilizer, crop disease management,
        and remedies for farmers in simple terms.
        """

        payload = {
            "model": "mixtral-8x7b-32768",  # example Groq model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.4
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ Error: {e}"
