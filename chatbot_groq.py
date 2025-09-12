import os
import requests

class SmartChatbot:
    def __init__(self, user_lang="en"):
        self.user_lang = user_lang
        # ✅ Fix: fetch API key properly from env, not hardcoded
        self.api_key = os.getenv("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")
        self.api_url = "https://api.groq.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY not found. Please set it in your environment or .env file.")

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
            "model": "mixtral-8x7b-32768",  # Example Groq model
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.4
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            data = response.json()

            # Extract assistant reply
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"⚠️ Error: {str(e)}"
