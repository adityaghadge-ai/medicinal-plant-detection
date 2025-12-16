🌿 Medicinal Plant Disease Detection & AI Farmer Assistant

An AI-powered smart agriculture system for medicinal plant disease detection, fertilizer guidance, weather insights, and farmer-friendly multilingual assistance.

This project combines Deep Learning, Computer Vision, and Generative AI to support precision agriculture and medicinal plant health monitoring.

🚀 Key Features
🔍 Disease Detection (Deep Learning)

MobileNetV2-based CNN (Transfer Learning + Fine-tuning)

Detects healthy vs diseased medicinal plant leaves

Confidence score with prediction

Trained on AI-MedLeafX (2025) dataset

🌱 AI-Powered Remedies & Prevention

Automatic fertilizer recommendations

AI-generated remedies & preventive measures

Farmer-friendly language (English / Marathi)

🌦 Weather Intelligence

Real-time weather via OpenWeather API

AI-based agricultural alerts & cultivation tips

📅 72-Hour Farming Action Planner (Novel Feature)

Crop-stage–aware action plan

Risk assessment + scheduled tasks

Market & weather-aware decisions

💬 Multilingual Farmer Chatbot

Marathi voice + text support

Speech-to-text & text-to-speech

Groq LLM–powered responses


🧠 Tech Stack

Deep Learning: TensorFlow / Keras (MobileNetV2)

Backend: Flask (Python)

Frontend: HTML, Bootstrap, JavaScript

AI APIs: Groq (LLMs), OpenWeather

Computer Vision: OpenCV

Deployment Ready: Modular & scalable



📂 Project Structure
medicinal-plant-detection/
│── dataset/                  # Original dataset (not pushed to GitHub)
│── dataset_split/            # Train / Val / Test split
│── disease_dataset_split/    # Disease-wise organized dataset
│── models/                   # JSON configs (labels, fertilizers)
│── static/uploads/           # Runtime uploads
│── templates/                # Flask HTML templates
│── test_samples/             # Sample images
│── train.py                  # Model training
│── app.py                    # Flask web app
│── requirements.txt
│── .gitignore
│── README.md



📊 Dataset Used

AI-MedLeafX: A Large-Scale Computer Vision Dataset for Medicinal Plant Diagnosis (2025)

10,858 original images

65,178 augmented images

4 medicinal plant species

Multiple disease categories

📄 DOI: 10.17632/zz7r5y4dc6.1


⚙️ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/adityaghadge-ai/medicinal-plant-detection.git
cd medicinal-plant-detection

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

3️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Train the Model
python train.py

6️⃣ Run Web App
python app.py


Open browser:
👉 http://127.0.0.1:5000


📌 Future Improvements

🔹 Increase accuracy beyond 99%

🔹 Add fertilizer prediction model

🔹 Deploy on cloud (AWS / Streamlit)

🔹 Mobile app integration

🔹 Edge deployment for farmers


🤝 Collaboration

Code is open-source on GitHub

Large datasets shared separately (Drive / OneDrive)

Contributions welcome via Pull Requests

