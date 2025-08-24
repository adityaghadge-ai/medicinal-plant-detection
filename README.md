🌿 Medicinal Plant Detection

This project detects medicinal plants from images using deep learning.  
Built with **Python, TensorFlow/Keras, Flask** for web deployment.

---

## 📂 Project Structure
medicinal-plant-detection/
│── dataset/ # Training dataset (ignored in git, share separately)
│── dataset_split/ # Train/test split (generated)
│── models/ # Saved models & class_labels.json
│── static/uploads/ # Uploaded images (runtime only)
│── templates/ # Flask HTML templates
│── test_images/ # Test images (ignored in git)
│── test_samples/ # Sample images for testing
│── app.py # Flask web app
│── train.py # Model training script
│── predict.py # Script for inference
│── requirements.txt # Dependencies
│── .gitignore

## ⚡ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/adityaghadge-ai/medicinal-plant-detection.git
cd medicinal-plant-detection
2. Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
3. Install Dependencies
pip install -r requirements.txt

4. Add Dataset

⚠️ Dataset is not included in the repo (too large).
Ask the project owner to share dataset/, test_images/, etc.
Place them inside the project root folder:

medicinal-plant-detection/
    ├── dataset/
    ├── test_images/
    ├── ...

5. Train the Model
python train.py

6. Run Prediction
python predict.py --image test_samples/40.jpg

7. Run Web App
python app.py


Then open http://127.0.0.1:5000
 in your browser.

🤝 Collaboration

Code is shared via GitHub.

Datasets should be shared separately (Google Drive/OneDrive/Pen drive).

Contributions are welcome via pull requests.

📝 To Do

 Improve dataset quality

 Add model evaluation metrics

 Deploy on cloud (Heroku/Streamlit)
