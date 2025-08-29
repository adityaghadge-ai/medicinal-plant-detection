import json
import os

# Path to disease labels
disease_labels_path = "models/disease_labels.json"
# Path to save remedies.json inside models folder
remedies_path = "models/remedies.json"

# Load disease labels
with open(disease_labels_path, "r") as f:
    disease_labels = json.load(f)

# Create remedies dictionary with empty strings as placeholders
remedies = {disease: "" for disease in disease_labels.keys()}

# Save remedies.json inside models folder
with open(remedies_path, "w") as f:
    json.dump(remedies, f, indent=4)

print(f"✅ remedies.json created in {remedies_path}")
