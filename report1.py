import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_PATH = "best_disease_model.h5"
LABELS_PATH = "models/disease_labels.json"
TEST_DIR = "disease_dataset_split/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32


model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = {int(v): k for k, v in json.load(f).items()}

class_names = list(labels.values())

# ============================
# Test Data Generator
# ============================
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ============================
# Predictions
# ============================
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=1)

# ============================
# Classification Report
# ============================
report = classification_report(
    y_true, y_pred,
    target_names=class_names,
    output_dict=True
)

# Save report.json
with open("report.json", "w") as f:
    json.dump(report, f, indent=4)

print("✅ Classification report saved as report.json")

# ============================
# Confusion Matrix
# ============================
cm = confusion_matrix(y_true, y_pred)

# Save confusion matrix also in JSON
cm_dict = {
    "labels": class_names,
    "matrix": cm.tolist()
}

with open("confusion_matrix.json", "w") as f:
    json.dump(cm_dict, f, indent=4)

print("✅ Confusion matrix saved as confusion_matrix.json")

# ============================
# Optional: Plot and Save CM
# ============================
plt.figure(figsize=(10,8))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=90)
plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix1.png")
print("✅ Confusion matrix image saved as confusion_matrix1.png")
