import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================
# CONFIG
# ============================
MODEL_PATH = "best_disease_model.h5"       # your trained model
LABELS_PATH = "models/disease_labels.json" # json file with class indices
TEST_DIR = "disease_dataset_split/test"          # test dataset folder
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ============================
# LOAD MODEL & LABELS
# ============================
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

# reverse mapping: idx -> class name
idx_to_class = {v: k for k, v in class_indices.items()}

# ============================
# DATA GENERATOR FOR TEST
# ============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ============================
# PREDICTIONS
# ============================
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================
# CLASSIFICATION REPORT
# ============================
report = classification_report(
    y_true, y_pred,
    target_names=[idx_to_class[i] for i in range(len(idx_to_class))],
    output_dict=True
)

# Save as CSV
import pandas as pd
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("classification_report.csv", index=True)
print("✅ Classification report saved as classification_report.csv")

# ============================
# CONFUSION MATRIX
# ============================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[idx_to_class[i] for i in range(len(idx_to_class))],
            yticklabels=[idx_to_class[i] for i in range(len(idx_to_class))])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("✅ Confusion matrix saved as confusion_matrix.png")
plt.show()
