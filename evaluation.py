# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# =============================
# LOAD MODEL AND TEST DATA
# =============================
model = load_model("best_disease_model.h5")   # 👈 change name if different

test_dir = "disease_dataset_split/test"              # 👈 adjust path as per your folder
img_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# =============================
# CONFUSION MATRIX + METRICS
# =============================
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
labels = list(test_generator.class_indices.keys())

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")   # Save to file
plt.show()

# ✅ Classification Report
report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("performance_metrics.csv", index=True)
print("✅ Saved 'confusion_matrix.png' and 'performance_metrics.csv'")

# ✅ Print summary
print("\nPerformance Summary:")
print(report_df.round(3).head(len(labels)))
