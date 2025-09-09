import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# ✅ Load model
model = tf.keras.models.load_model("best_disease_model.h5")

# ✅ Load validation generator (make sure to use same preprocessing as in training)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    "disease_dataset_split/test",   # ✅ correct path
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)


# ✅ Predictions
val_generator.reset()
pred_probs = model.predict(val_generator, verbose=1)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# ✅ Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Disease Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")   # ✅ Save PNG
plt.show()

# ✅ Classification Report
report = classification_report(true_classes, pred_classes,
                               target_names=class_labels, output_dict=True)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv("classification_report.csv", index=True)   # ✅ Save CSV
print("\nClassification Report saved as classification_report.csv\n")
print(report_df)

# ✅ Per-class Accuracy
cm_diag = np.diag(cm)
per_class_acc = cm_diag / cm.sum(axis=1)

acc_df = pd.DataFrame({
    "Class": class_labels,
    "Accuracy": per_class_acc * 100
})
acc_df.to_csv("per_class_accuracy.csv", index=False)   # ✅ Save per-class accuracy
print("\nPer-class accuracy saved as per_class_accuracy.csv\n")
print(acc_df)
