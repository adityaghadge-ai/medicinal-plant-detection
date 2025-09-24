import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os, json, cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Paths
# =============================
BASE_DIR = "disease_dataset_split"   # new dataset
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")
JSON_DIR = "models"                   # save JSONs here
os.makedirs(JSON_DIR, exist_ok=True)

# =============================
# Parameters
# =============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
WARMUP_EPOCHS = 10
FINE_TUNE_EPOCHS = 40
LR = 1e-4

# =============================
# Data Augmentation + Generators
# =============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# =============================
# Save disease class indices to JSON
# =============================
disease_labels = train_generator.class_indices
with open(os.path.join(JSON_DIR, "disease_labels.json"), "w") as f:
    json.dump(disease_labels, f, indent=4)
print(f"✅ Disease classes saved: {disease_labels}")

# =============================
# Model (MobileNetV2 Transfer Learning)
# =============================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # frozen for warmup

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=LR), loss="categorical_crossentropy", metrics=["accuracy"])

# =============================
# Callbacks
# =============================
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    ModelCheckpoint("best_disease_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
]

# =============================
# Stage 1: Warmup Training
# =============================
print("\n🔹 Stage 1: Warmup Training (frozen base)...")
history = model.fit(
    train_generator,
    epochs=WARMUP_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# =============================
# Stage 2: Fine-Tuning last 50 layers
# =============================
print("\n🔹 Stage 2: Fine-Tuning last 50 layers...")
for layer in base_model.layers[-50:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

history_fine = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks
)

# =============================
# Combine histories for plotting
# =============================
def combine_history(h1, h2):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]
    return {"accuracy": acc, "val_accuracy": val_acc, "loss": loss, "val_loss": val_loss}

full_history = combine_history(history, history_fine)

# =============================
# Plot Training Curves
# =============================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(full_history["accuracy"], label="Train Accuracy")
plt.plot(full_history["val_accuracy"], label="Val Accuracy")
plt.axvline(x=WARMUP_EPOCHS, color="red", linestyle="--", label="Fine-tune Start")
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(full_history["loss"], label="Train Loss")
plt.plot(full_history["val_loss"], label="Val Loss")
plt.axvline(x=WARMUP_EPOCHS, color="red", linestyle="--", label="Fine-tune Start")
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# =============================
# Evaluate on Test Set
# =============================
loss, acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# =============================
# Save Final Model
# =============================
model.save("final_disease_model.h5")
print("💾 Final fine-tuned model saved as final1_disease_model.h5")
