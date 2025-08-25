from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os, json

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

# ----------------------------
# Data generators
# ----------------------------
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    "dataset_split/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    "dataset_split/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Save class indices
os.makedirs("models", exist_ok=True)
with open("models/class_labels.json", "w") as f:
    json.dump(train_generator.class_indices, f, indent=4)

# ----------------------------
# Build model (MobileNetV2)
# ----------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # freeze layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ----------------------------
# Training
# ----------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Save model
model.save("models/medicinal_mobilenet.h5")
print("✅ MobileNetV2 model saved!")

# ----------------------------
# Evaluation (using validation set as test for now)
# ----------------------------
print("\n🔍 Running evaluation on validation set (temporary test)...")

test_loss, test_acc = model.evaluate(val_generator, verbose=1)

print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")
print("✅ Evaluation complete!")