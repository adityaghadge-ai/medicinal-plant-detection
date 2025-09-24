import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your plant dataset
PLANT_DATASET_DIR = "dataset_split/train"   # adjust if different
OUTPUT_JSON = "models/plant_labels.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    PLANT_DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Save mapping
plant_labels = train_generator.class_indices
with open(OUTPUT_JSON, "w") as f:
    json.dump(plant_labels, f, indent=4)

print(f"✅ plant_labels.json saved: {plant_labels}")
