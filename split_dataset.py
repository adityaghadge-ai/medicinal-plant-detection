import os
import shutil
import random

# ✅ Correct path to your dataset
src_dir = r"D:\ALL PROJECTS\medicinal-plant-detection\Resized Image"
dest_dir = r"D:\ALL PROJECTS\medicinal-plant-detection\dataset_split"

# Split ratios
train_split = 0.7
val_split = 0.2
test_split = 0.1

# Create destination folders
for split in ["train", "val", "test"]:
    split_path = os.path.join(dest_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Loop through each plant folder
for plant in os.listdir(src_dir):
    plant_path = os.path.join(src_dir, plant)
    if not os.path.isdir(plant_path):
        continue

    # Loop through disease categories inside each plant
    for disease in os.listdir(plant_path):
        disease_path = os.path.join(plant_path, disease)
        if not os.path.isdir(disease_path):
            continue

        # Create subfolders in train/val/test
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(dest_dir, split, plant, disease), exist_ok=True)

        # Get all images for this disease
        images = os.listdir(disease_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(train_split * total)
        val_end = train_end + int(val_split * total)

        # Assign splits
        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        # Copy files
        for img in train_imgs:
            shutil.copy(os.path.join(disease_path, img), os.path.join(dest_dir, "train", plant, disease, img))

        for img in val_imgs:
            shutil.copy(os.path.join(disease_path, img), os.path.join(dest_dir, "val", plant, disease, img))

        for img in test_imgs:
            shutil.copy(os.path.join(disease_path, img), os.path.join(dest_dir, "test", plant, disease, img))

print("✅ Dataset split completed! Saved in 'dataset_split/'")
