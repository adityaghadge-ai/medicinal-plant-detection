import os
import shutil
import random

# Paths
DATASET_DIR = "disease_dataset/color"   # your source dataset
OUTPUT_DIR = "disease_dataset_split"    # where split dataset will go

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output folders
for split in ["train", "val", "test"]:
    split_path = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_path, exist_ok=True)

# Loop through each class folder
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    # Split indexes
    train_end = int(train_ratio * len(images))
    val_end = train_end + int(val_ratio * len(images))

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Function to copy files
    def copy_files(files, split):
        split_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)
        for f in files:
            src = os.path.join(class_path, f)
            dst = os.path.join(split_class_dir, f)
            shutil.copy(src, dst)

    # Copy images
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

print("✅ Dataset successfully split into train/val/test in", OUTPUT_DIR)
