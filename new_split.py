import os
import shutil

# Paths
dataset_split = "dataset_split"           # original dataset
disease_dataset = "disease_dataset_split"  # new dataset for disease training

# Loop over splits
for split in ["train", "val", "test"]:
    for plant in os.listdir(os.path.join(dataset_split, split)):
        plant_path = os.path.join(dataset_split, split, plant)
        if not os.path.isdir(plant_path):
            continue
        
        # Each disease folder inside plant
        for disease in os.listdir(plant_path):
            disease_path = os.path.join(plant_path, disease)
            if not os.path.isdir(disease_path):
                continue
            
            # Destination folder for this disease
            dest_folder = os.path.join(disease_dataset, split, disease)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy all images to the new disease folder
            for img_file in os.listdir(disease_path):
                shutil.copy(os.path.join(disease_path, img_file),
                            os.path.join(dest_folder, img_file))

print("✅ Dataset reorganized by disease successfully!")
