import os
import json

def create_labels_json(dataset_path, output_json):
    """
    Generate JSON file with class labels from folder structure.
    Each subfolder = one class.
    """
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

    with open(output_json, "w") as f:
        json.dump(class_dict, f, indent=4)

    print(f"[INFO] Saved {len(classes)} classes to {output_json}")


if __name__ == "__main__":
    # 🌿 Plant dataset labels
    plant_dataset_path = "dataset_split/train"       # adjust if needed
    plant_output_json = "models/class_labels.json"

    # 🦠 Disease dataset labels
    disease_dataset_path = "disease_dataset_split/train"
    disease_output_json = "models/disease_labels.json"

    # Generate both
    create_labels_json(plant_dataset_path, plant_output_json)
    create_labels_json(disease_dataset_path, disease_output_json)

    print("\n✅ All label JSON files generated successfully!")
