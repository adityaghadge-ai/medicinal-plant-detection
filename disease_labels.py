import os
import json

def create_labels_json(dataset_path, output_json):
    """
    Generate JSON file with class labels from folder structure.
    Each subfolder = one class (e.g., Camphora_Healthy, Indica_BacterialSpot).
    """
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset path not found: {dataset_path}")
        return

    # Get sorted subfolder names = class labels
    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    class_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Save JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(class_dict, f, ensure_ascii=False, indent=4)

    print(f"[INFO] Saved {len(classes)} classes to {output_json}")


if __name__ == "__main__":
    # Path to your training dataset
    dataset_path = "dataset_split/train"   # adjust if needed
    output_json = "models/class_labels.json"

    # Generate JSON
    create_labels_json(dataset_path, output_json)

    print("\n✅ Label JSON file generated successfully!")
