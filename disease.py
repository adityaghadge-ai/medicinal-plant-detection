import os
import json

dataset_path = "dataset_split/train"
output_json = "models/disease_labels.json"

classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
class_dict = {cls_name: idx for idx, cls_name in enumerate(classes)}

with open(output_json, "w") as f:
    json.dump(class_dict, f, indent=4)

print(f"✅ Disease classes saved: {class_dict}")
