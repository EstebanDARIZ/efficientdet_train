import os
import json
import shutil
import random
from pathlib import Path
from PIL import Image

DATASET_ROOT = "/home/esteban-dreau-darizcuren/doctorat/dataset/dataset_efficientdet"  # dossier principal contenant images/ et labels/
OUTPUT_ROOT = "/home/esteban-dreau-darizcuren/doctorat/dataset/dataset_coco"  # dossier COCO final
TRAIN_SPLIT = 0.8  # train = 80%, val = 20%

out = Path(OUTPUT_ROOT)
(out / "annotations").mkdir(parents=True, exist_ok=True)
(out / "train2017").mkdir(parents=True, exist_ok=True)
(out / "val2017").mkdir(parents=True, exist_ok=True)

images_dir = Path(DATASET_ROOT) / "images"
labels_dir = Path(DATASET_ROOT) / "labels"

image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

random.shuffle(image_files)
n_train = int(len(image_files) * TRAIN_SPLIT)
train_files = image_files[:n_train]
val_files = image_files[n_train:]

print(f"Total images: {len(image_files)}")
print(f"Train: {len(train_files)}, Val: {len(val_files)}")

def coco_structure():
    return {
        "info": {
            "description": "Custom Dataset",
            "version": "1.0"
        },
        "licenses": [
            {"id": 1, "name": "none", "url": ""}
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }


coco_train = coco_structure()
coco_val = coco_structure()

# Ajout de catégories (id = 0 → num_classes-1)
# Tu as 6 classes, donc :
NUM_CLASSES = 6
for cid in range(NUM_CLASSES):
    coco_train["categories"].append({"id": cid, "name": f"class_{cid}"})
    coco_val["categories"].append({"id": cid, "name": f"class_{cid}"})

ann_id = 1
img_id_counter = 1  # nouveau compteur d'IDs

def process_image(img_path, coco_dict, target_folder):
    global ann_id
    global img_id_counter

    # Copy image
    out_path = Path(OUTPUT_ROOT) / target_folder / img_path.name
    shutil.copy(img_path, out_path)

    # Read image size
    img = Image.open(img_path)
    width, height = img.size

    # Create unique COCO image ID
    img_id = img_id_counter
    img_id_counter += 1

    # Add image entry
    coco_dict["images"].append({
        "id": img_id,
        "file_name": img_path.name,
        "width": width,
        "height": height
    })

    # Labels
    txt_path = labels_dir / f"{img_path.stem}.txt"
    if not txt_path.exists():
        return

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, xmin, ymin, xmax, ymax = map(float, parts)
            w = xmax - xmin
            h = ymax - ymin

            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1



print("Processing train set...")
for img_path in train_files:
    process_image(img_path, coco_train, "train2017")

print("Processing val set...")
for img_path in val_files:
    process_image(img_path, coco_val, "val2017")

with open(out / "annotations" / "instances_train2017.json", "w") as f:
    json.dump(coco_train, f, indent=4)

with open(out / "annotations" / "instances_val2017.json", "w") as f:
    json.dump(coco_val, f, indent=4)

print("\nCOCO dataset created in:", OUTPUT_ROOT)
print("Done.")
