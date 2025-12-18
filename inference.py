#!/usr/bin/env python3
import argparse
import torch
from effdet import create_model
from PIL import Image
import numpy as np
import cv2
import os

from config import CLASS_NAMES

'''
Inference script for EfficientDet object detection model using PyTorch.
Exemple usage:
python inference.py \
    --model tf_efficientdet_d0 \
    --checkpoint /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/output/train/slurm/20251205-170046-tf_efficientdet_d0_full/model_best.pth.tar \
<<<<<<< HEAD
    --image-dir /home/esteban-dreau-darizcuren/doctorat/dataset/img_processed/img_enhanced/UCD \
    --num-classes 5 \
    --output-dir /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/output/inference/UCD \
    --device cuda \
    --score-thresh 0.4

python inference.py \
    --model tf_efficientdet_d0 \
    --checkpoint /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/output/train/slurm/20251209-165945-tf_efficientdet_d0/model_best.pth.tar \
    --image-dir /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/color_jitter \
    --num-classes 5 \
    --output-dir /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/color_jitter/output/step_5 \
    --device cuda \
    --score-thresh 0.5


=======
    --image /home/esteban-dreau-darizcuren/doctorat/dataset/dataset_test_2.0/images \
    --num-classes 5 \
    --output /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/output/inference/dataset_test
>>>>>>> 840c168 (create compute_IOU based on inference script)
'''
import time 
from codecarbon import OfflineEmissionsTracker
import torchprofile #To calculate GFLOPs


def load_model(model_name, checkpoint, num_classes, image_size, device):
    print(f"[INFO] Loading model {model_name} …")

    bench = create_model(
        model_name,
        bench_task='predict',
        num_classes=num_classes,
        checkpoint_path=checkpoint,
        image_size=(image_size, image_size),
    )

    bench.eval()
    bench.to(device)

    print("[INFO] Model loaded.")
    return bench


def preprocess_image(image_path, image_size, device):
    img = Image.open(image_path).convert("RGB")

    img_resized = img.resize((image_size, image_size))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img, img_tensor


def run_inference(bench, img_tensor):
    with torch.no_grad():
        detections = bench(img_tensor)
    return detections[0]


def draw_detections(img, detections, image_size, score_thresh=0.1):
    det = detections.cpu().numpy()   # detections is (N, 6)
    detected_names = []

    if det.size == 0:
        print("[INFO] No detections")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    boxes  = det[:, 0:4]
    scores = det[:, 4]
    labels = det[:, 5].astype(int)

    w, h = img.size
    scale_x = w / image_size
    scale_y = h / image_size

    # scale coords back to original image size
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    for box, score, cls in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[cls]
        detected_names.append(cls_name)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{cls}:{score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)

    return img_cv, detected_names 



def main():
    parser = argparse.ArgumentParser(description="EfficientDet inference script")

    parser.add_argument("--model", required=True, help="Model name (ex: tf_efficientdet_lite0)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth.tar")
    parser.add_argument("--image-dir", required=True, help="Path to input images directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--image-size", type=int, default=512, help="Image size used during training")
    parser.add_argument("--output-dir", default="result.jpg", help="Output directory to save results")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--score-thresh", type=float, default=0.1, help="Score threshold for displaying detections")

    args = parser.parse_args()

    score_thresh = args.score_thresh
    img_dir = args.image_dir
    out_dir = args.output_dir
    class_count = {name: 0 for name in CLASS_NAMES}


    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    start_load = time.time()
    # Load model
    bench = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,
        args.image_size,
        device
    )
    list_cls = []
    print("Loading model time : ", time.time() - start_load)
    total_params = sum(p.numel() for p in bench.parameters())
    print(f"[INFO] Number of parameters: {total_params:,}")

    # Load & preprocess image
    for img_name in os.listdir(img_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, img_name)
        img, img_tensor = preprocess_image(img_path, args.image_size, device)

        # Inference
        start_inf = time.time()
        detections = run_inference(bench, img_tensor)
        print("Inference time : ", time.time() - start_inf) 
        # To get the GFLOPS info 
        macs = torchprofile.profile_macs(bench, img_tensor)
        print(macs/1000000000, "GFLOPs")

        # Draw results
        result, cls_name = draw_detections(img, detections, args.image_size, score_thresh)
        list_cls.append((img_name, cls_name))
        for cname in cls_name:
            class_count[cname] += 1

        # Save image
        output_path = os.path.join(out_dir, img_name)
        cv2.imwrite(output_path, result)
        # print(f"[INFO] Saved detection result to: {args.output}")

    txt_path = os.path.join(out_dir, "classes_detected.txt")
    
    with open(txt_path, "w", encoding="utf-8") as f:

        # Écrire les détections image par image
        for img_name, cls_list in list_cls:

            if len(cls_list) == 0:
                f.write(f"{img_name} : \n")
            else:
                classes_str = ", ".join(cls_list)
                f.write(f"{img_name} : {classes_str}\n")

            # Mise à jour du total
            for cname in cls_list:
                class_count[cname] += 1

        # Résumé final
        f.write("\n--- TOTAL DETECTIONS ---\n")
        for cname, count in class_count.items():
            f.write(f"{cname} : {count}\n")
        f.write("------------------------\n")



if __name__ == "__main__":
    # tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    # tracker.start()
    main()
    # tracker.stop()
