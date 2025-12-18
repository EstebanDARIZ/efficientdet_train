#!/usr/bin/env python3
import argparse
import torch
from effdet import create_model
from PIL import Image
import numpy as np
import cv2
import os
import time


'''
Inference script for EfficientDet object detection model using PyTorch.
Exemple usage:
python compute_IOU.py \
    --model tf_efficientdet_d0 \
    --checkpoint /home/esteban-dreau-darizcuren/doctorat/code/detector/efficientdet_train/output/train/slurm/20251205-170046-tf_efficientdet_d0_full/model_best.pth.tar \
    --image-dir /home/esteban-dreau-darizcuren/doctorat/dataset/dataset_test_2.0/images \
    --num-classes 5 \
    --output-dir /home/esteban-dreau-darizcuren/doctorat/dataset/dataset_test_2.0/test \
    --save-images \
    --score-thresh 0.5

'''

CLASS_NAMES = ["Bait_1_Squid", "Bait_2_Sardine", "Ray", "Sunfish", "PilotFish", "Misterious_fish"]



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
    det = detections.cpu().numpy()
    detected_names = []

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if det.size == 0:
        return img_cv, detected_names   # ✅ toujours 2 valeurs

    boxes  = det[:, 0:4]
    scores = det[:, 4]
    labels = det[:, 5].astype(int)

    w, h = img.size
    scale_x = w / image_size
    scale_y = h / image_size

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    for box, score, cls in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[cls]
        detected_names.append(cls_name)

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_cv,
            f"{cls}:{score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

    return img_cv, detected_names



def load_yolo_labels(label_path, img_w, img_h):
    """
    Retourne une liste de GT boxes au format [x1, y1, x2, y2, class_id]
    """
    gt_boxes = []

    if not os.path.exists(label_path):
        return gt_boxes

    with open(label_path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())

            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h

            gt_boxes.append([x1, y1, x2, y2, int(cls)])

    return gt_boxes

def compute_iou(boxA, boxB):
    """
    boxA, boxB : [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea

    if union == 0:
        return 0.0

    return interArea / union

def get_pred_boxes(detections, img_w, img_h, image_size, score_thresh):
    """
    Retourne [x1, y1, x2, y2, cls, score]
    """
    det = detections.cpu().numpy()
    pred_boxes = []

    scale_x = img_w / image_size
    scale_y = img_h / image_size

    for d in det:
        x1, y1, x2, y2, score, cls = d
        if score < score_thresh:
            continue

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        pred_boxes.append([x1, y1, x2, y2, int(cls), score])

    return pred_boxes

def match_predictions_to_gt(pred_boxes, gt_boxes, iou_thresh=0.5):
    matches = []

    for pb in pred_boxes:
        best_iou = 0
        best_gt = None

        for gt in gt_boxes:
            if pb[4] != gt[4]:  # classe différente
                continue

            iou = compute_iou(pb[:4], gt[:4])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        matches.append({
            "pred": pb,
            "best_iou": best_iou,
            "is_tp": best_iou >= iou_thresh
        })

    return matches




def main():
    parser = argparse.ArgumentParser(description="EfficientDet inference + IoU evaluation (YOLO GT)")

    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--save-images", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load model
    # -----------------------------
    start = time.time()
    bench = load_model(
        args.model,
        args.checkpoint,
        args.num_classes,
        args.image_size,
        device
    )
    print(f"[INFO] Model loaded in {time.time() - start:.2f}s")

    # -----------------------------
    # Stats
    # -----------------------------
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ious = []

    # YOLO labels directory (adapt if needed)
    label_dir = args.image_dir.replace("images", "labels")

    # -----------------------------
    # Loop over images
    # -----------------------------
    for img_name in sorted(os.listdir(args.image_dir)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(args.image_dir, img_name)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        # Load image
        img, img_tensor = preprocess_image(
            img_path, args.image_size, device
        )
        img_w, img_h = img.size

        # Inference
        detections = run_inference(bench, img_tensor)

        # Predictions
        pred_boxes = get_pred_boxes(
            detections,
            img_w,
            img_h,
            args.image_size,
            args.score_thresh
        )

        # Ground truth
        gt_boxes = load_yolo_labels(label_path, img_w, img_h)

        matched_gt = set()

        # -----------------------------
        # Matching predictions ↔ GT
        # -----------------------------
        for pb in pred_boxes:
            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):
                if pb[4] != gt[4]:  # class mismatch
                    continue

                iou = compute_iou(pb[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= args.iou_thresh:
                total_tp += 1
                all_ious.append(best_iou)
                matched_gt.add(best_gt_idx)
            else:
                total_fp += 1

        # False negatives
        total_fn += len(gt_boxes) - len(matched_gt)

        # -----------------------------
        # Optional image saving
        # -----------------------------
        if args.save_images:
            result, _ = draw_detections(
                img, detections, args.image_size, args.score_thresh
            )
            out_path = os.path.join(args.output_dir, img_name)
            cv2.imwrite(out_path, result)

    # -----------------------------
    # Final metrics
    # -----------------------------
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0

    print("\n================= RESULTS =================")
    print(f"TP : {total_tp}")
    print(f"FP : {total_fp}")
    print(f"FN : {total_fn}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"Mean IoU  : {mean_iou:.4f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
