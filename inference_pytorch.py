import argparse
import torch
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet
from effdet.bench import DetBenchPredict  
import cv2
import numpy as np

"""
Inference script for EfficientDet object detection model using PyTorch.
Exemple usage:
    python inference_pytorch.py \
    --model-name tf_efficientdet_lite0 \
    --num-classes 6 \
    --image-size 320 \
    --checkpoint training/checkpoints/best_model.pth \
    --input 5.png \
    --output result.jpg \
    --threshold 0.3
"""

def load_model(model_name, num_classes, image_size, checkpoint_path, device="cpu"):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    # Créer EfficientDet brut
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)

    # Emballer dans DetBenchPredict
    model = DetBenchPredict(net)

    # Charger les poids entraînés
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)

    model.eval()
    model.to(device)
    return model, config



def preprocess_image(path, image_size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = image_size / max(h, w)
    resized = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Creation odf padded image (image_size, image_size) than placing resized image at top-left corner 
    padded = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized

    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), img, scale

from torchvision.ops import nms

@torch.no_grad()
def predict(model, tensor, threshold):
    out = model(tensor)[0]  # (num_det, 6)

    boxes = out[:, :4]
    scores = out[:, 4]
    labels = out[:, 5].long()

    mask = scores > threshold

    keep = nms(boxes[mask], scores[mask], iou_threshold=0.5)

    return boxes[keep], scores[keep], labels[keep]

def restore_boxes_to_original(boxes, original_h, original_w, scale):
    # Undo scaling
    boxes = boxes.clone()
    boxes[:, 0] /= scale   # x1
    boxes[:, 1] /= scale   # y1
    boxes[:, 2] /= scale   # x2
    boxes[:, 3] /= scale   # y2

    # Clamp to the image boundaries
    boxes[:, 0].clamp_(0, original_w)
    boxes[:, 2].clamp_(0, original_w)
    boxes[:, 1].clamp_(0, original_h)
    boxes[:, 3].clamp_(0, original_h)

    return boxes


def draw_boxes(img, boxes, scores, labels):
    img = img.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, f"{int(label)}:{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="EfficientDet Inference CLI")

    parser.add_argument("--model-name", default="tf_efficientdet_lite0", help="EfficientDet model variant (default: tf_efficientdet_lite0)")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--checkpoint", required=True, help="Path to .pth model checkpoint")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default="result.jpg", help="Output visualization image")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for NMS")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model_name = args.model_name

    model, config = load_model(
        model_name=model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        checkpoint_path=args.checkpoint,
        device=device
    )

    # Charger image
    tensor, original_img, _ = preprocess_image(args.input, args.image_size)
    tensor = tensor.to(device)

    # Inférence
    boxes, scores, labels = predict(model, tensor,  args.threshold)

    # Restore box coordinates
    h, w = original_img.shape[:2]
    scale = args.image_size / max(h, w)
    boxes = restore_boxes_to_original(boxes, h, w, scale)

    print("Detections:")
    for b, s, l in zip(boxes, scores, labels):
        print(f"Classe={int(l)}  Score={s:.2f}  BBox={b.tolist()}")

    # Sauvegarde output
    result = draw_boxes(original_img, boxes, scores, labels)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"\n Résultat sauvegardé dans : {args.output}")


if __name__ == "__main__":
    main()
