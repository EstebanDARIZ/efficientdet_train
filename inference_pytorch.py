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
    --input path/to/image.jpg \
    --output result.jpg \
    --threshold 0.3
"""

def load_model(model_name, num_classes, image_size, checkpoint_path, device="cpu"):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)

    model = DetBenchPredict(net)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval().to(device)

    return model, config

def load_model_raw(model_name, num_classes, image_size, checkpoint_path, device="cpu"):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net.eval().to(device)

    return net, config


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

    # Centering the resized image in the padded image 
    # y_off = (image_size - resized.shape[0]) // 2
    # x_off = (image_size - resized.shape[1]) // 2
    # padded[y_off:y_off+resized.shape[0], x_off:x_off+resized.shape[1]] = resized


    tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), img, scale


from torchvision.ops import nms 
from effdet.anchors import Anchors
import torch.nn.functional as F

def decode_boxes(boxes, anchors, scale=1.0):
    # boxes : [N, 4]
    # anchors : [N, 4]
    ax, ay, aw, ah = anchors.unbind(1)
    tx, ty, tw, th = boxes.unbind(1)

    x1 = (tx * aw * scale) + ax
    y1 = (ty * ah * scale) + ay
    x2 = torch.exp(tw) * aw * scale + x1
    y2 = torch.exp(th) * ah * scale + y1

    return torch.stack([x1, y1, x2, y2], dim=1)



@torch.no_grad()
def predict_postprocess(net, tensor, config, score_thresh=0.3, iou_thresh=0.5, max_det=100):
    # 1) Forward brut EfficientDet
    features = net.backbone(tensor)              # liste de features
    cls_outputs = net.class_net(features)        # list: levels → [B, A*C, H, W]
    box_outputs = net.box_net(features)          # list: levels → [B, A*4, H, W]

    # 2) Flatten outputs
    cls_outputs = torch.cat([c.reshape(-1, config.num_classes) for c in cls_outputs], dim=0)
    box_outputs = torch.cat([b.reshape(-1, 4) for b in box_outputs], dim=0)

    # 3) Anchors (alignés avec cls/box)
    anchors = Anchors.from_config(config).boxes.to(tensor.device)

    # 4) Décodage boîtes
    boxes = decode_boxes(box_outputs, anchors)

    # 5) Sigmoid sur scores
    scores = torch.sigmoid(cls_outputs)

    # 6) Score max + label
    max_scores, labels = scores.max(dim=1)

    keep = max_scores > score_thresh
    boxes = boxes[keep]
    labels = labels[keep]
    max_scores = max_scores[keep]

    # 7) NMS global (optionnel : NMS par classe)
    keep_nms = nms(boxes, max_scores, iou_thresh)

    if len(keep_nms) > max_det:
        keep_nms = keep_nms[:max_det]

    return boxes[keep_nms], max_scores[keep_nms], labels[keep_nms]



@torch.no_grad()
def predict(model, tensor, score_thresh=0.3, iou_threshold=0.5):
    outputs = model(tensor)[0]   # shape (N, 6)

    boxes  = outputs[:, :4]
    scores = outputs[:, 4]
    labels = outputs[:, 5].long()

    mask = scores > score_thresh
    boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

    keep = nms(boxes, scores, iou_threshold= iou_threshold)

    return boxes[mask], scores[mask], labels[mask]

def postprocess(boxes, path, image_size):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    h, w = img.shape[:2]
    scale = image_size / max(h, w)

    # Remove scaling
    boxes /= scale

    # Clamp to image boundaries
    boxes[:, 0] = boxes[:, 0].clamp(0, w)   # x1
    boxes[:, 1] = boxes[:, 1].clamp(0, h)   # y1
    boxes[:, 2] = boxes[:, 2].clamp(0, w)   # x2
    boxes[:, 3] = boxes[:, 3].clamp(0, h)   # y2
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
    iou_threshold = args.iou_threshold
    # Charger modèle
    # model, config = load_model(
    #     model_name=model_name,
    #     num_classes=args.num_classes,
    #     image_size=args.image_size,
    #     checkpoint_path=args.checkpoint,
    #     device=device
    # )

    model, config = load_model_raw(
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
    boxes, scores, labels = predict_postprocess(model, tensor, config,  args.threshold, iou_threshold)

    # boxes = postprocess(boxes, args.input, args.image_size)


    print("Detections:")
    for b, s, l in zip(boxes, scores, labels):
        print(f"Classe={int(l)}  Score={s:.2f}  BBox={b.tolist()}")

    # Sauvegarde output
    result = draw_boxes(original_img, boxes, scores, labels)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    print(f"\n✔ Résultat sauvegardé dans : {args.output}")


if __name__ == "__main__":
    main()
