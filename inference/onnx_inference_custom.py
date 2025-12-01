import onnxruntime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.postprocess_lite0 import generate_anchors_lite0, decode_boxes, nms
from config import CLASS_NAMES, IMAGE_SIZE, MODEL_PATH
from utils.preprocessing import preprocess


def detect(image_path, score_threshold=0.3, iou_threshold=0.5):
    session = onnxruntime.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    img_original, blob, orig_w, orig_h = preprocess(image_path)

    print("Nombre de sorties ONNX :", len(session.get_outputs()))
    for i, o in enumerate(session.get_outputs()):
        print(f"Output {i} → name={o.name} shape={o.shape}")


    outputs = session.run(None, {input_name: blob})
    class_heads = outputs[:5]     # sorties 0..4
    box_heads   = outputs[5:]     # sorties 5..9

    # Fusion class logits
    class_out = np.concatenate(
        [c.reshape(c.shape[0], c.shape[1], -1) for c in class_heads],
        axis=2
    )
    class_out = class_out.transpose(0, 2, 1)[0]  # shape = (N_anchors, 54)

    # Fusion box regression
    box_out = np.concatenate(
        [b.reshape(b.shape[0], b.shape[1], -1) for b in box_heads],
        axis=2
    )
    box_out = box_out.transpose(0, 2, 1)[0]  # shape = (N_anchors, 36)

    num_classes = 6
    num_anchors = 9   # Lite0 = 9 anchors par cellule

    # class_out : (2134, 54) → (2134, 9, 6)
    class_out = class_out.reshape(-1, num_anchors, num_classes)

    # box_out : (2134, 36) → (2134, 9, 4)
    box_out = box_out.reshape(-1, num_anchors, 4)

    # Flatten pour avoir 1 anchor = 1 ligne
    class_out = class_out.reshape(-1, num_classes)   # (2134*9 = 19206, 6)
    box_out = box_out.reshape(-1, 4)                 # (19206, 4)




    for i, o in enumerate(outputs[:5]):
        print(f"Class head {i} shape: {o.shape}")
    for i, o in enumerate(outputs[5:]):
        print(f"Box head {i} shape: {o.shape}")


    anchors = generate_anchors_lite0(IMAGE_SIZE)
    anchors = anchors.reshape(-1, 4)   # (19206, 4)


    scores = class_out.max(axis=1)
    labels = class_out.argmax(axis=1)

    class_out = 1 / (1 + np.exp(-class_out))   # sigmoid


    mask = scores > score_threshold
    scores = scores[mask]
    labels = labels[mask]
    box_out = box_out[mask]
    anchors = anchors[mask]

    if len(scores) == 0:
        print("Aucune détection.")
        return

    boxes = decode_boxes(box_out, anchors)

    scale_x = orig_w / IMAGE_SIZE
    scale_y = orig_h / IMAGE_SIZE

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    fig, ax = plt.subplots(1)
    ax.imshow(img_original)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"{CLASS_NAMES[label]} {score:.2f}",
            color="lime",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5)
        )

    plt.show()


if __name__ == "__main__":
    image_path = "/home/esteban-dreau-darizcuren/doctorat/dataset/dataset/images/L_frame_16295.jpg"
    detect(image_path, score_threshold=0.3, iou_threshold=0.5)
