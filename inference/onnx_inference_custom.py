import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.preprocessing import preprocess
from utils.nms import nms

from config import ONNX_MODEL_PATH, IMAGE_SIZE, CLASS_NAMES

def detect(image_path, score_threshold=0.3, iou_threshold=0.5):

    img_original, blob, orig_w, orig_h = preprocess(image_path)

    outputs = session.run(None, {input_name: blob})
    scores, labels, boxes = outputs
    scores = scores[0]
    labels = labels[0]
    boxes = boxes[0]

    # Filtrage score
    mask = scores > score_threshold
    scores = scores[mask]
    labels = labels[mask]
    boxes = boxes[mask]

    if len(boxes) == 0:
        print("Aucune détection")
        return

    # Remise à l'échelle
    scale_x = orig_w / IMAGE_SIZE
    scale_y = orig_h / IMAGE_SIZE
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # NMS
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]

    # Affichage Matplotlib
    fig, ax = plt.subplots(1)
    ax.imshow(img_original)

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            w, h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

        class_name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        ax.text(
            x1, y1 - 5,
            f"{class_name} {score:.2f}",
            color="lime",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.5)
        )

    plt.show()


if __name__ == "__main__":
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    class_names = CLASS_NAMES

    # Exemple d'image
    image_path = "/home/esteban-dreau-darizcuren/doctorat/dataset/dataset/images/L_frame_16295.jpg"
    detect(image_path, score_threshold=0.3, iou_threshold=0.5)