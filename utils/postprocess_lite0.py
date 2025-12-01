import numpy as np

# -------------------
# EfficientDet-Lite0 
# -------------------

# Strides of each FPN level
STRIDES = [8, 16, 32, 64, 128]

# Scales & ratios used in EfficientDet-Lite0
SCALES = [2 ** 0, 2 ** (1/3), 2 ** (2/3)]
RATIOS = [0.5, 1.0, 2.0]

# def generate_anchors(img_size):
#     """Generate all anchors for EfficientDet Lite0."""
#     anchors = []
#     for stride in STRIDES:
#         fm_size = img_size // stride

#         for i in range(fm_size):
#             for j in range(fm_size):
#                 cx = (j + 0.5) * stride
#                 cy = (i + 0.5) * stride

#                 for scale in SCALES:
#                     for ratio in RATIOS:
#                         w = stride * scale * np.sqrt(ratio)
#                         h = stride * scale / np.sqrt(ratio)

#                         anchors.append([cx, cy, w, h])

    # return np.array(anchors, dtype=np.float32)


import numpy as np

def generate_anchors_lite0(image_size=320):
    """
    EfficientDet Lite0 anchor generator matching ONNX output.
    Produces exactly 2134 locations and 2134*9 anchors.
    """
    feature_sizes = [
        (image_size // 8,  image_size // 8),   # 40x40
        (image_size // 16, image_size // 16),  # 20x20
        (image_size // 32, image_size // 32),  # 10x10
        (image_size // 64, image_size // 64),  # 5x5
        (3, 3),                                 # SPECIAL CASE: ONNX Lite0 uses 3×3 for P7
    ]

    strides = [8, 16, 32, 64, 128]

    scales = [1.0, 1.414, 2.0]
    ratios = [0.5, 1.0, 2.0]

    anchors = []

    for (fh, fw), stride in zip(feature_sizes, strides):
        for iy in range(fh):
            cy = (iy + 0.5) * stride
            for ix in range(fw):
                cx = (ix + 0.5) * stride
                for s in scales:
                    for r in ratios:
                        w = stride * s / np.sqrt(r)
                        h = stride * s * np.sqrt(r)
                        anchors.append([
                            cx - w / 2, cy - h / 2,
                            cx + w / 2, cy + h / 2
                        ])

    return np.array(anchors, dtype=np.float32)



def decode_boxes(pred_boxes, anchors):
    """
    EfficientDet decode step :
    anchors : (N,4) XYXY
    pred_boxes : (N,4) (tx, ty, tw, th)
    Output : decoded XYXY boxes
    """
    ax1 = anchors[:, 0]
    ay1 = anchors[:, 1]
    ax2 = anchors[:, 2]
    ay2 = anchors[:, 3]

    aw = ax2 - ax1
    ah = ay2 - ay1
    acx = ax1 + 0.5 * aw
    acy = ay1 + 0.5 * ah

    tx = pred_boxes[:, 0]
    ty = pred_boxes[:, 1]
    tw = pred_boxes[:, 2]
    th = pred_boxes[:, 3]

    # EfficientDet Box Decoding
    cx = acx + tx * aw
    cy = acy + ty * ah
    w = aw * np.exp(tw)
    h = ah * np.exp(th)

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return np.stack([x1, y1, x2, y2], axis=1)


def nms(boxes, scores, iou_threshold):
    """
    NMS simple en NumPy (suffisant pour post-process ONNX).
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_threshold)[0] + 1
        order = order[remaining]

    return np.array(keep, dtype=np.int32)




# -----------
# NMS
# -----------
def nms(boxes, scores, iou_threshold=0.5, max_detections=100):
    idxs = scores.argsort()[::-1]
    keep = []

    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(keep) >= max_detections:
            break

        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_others = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])

        union = area_i + area_others - inter
        iou = inter / (union + 1e-6)

        idxs = idxs[1:][iou < iou_threshold]

    return keep
