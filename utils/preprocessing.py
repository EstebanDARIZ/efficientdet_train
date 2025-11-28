from PIL import Image
import numpy as np

from config import IMAGE_SIZE

def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.asarray(img_resized).astype(np.float32) / 255.0

    img_np = img_np.transpose(2, 0, 1)  # CHW
    img_np = np.expand_dims(img_np, axis=0)

    return img, img_np, orig_w, orig_h