from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class TXTDetectionDataset(Dataset):
    """
    Dataset for object detection using TXT annotation files.
    Each image has a corresponding .txt file with bounding box annotations.
    Exemple of .txt file format:
    0 34 45 200 300
    1 120 150 400 500
    where each line represents: class_id xmin ymin xmax ymax
    0-based class IDs are used.

    Args:
        images_dir (str or Path): Directory containing images.
        labels_dir (str or Path): Directory containing TXT annotation files.
        image_size (int): Size to which images are resized (image_size x image_size).
    
    Returns:
        image (Tensor): Resized image tensor.
        target (dict): Dictionary containing:
            - "bbox": Tensor of shape (num_boxes, 4) with bounding boxes in (ymin, xmin, ymax, xmax) format.
            - "cls": Tensor of shape (num_boxes,) with class IDs.
            - "img_size": Tensor with original image size.
            - "img_scale": Tensor with scaling factor applied to the image.
    """

    def __init__(self, images_dir, labels_dir, image_size=512):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size

        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        orig_w, orig_h = img.size
        target_w = target_h = self.image_size

        img = img.resize((target_w, target_h))
        scale_x = target_w / orig_w  #Used to scale bounding boxes
        scale_y = target_h / orig_h  #Used to scale bounding boxes

        txt_path = self.labels_dir / (img_path.stem + ".txt")

        bboxes = []
        labels = []

        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, xmin, ymin, xmax, ymax = map(float, parts)
                    
                    xmin *= scale_x # Scale bounding box coordinates
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y

                    bboxes.append([xmin, ymin,  xmax, ymax])
                    labels.append(cls)
        # Handle case with no bounding boxes
        if len(bboxes) == 0:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.float32)
        else:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)

        return (
            F.to_tensor(img),
            {
                "bbox": torch.tensor(bboxes, dtype=torch.float32),
                "cls": torch.tensor(labels, dtype=torch.float32),
                "img_size": torch.tensor([target_h, target_w]),
                "img_scale": torch.tensor([1.0]), # No scaling applied after resizing, efficentde doesn't use it
            },
        )


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
