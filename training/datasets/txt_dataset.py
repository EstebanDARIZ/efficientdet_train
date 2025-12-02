from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class TXTDetectionDataset(Dataset):
    """
    Dataset compatible with EfficientDet (effdet).
    TXT files format: class xmin ymin xmax ymax
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
        new_size = self.image_size

        # Resize
        img = img.resize((new_size, new_size))
        scale_x = new_size / orig_w
        scale_y = new_size / orig_h

        # Load labels
        txt_path = self.labels_dir / (img_path.stem + ".txt")
        boxes = []
        labels = []

        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    cls, xmin, ymin, xmax, ymax = map(float, line.split())

                    # Scale
                    xmin *= scale_x
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y

                    # EfficientDet wants (ymin, xmin, ymax, xmax)
                    boxes.append([ymin, xmin, ymax, xmax])
                    labels.append(int(cls))  # class ID as int

        # Convert to tensor
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return (
            F.to_tensor(img),
            {
                "bbox": boxes,
                "cls": labels,
                "img_size": torch.tensor([new_size, new_size], dtype=torch.float32),
                "img_scale": torch.tensor([1.0], dtype=torch.float32),
            },
        )


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
