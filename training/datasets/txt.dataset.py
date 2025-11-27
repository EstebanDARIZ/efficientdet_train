from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


class TXTDetectionDataset(Dataset):
    """
    Format attendu :
    images/
        image1.jpg
        image2.jpg
        ...
    labels/
        image1.txt
        image2.txt
        ...

    Chaque fichier .txt contient :
    class_id xmin ymin xmax ymax
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
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h

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
                    
                    xmin *= scale_x
                    xmax *= scale_x
                    ymin *= scale_y
                    ymax *= scale_y

                    bboxes.append([ymin, xmin, ymax, xmax])
                    labels.append(cls)

        if len(bboxes) == 0:
            bboxes = [[0, 0, 1, 1]]
            labels = [0]

        return (
            F.to_tensor(img),
            {
                "bbox": torch.tensor(bboxes, dtype=torch.float32),
                "cls": torch.tensor(labels, dtype=torch.float32),
                "img_size": torch.tensor([target_h, target_w]),
                "img_scale": torch.tensor([1.0]),
            },
        )


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets
