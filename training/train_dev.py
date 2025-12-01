import argparse
import torch
from torch.utils.data import DataLoader
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet
from effdet.bench import DetBenchTrain
from tqdm import tqdm
# from effdet.onnx_export import EfficientDetONNXExport

from datasets.txt_dataset import TXTDetectionDataset, collate_fn

"""
Training script for EfficientDet object detection model using a custom TXT dataset.

Args:
    --images-dir: Directory containing training images.
    --labels-dir: Directory containing TXT annotation files.
    --num-classes: Number of object classes.
    --model-name: EfficientDet model variant (default: tf_efficientdet_lite0).
    --image-size: Size to which images are resized (default: 320).
    --batch-size: Training batch size (default: 4).
    --epochs: Number of training epochs (default: 10).
    --lr: Learning rate (default: 1e-4).
    --checkpoint: Path to save the best model checkpoint (default: training/checkpoints/model.pth).
    --onnx-out: Path to export the trained model in ONNX format (optional).

Exemple usage:
    python training/train.py \
    --images-dir /home/esteban-dreau-darizcuren/doctorat/dataset/dataset_efficientdet/images \
    --labels-dir /home/esteban-dreau-darizcuren/doctorat/dataset/dataset_efficientdet/labels \
    --num-classes 6 \
    --model-name tf_efficientdet_lite0 \
    --image-size 512 \
    --epochs 1 \
    --batch-size 8 \
    --lr 0.0001 \
    --checkpoint training/checkpoints/best_model.pth \
    --onnx-out onnx_models/efficientdet.onnx   
"""

def create_model(model_name, num_classes, image_size):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config, num_outputs=num_classes)

    return DetBenchTrain(net, config), config


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for imgs, targets in tqdm(loader, desc="Training", ncols=80):
        # Move images
        imgs = imgs.to(device)

        # Move each dict of targets to device
        for t in targets:
            t["bbox"] = t["bbox"].to(device)
            t["cls"] = t["cls"].to(device)
            t["img_size"] = t["img_size"].to(device)
            t["img_scale"] = t["img_scale"].to(device)

        optimizer.zero_grad()
        
        # EfficientDet expects:
        # imgs: Tensor[B, 3, H, W]
        # targets: list of dicts
        out = model(imgs, targets)
        loss = out["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)



def export_onnx(model, config, path, image_size, device):
    print("Export ONNX (raw outputs)…")

    # Modèle d'export ONNX (raw classification + raw box regression)
    # export_model = DetBenchEval(model.model, config)
    # export_model.eval()
    # export_model.to("cpu")

    pred = model.model 
    pred.eval().to('cpu')

    # Dummy input
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    # Export ONNX : 2 sorties -> class_out, box_out
    torch.onnx.export(
        pred,
        dummy,
        path,
        input_names=["images"],
        output_names=["class_out", "box_out"],
        opset_version=12,
        dynamic_axes={
            "images": {0: "batch"},
            "class_out": {0: "batch"},
            "box_out": {0: "batch"},
        }
    )

    print("✔ ONNX exporté :", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--model-name", default="tf_efficientdet_lite0")
    parser.add_argument("--image-size", type=int, default=320)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--checkpoint", default="training/checkpoints/model.pth")
    parser.add_argument("--onnx-out", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("This will be trained on", device)

    dataset = TXTDetectionDataset(args.images_dir, args.labels_dir, args.image_size)
    print(f"Dataset size: {len(dataset)} images")

    n = len(dataset)
    n_train = int(0.8*n)
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n-n_train])
    print(f"Train set: {len(train_set)} images")
    print(f"Validation set: {len(val_set)} images")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    model, config = create_model(args.model_name, args.num_classes, args.image_size)
    model.to(device)
    print("Model created.")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = float("inf")
    print("Starting training...")
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, train_loader, optim, device)
        print(f"[Epoch {epoch}] Loss={loss:.4f}")

        if loss < best:
            best = loss
            torch.save(model.state_dict(), args.checkpoint)
            print("→ Nouveau meilleur modèle sauvegardé.")

    if args.onnx_out:
        export_onnx(model, config, args.onnx_out, args.image_size, device)


if __name__ == "__main__":
    main()
