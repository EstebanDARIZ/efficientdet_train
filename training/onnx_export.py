import torch
import argparse
from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet

from onnx_wrapper import EfficientDetONNX


def create_model(model_name, num_classes, image_size):
    config = get_efficientdet_config(model_name)
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    return net, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx-out", required=True)
    parser.add_argument("--model-name", default="tf_efficientdet_lite2")
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_name = args.model_name

    print("Loading model...")
    model, config = create_model(model_name, args.num_classes, args.image_size)

    print("Loading checkpoint:", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)

    # Cas 1 : checkpoint effdet (probable)
    if "model" in ckpt:
        print("Checkpoint detected as EffDet format → using ckpt['model']")
        state = ckpt["model"]
    else:
        print("Checkpoint is raw state_dict")
        state = ckpt

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print("Building ONNX wrapper (no NMS)...")
    model_onnx = EfficientDetONNX(model).to(device)
    dummy = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    print("Exporting ONNX…")
    torch.onnx.export(
        model_onnx,
        dummy,
        args.onnx_out,
        input_names=["images"],
        output_names=["scores", "labels", "boxes"],
        opset_version=12,
        dynamic_axes={
            "images": {0: "batch"},
            "scores": {0: "batch"},
            "labels": {0: "batch"},
            "boxes": {0: "batch"},
        }
    )

    print("✔ ONNX export done:", args.onnx_out)


if __name__ == "__main__":
    main()
