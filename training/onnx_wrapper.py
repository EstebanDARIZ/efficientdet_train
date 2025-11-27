import torch
import torch.nn as nn

class EfficientDetONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # model = EfficientDet(...).model

    def forward(self, x):
        # Retourne directement les prédictions brutes
        class_out, box_out = self.model(x)
        
        # Output ONNX = raw cls + raw boxes
        return class_out, box_out
