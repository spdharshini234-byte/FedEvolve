"""ResNet-18 with transfer learning."""
import warnings
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    weights = None
    build_resnet18.last_pretrained_loaded = False
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            build_resnet18.last_pretrained_loaded = True
        except Exception as e:  # no internet / download blocked
            warnings.warn(f"Could not load ImageNet weights ({type(e).__name__}); using random initialisation instead.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new classification head
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return model


def build_model(cfg, num_classes=None):
    n = num_classes or len(cfg["dataset"]["classes"])
    m = cfg["model"]
    if m.get("name", "resnet18") != "resnet18":
        raise ValueError("Only resnet18 is supported in this project.")
    return build_resnet18(n, m.get("pretrained", True), m.get("freeze_backbone", False))


build_resnet18.last_pretrained_loaded = False
