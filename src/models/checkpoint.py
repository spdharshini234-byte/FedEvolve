import time
from pathlib import Path
import torch
from .resnet_model import build_resnet18


def save_checkpoint(path, model, classes, meta=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "classes": list(classes),
               "meta": dict(meta or {}, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))}
    torch.save(payload, path)
    return path


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Train a model first.")
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_resnet18(len(payload["classes"]), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload["classes"], payload.get("meta", {})
