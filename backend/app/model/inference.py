"""Loads a saved checkpoint and classifies one uploaded image."""
from pathlib import Path
import torch
from PIL import Image
from src.data.augmentation import get_transforms
from src.models.checkpoint import load_checkpoint
from src.utils.paths import project_root

PREFERRED = ["federated_continual", "fedavg", "centralized", "federated_no_replay"]
_CACHE = {}


def models_dir() -> Path:
    return project_root() / "models" / "global"


def available_models():
    return [n for n in PREFERRED if (models_dir() / f"{n}.pt").exists()]


def _load(name):
    path = models_dir() / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No trained model '{name}'. Train one first (python run.py train-fedavg --demo).")
    key = (str(path), path.stat().st_mtime)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load_checkpoint(path)
    return _CACHE[key]


def predict_image(img: Image.Image, model_name=None):
    names = available_models()
    if not names:
        raise FileNotFoundError("No trained model found. Run a training command first.")
    name = model_name or names[0]
    model, classes, meta = _load(name)
    size = meta.get("image_size", 64)
    x = get_transforms(size, train=False)(img.convert("RGB")).unsqueeze(0)
    seen = [classes.index(c) for c in meta.get("seen_classes", classes)]
    mask = torch.zeros(len(classes), dtype=torch.bool)
    mask[seen] = True
    with torch.no_grad():
        logits = model(x).masked_fill(~mask, -1e9)
        probs = torch.softmax(logits, 1)[0]
    i = int(probs.argmax())
    return {"predicted_class": classes[i], "confidence": float(probs[i]), "model_name": meta.get("display_name", name),
            "probabilities": {classes[k]: float(probs[k]) for k in seen}, "dataset_source": meta.get("dataset_source")}
