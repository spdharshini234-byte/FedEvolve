import numpy as np
import torch
from .loss import apply_class_mask


@torch.no_grad()
def predict(model, loader, device, class_mask=None):
    model.to(device).eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        logits = apply_class_mask(model(x.to(device)), class_mask)
        pr = torch.softmax(logits, 1).cpu()
        ps.append(pr.argmax(1).numpy())
        probs.append(pr.numpy())
        ys.append(y.numpy())
    if not ys:
        return np.array([]), np.array([]), np.zeros((0, 1))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)
