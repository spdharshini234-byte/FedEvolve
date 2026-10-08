"""Local training loop (used by centralized, local-only and federated clients)."""
import torch
from .loss import apply_class_mask, get_loss


def train_epochs(model, loader, optimizer, device, epochs=1, class_mask=None):
    """Train `model` in place. Returns average loss and accuracy of the last epoch."""
    model.to(device).train()
    criterion = get_loss()
    stats = {"loss": None, "accuracy": None, "steps": 0}
    if len(loader.dataset) < 2:
        return stats  # BatchNorm needs >= 2 samples
    for _ in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = apply_class_mask(model(x), class_mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
            stats["steps"] += 1
        stats["loss"], stats["accuracy"] = loss_sum / max(total, 1), correct / max(total, 1)
    return stats
