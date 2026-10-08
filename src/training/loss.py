import torch
import torch.nn as nn


def class_mask_tensor(active_indices, num_classes):
    m = torch.zeros(num_classes, dtype=torch.bool)
    m[list(active_indices)] = True
    return m


def apply_class_mask(logits, mask):
    """Hide classes that are not part of the current task / seen set."""
    if mask is None:
        return logits
    return logits.masked_fill(~mask.to(logits.device), -1e9)


def get_loss():
    return nn.CrossEntropyLoss()
