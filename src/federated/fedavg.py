"""FedAvg: weighted average of client model weights (weights = number of local samples)."""
from collections import OrderedDict
import torch


def fedavg(updates):
    """updates: list of (state_dict, num_samples). Returns the averaged state_dict."""
    if not updates:
        raise ValueError("fedavg needs at least one client update")
    total = float(sum(n for _, n in updates))
    weights = [n / total for _, n in updates] if total > 0 else [1 / len(updates)] * len(updates)
    keys = list(updates[0][0].keys())
    for sd, _ in updates:
        if list(sd.keys()) != keys:
            raise ValueError("Client updates have different parameter names")
    out = OrderedDict()
    for k in keys:
        first = updates[0][0][k]
        if first.is_floating_point():
            out[k] = sum(w * sd[k].float() for w, (sd, _) in zip(weights, updates)).to(first.dtype)
        else:  # integer buffers such as BatchNorm num_batches_tracked
            out[k] = torch.stack([sd[k] for sd, _ in updates]).max(0).values
    return out
