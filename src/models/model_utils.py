import copy
from collections import OrderedDict
import torch


def get_weights(model):
    """CPU copy of the full state (parameters + BatchNorm buffers). This is what a factory uploads."""
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def set_weights(model, state):
    model.load_state_dict(state)


def clone_model(model):
    return copy.deepcopy(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def state_nbytes(state):
    """Size in bytes of a state dict (what would be sent over the network)."""
    return int(sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v)))
