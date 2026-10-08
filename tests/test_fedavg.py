from collections import OrderedDict
import pytest
import torch
from src.federated.fedavg import fedavg
from src.federated.server import Server
from src.models.resnet_model import build_resnet18


def sd(v, steps=1):
    return OrderedDict(w=torch.full((2,), float(v)), num_batches_tracked=torch.tensor(steps))


def test_weighted_average():
    out = fedavg([(sd(0), 10), (sd(4), 30)])
    assert torch.allclose(out["w"], torch.tensor([3.0, 3.0]))   # (0*10 + 4*30)/40


def test_empty_raises():
    with pytest.raises(ValueError):
        fedavg([])


def test_server_aggregates_and_rejects_raw_data():
    s = Server(build_resnet18(2, pretrained=False))
    base = s.broadcast()
    with pytest.raises(TypeError):
        s.receive_update(1, {"image": [1, 2, 3]}, 5)   # raw data is refused
    s.receive_update(1, base, 10)
    s.receive_update(2, base, 20)
    s.aggregate(2)
    assert s.round == 1 and s.comm.total_bytes > 0 and s.comm.raw_image_bytes == 0
