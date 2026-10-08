import torch
from src.models.resnet_model import build_resnet18
from src.models.checkpoint import save_checkpoint, load_checkpoint


def test_model_output_and_freeze():
    m = build_resnet18(4, pretrained=False, freeze_backbone=True)
    assert m(torch.randn(2, 3, 32, 32)).shape == (2, 4)
    assert all(p.requires_grad == n.startswith("fc.") for n, p in m.named_parameters())


def test_checkpoint_roundtrip(tmp_path):
    m = build_resnet18(3, pretrained=False)
    save_checkpoint(tmp_path / "m.pt", m, ["a", "b", "c"], {"image_size": 32})
    m2, classes, meta = load_checkpoint(tmp_path / "m.pt")
    assert classes == ["a", "b", "c"] and meta["image_size"] == 32
    x = torch.randn(1, 3, 32, 32)
    assert torch.allclose(m.eval()(x), m2(x), atol=1e-5)
