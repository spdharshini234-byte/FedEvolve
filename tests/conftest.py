import os
import shutil
import sys
from pathlib import Path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TINY = {
    "dataset": {"image_size": 32, "min_images_per_class": 5, "demo": {"images_per_class": 24}},
    "model": {"pretrained": False},
    "training": {"batch_size": 8},
    "federated": {"rounds": 2, "local_epochs": 1},
    "continual": {"rounds_per_task": 1, "local_epochs": 1, "replay": {"buffer_size": 10}},
}


def _deep(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep(base[k], v)
        else:
            base[k] = v


@pytest.fixture(scope="session")
def tiny_project(tmp_path_factory):
    """A throw-away project root with tiny settings. Runs the whole real pipeline once (shared by tests)."""
    root = tmp_path_factory.mktemp("fcdd_root")
    (root / "configs").mkdir()
    for f in ("config.yaml", "federated_config.yaml", "continual_config.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / f).read_text())
        _deep(cfg, TINY)
        cfg = {k: v for k, v in cfg.items() if k in yaml.safe_load((ROOT / "configs" / f).read_text())}
        (root / "configs" / f).write_text(yaml.safe_dump(cfg))
    os.environ["FCDD_ROOT"] = str(root)
    from src import pipeline as pl
    ctx = pl.Context(root)
    pl.prepare_demo(ctx)
    pl.run_centralized(ctx)
    pl.run_local_only(ctx)
    pl.run_fedavg(ctx)
    pl.run_continual_both(ctx)
    pl.evaluate_all(ctx)
    yield root
    os.environ.pop("FCDD_ROOT", None)
    shutil.rmtree(root, ignore_errors=True)
