"""Loads and merges the three YAML config files into one dictionary."""
from pathlib import Path
import yaml
from .paths import project_root, resolve

CONFIG_FILES = ["config.yaml", "federated_config.yaml", "continual_config.yaml"]


def load_config(root=None, overrides=None) -> dict:
    root = Path(root) if root else project_root()
    cfg = {}
    for name in CONFIG_FILES:
        f = root / "configs" / name
        if not f.exists():
            raise FileNotFoundError(f"Config file not found: {f}. Is this the project root?")
        with open(f, "r", encoding="utf-8") as fh:
            cfg.update(yaml.safe_load(fh) or {})
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def get_paths(cfg, root=None) -> dict:
    root = Path(root) if root else project_root()
    return {k: resolve(v, root) for k, v in cfg["paths"].items()} | {"root": root}


def class_index(cfg) -> dict:
    return {c: i for i, c in enumerate(cfg["dataset"]["classes"])}


def task_class_indices(cfg) -> list:
    """Continual tasks as lists of class indices, e.g. [[0, 1], [2], [3]]."""
    idx = class_index(cfg)
    tasks = []
    for t in cfg["continual"]["tasks"]:
        for c in t["classes"]:
            if c not in idx:
                raise ValueError(f"Task class '{c}' is not in dataset.classes {list(idx)}")
        tasks.append([idx[c] for c in t["classes"]])
    return tasks


def num_epochs(cfg) -> int:
    e = cfg["training"].get("epochs", "auto")
    if isinstance(e, int):
        return e
    return int(cfg["federated"]["rounds"]) * int(cfg["federated"]["local_epochs"])
