"""Create NON-IID factory (client) datasets from the training split."""
import json
from pathlib import Path
import numpy as np
from .dataset import read_manifest, write_manifest, class_counts


def partition_items(train_items, num_classes, num_clients, mode="matrix", class_weights=None, alpha=0.3, seed=42):
    """Split training items among clients. class_weights: list (clients) of lists (classes)."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        items = [it for it in train_items if it[1] == c]
        rng.shuffle(items)
        if mode == "dirichlet":
            share = rng.dirichlet([alpha] * num_clients)
        else:
            share = np.array([class_weights[k][c] for k in range(num_clients)], dtype=float)
            if share.sum() <= 0:
                share = np.ones(num_clients)
        share = share / share.sum()
        cuts = np.round(np.cumsum(share) * len(items)).astype(int)
        start = 0
        for k in range(num_clients):
            parts[k].extend(items[start:cuts[k]])
            start = cuts[k]
    return parts


def weights_from_cfg(cfg):
    classes = cfg["dataset"]["classes"]
    cw = cfg["non_iid"]["class_weights"]
    n = cfg["federated"]["num_clients"]
    mat = []
    for k in range(n):
        row = cw.get(f"client_{k + 1}", {})
        mat.append([float(row.get(c, 0.0)) for c in classes])
    return mat


def create_clients(cfg, processed_dir, clients_dir):
    """Writes data/clients/client_i/train.csv + client_info.json. Only manifests are written (no data copied)."""
    classes = cfg["dataset"]["classes"]
    n = int(cfg["federated"]["num_clients"])
    train = read_manifest(Path(processed_dir) / "train.csv")
    mode = cfg["non_iid"].get("mode", "matrix")
    mat = weights_from_cfg(cfg) if mode == "matrix" else None
    parts = partition_items(train, len(classes), n, mode, mat, cfg["non_iid"].get("dirichlet_alpha", 0.3),
                            cfg["project"]["seed"])
    names = cfg["federated"].get("client_names", [f"Factory {k + 1}" for k in range(n)])
    infos = []
    for k, items in enumerate(parts):
        d = Path(clients_dir) / f"client_{k + 1}"
        d.mkdir(parents=True, exist_ok=True)
        write_manifest(items, d / "train.csv", classes)
        counts = class_counts(items, len(classes))
        info = {"client_id": k + 1, "name": names[k] if k < len(names) else f"Factory {k + 1}",
                "num_images": len(items), "class_names": classes,
                "class_distribution": dict(zip(classes, counts)), "non_iid_mode": mode}
        (d / "client_info.json").write_text(json.dumps(info, indent=2))
        infos.append(info)
    return infos
