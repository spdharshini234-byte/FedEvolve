"""Dataset verification and train/val/test splitting."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .dataset import list_images, write_manifest
from .synthetic import DEMO_MARKER
from ..utils.paths import rel_posix


def verify_dataset(raw_dir, classes, min_images=10):
    """Check data/raw/<class>/ folders. Returns a report dict with errors/warnings (never crashes)."""
    raw_dir = Path(raw_dir)
    report = {"raw_dir": str(raw_dir), "classes": {}, "errors": [], "warnings": [],
              "is_demo": (raw_dir / DEMO_MARKER).exists()}
    if not raw_dir.exists():
        report["errors"].append(f"Raw data folder not found: {raw_dir}")
        return report
    for c in classes:
        folder = raw_dir / c
        if not folder.exists():
            report["errors"].append(f"Missing class folder: {folder}")
            report["classes"][c] = {"images": 0, "unreadable": 0}
            continue
        imgs = list_images(folder)
        bad = 0
        for p in imgs:
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                bad += 1
                report["warnings"].append(f"Unreadable image: {p}")
        report["classes"][c] = {"images": len(imgs) - bad, "unreadable": bad}
        if len(imgs) - bad < min_images:
            report["errors"].append(f"Class '{c}' has only {len(imgs) - bad} readable images (need >= {min_images}).")
    extra = [d.name for d in raw_dir.iterdir() if d.is_dir() and d.name not in classes]
    if extra:
        report["warnings"].append(f"Folders not listed in config classes (ignored): {extra}")
    report["ok"] = not report["errors"]
    return report


def split_dataset(raw_dir, processed_dir, classes, image_size, split, seed, root):
    """Stratified train/val/test split. Images are resized and copied to data/processed/<split>/<class>/."""
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    rng = np.random.default_rng(seed)
    for s in ("train", "val", "test"):
        if (processed_dir / s).exists():
            shutil.rmtree(processed_dir / s)  # only our own generated folders are rebuilt
    manifests = {"train": [], "val": [], "test": []}
    for ci, c in enumerate(classes):
        files = list_images(raw_dir / c)
        if not files:
            raise FileNotFoundError(f"No images for class '{c}' in {raw_dir / c}")
        order = rng.permutation(len(files))
        n_tr = int(round(len(files) * split["train"]))
        n_va = int(round(len(files) * split["val"]))
        parts = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
        for s, idxs in parts.items():
            out = processed_dir / s / c
            out.mkdir(parents=True, exist_ok=True)
            for j in idxs:
                src = files[j]
                dst = out / f"{src.stem}.png"
                with Image.open(src) as im:
                    im.convert("RGB").resize((image_size, image_size)).save(dst)
                manifests[s].append((rel_posix(dst, root), ci))
    for s, items in manifests.items():
        write_manifest(items, processed_dir / f"{s}.csv", classes)
    info = {"classes": classes, "image_size": image_size, "seed": seed,
            "is_demo": (raw_dir / DEMO_MARKER).exists(),
            "dataset_source": "synthetic demo images" if (raw_dir / DEMO_MARKER).exists() else "user-provided images",
            "counts": {s: len(v) for s, v in manifests.items()}}
    (processed_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return info
