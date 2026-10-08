"""Dataset utilities: manifests (CSV lists of image paths + labels) and a PyTorch Dataset."""
import csv
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_IMG_CACHE = {}  # small in-memory cache so images are decoded only once


def clear_image_cache():
    _IMG_CACHE.clear()


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)


def write_manifest(items, csv_path, classes=None):
    """items = list of (relative_posix_path, label_index)."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        for p, y in items:
            w.writerow([p, int(y), classes[y] if classes else ""])


def read_manifest(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}. Did you run prepare-data / create-clients?")
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [(r["path"], int(r["label"])) for r in csv.DictReader(f)]


def filter_items(items, class_indices):
    keep = set(class_indices)
    return [(p, y) for p, y in items if y in keep]


def class_counts(items, num_classes):
    counts = [0] * num_classes
    for _, y in items:
        counts[y] += 1
    return counts


class ManifestDataset(Dataset):
    """Loads images listed in `items` (relative paths under `root`)."""

    def __init__(self, items, root, transform=None):
        self.items = list(items)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def _load(self, rel):
        key = (str(self.root), rel)
        if key not in _IMG_CACHE:
            path = self.root / rel
            try:
                with Image.open(path) as im:
                    _IMG_CACHE[key] = im.convert("RGB").copy()
            except FileNotFoundError:
                raise FileNotFoundError(f"Image listed in manifest is missing: {path}")
        return _IMG_CACHE[key]

    def __getitem__(self, i):
        rel, y = self.items[i]
        img = self._load(rel)
        if self.transform:
            img = self.transform(img)
        return img, y


def make_loader(items, root, transform, batch_size, shuffle=False, num_workers=0, seed=None):
    ds = ManifestDataset(items, root, transform)
    drop_last = shuffle and len(ds) > batch_size and len(ds) % batch_size == 1  # avoid a size-1 batch (BatchNorm)
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last, generator=gen)
