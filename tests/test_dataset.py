from pathlib import Path
from src.data.client_partition import partition_items
from src.data.dataset import ManifestDataset, read_manifest
from src.data.preprocessing import verify_dataset
from src.data.synthetic import generate_demo_dataset


def test_verify_and_missing(tmp_path):
    rep = verify_dataset(tmp_path / "nope", ["a"])
    assert not rep["errors"] == []
    generate_demo_dataset(tmp_path / "raw", ["scratch", "crack"], 6, size=32)
    rep = verify_dataset(tmp_path / "raw", ["scratch", "crack"], min_images=5)
    assert rep["ok"] and rep["is_demo"] and rep["classes"]["scratch"]["images"] == 6


def test_processed_split_and_loading(tiny_project):
    train = read_manifest(tiny_project / "data/processed/train.csv")
    ds = ManifestDataset(train, tiny_project)
    img, y = ds[0]
    assert img.size == (32, 32) and y in range(4)


def test_non_iid_partition_is_not_equal_split():
    items = [(f"p{i}", i % 3) for i in range(90)]
    w = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]]
    parts = partition_items(items, 3, 3, "matrix", w, seed=1)
    assert sum(len(p) for p in parts) == 90                       # nothing lost
    assert len({x for p in parts for x in p}) == 90               # no duplicates
    assert sum(1 for _, y in parts[0] if y == 0) > sum(1 for _, y in parts[0] if y == 1) * 3  # skewed
    assert all(y == 2 for _, y in parts[2])
