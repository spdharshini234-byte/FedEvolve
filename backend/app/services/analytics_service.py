"""Reads REAL saved results (results/metrics/*.json). Nothing here is made up."""
import json
from pathlib import Path
from src.evaluation.compare_models import ORDER, load_results, comparison_rows
from src.utils.config import load_config
from src.utils.paths import project_root
from src.utils.status import read_status


def _results_dir() -> Path:
    return project_root() / "results"


def all_results():
    return load_results(_results_dir())


def _best(results):
    for n in ("federated_continual", "fedavg", "centralized", "federated_no_replay", "local_only"):
        if n in results:
            return n
    return None


def system_status():
    root = project_root()
    cfg = load_config(root)
    results = all_results()
    status = read_status(_results_dir() / "status.json") or {}
    best = _best(results)
    info_f = root / cfg["paths"]["processed"] / "dataset_info.json"
    info = json.loads(info_f.read_text()) if info_f.exists() else None
    out = {
        "num_factories": int(cfg["federated"]["num_clients"]),
        "num_classes": len(cfg["dataset"]["classes"]),
        "classes": cfg["dataset"]["classes"],
        "configured_rounds": int(cfg["federated"]["rounds"]),
        "current_round": status.get("current_round", 0),
        "total_rounds": status.get("total_rounds", 0),
        "training_state": status.get("state", "idle"),
        "dataset": info,
        "data_ready": info is not None,
        "global_model_status": "trained" if best else "not trained yet",
        "best_model": None,
        "experiments_available": list(results),
    }
    if best:
        r = results[best]
        out["best_model"] = {"experiment": best, "display_name": r["display_name"],
                             "accuracy": r["overall"]["accuracy"], "macro_f1": r["overall"]["macro_f1"],
                             "created_at": r["created_at"]}
    return out


def list_experiments():
    return {"experiments": [{"experiment": n, "display_name": r["display_name"], "created_at": r["created_at"]}
                            for n, r in all_results().items()]}


def get_experiment(name):
    r = all_results().get(name)
    if r is None:
        raise FileNotFoundError(f"No results for experiment '{name}'. Run it first.")
    return r


def model_comparison():
    res = all_results()
    return {"rows": comparison_rows(res), "available": list(res), "expected": ORDER}


def confusion(name):
    r = get_experiment(name)
    return {"experiment": name, "display_name": r["display_name"], "confusion_matrix": r["confusion_matrix"],
            "per_class": r["per_class"], "overall": r["overall"]}


def continual_status():
    res = all_results()
    out = {"available": False, "runs": {}}
    for name in ("federated_no_replay", "federated_continual"):
        r = res.get(name)
        if r:
            out["available"] = True
            out["runs"][name] = {"display_name": r["display_name"], "old_new": r["old_new"],
                                 "forgetting_score": r["forgetting_score"], "continual": r["continual"],
                                 "overall": r["overall"], "created_at": r["created_at"]}
    return out


def privacy_summary():
    res = all_results()
    runs = {}
    for name in ("fedavg", "federated_no_replay", "federated_continual"):
        if name in res:
            c = res[name]["communication"]
            runs[name] = {"display_name": res[name]["display_name"], "rounds": c["rounds"], "total_mb": c["total_mb"],
                          "upload_bytes": c["upload_bytes"], "download_bytes": c["download_bytes"],
                          "raw_image_bytes_transferred": c["raw_image_bytes_transferred"]}
    return {"raw_images_transferred": 0, "model_updates_transferred": bool(runs), "runs": runs,
            "limitations": ["Factories are simulated software clients on one computer.",
                            "Raw training images remain local in the federated setup.",
                            "Basic Federated Learning does not guarantee complete privacy: model updates can leak information.",
                            "Secure aggregation and differential privacy are future improvements."]}
