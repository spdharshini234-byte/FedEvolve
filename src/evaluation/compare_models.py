"""Builds results/experiment_results.csv and comparison plots from the REAL saved experiment metrics."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["centralized", "local_only", "fedavg", "federated_no_replay", "federated_continual"]


def load_results(results_dir):
    out = {}
    for name in ORDER:
        f = Path(results_dir) / "metrics" / f"{name}.json"
        if f.exists():
            out[name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def comparison_rows(results):
    rows = []
    for name, r in results.items():
        o, on = r["overall"], r.get("old_new", {})
        rows.append({"experiment": name, "display_name": r["display_name"], "accuracy": o["accuracy"],
                     "macro_precision": o["macro_precision"], "macro_recall": o["macro_recall"],
                     "macro_f1": o["macro_f1"], "old_class_accuracy": on.get("old_class_accuracy"),
                     "new_class_accuracy": on.get("new_class_accuracy"),
                     "forgetting_score": r.get("forgetting_score"),
                     "communication_mb": r.get("communication", {}).get("total_mb", 0.0),
                     "rounds": r.get("communication", {}).get("rounds", 0),
                     "dataset_source": r.get("dataset", {}).get("dataset_source"),
                     "created_at": r.get("created_at")})
    return rows


def build_comparison(results_dir):
    results_dir = Path(results_dir)
    results = load_results(results_dir)
    rows = comparison_rows(results)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "experiment_results.csv", index=False)
    plots = results_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    # 1. accuracy / macro-F1 bars
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["accuracy"], 0.4, label="Accuracy")
    ax.bar([i + 0.2 for i in x], df["macro_f1"], 0.4, label="Macro-F1")
    ax.set_xticks(list(x), df["display_name"], rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Model comparison (test set)")
    fig.tight_layout()
    fig.savefig(plots / "model_comparison.png", dpi=110)
    plt.close(fig)
    # 2. old-class accuracy per continual stage (forgetting)
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    for name in ("federated_no_replay", "federated_continual"):
        st = results.get(name, {}).get("continual", {}).get("stages", [])
        pts = [(s["task_index"] + 1, s["old_class_accuracy"]) for s in st if s["old_class_accuracy"] is not None]
        if pts:
            ax.plot(*zip(*pts), marker="o", label=results[name]["display_name"])
            drawn = True
    if drawn:
        ax.set_xlabel("Task learned")
        ax.set_ylabel("Accuracy on OLD classes")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plots / "forgetting.png", dpi=110)
    plt.close(fig)
    # 3. training curves
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    for name, r in results.items():
        h = r.get("history", [])
        if h and "val_accuracy" in h[0]:
            ax.plot([e.get("round", e.get("epoch")) for e in h], [e["val_accuracy"] for e in h], label=r["display_name"])
            drawn = True
    if drawn:
        ax.set_xlabel("Round / epoch")
        ax.set_ylabel("Validation accuracy")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plots / "training_curves.png", dpi=110)
    plt.close(fig)
    return df
