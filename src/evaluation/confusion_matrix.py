import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_confusion_csv(cm, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + cm["labels"])
        for lab, row in zip(cm["labels"], cm["matrix"]):
            w.writerow([lab] + row)


def plot_confusion_matrix(cm, path, title=""):
    labels, mat = cm["labels"], cm["matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    top = max(max(r) for r in mat) or 1
    for i, row in enumerate(mat):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > top / 2 else "black")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
