"""Metrics: accuracy, precision, recall, F1, macro-F1, confusion matrix, per-class, old/new class accuracy."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)


def compute_metrics(y_true, y_pred, class_names, active=None):
    """class_names: all class names. active: class indices evaluated (default all)."""
    active = list(range(len(class_names))) if active is None else list(active)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("No evaluation samples.")
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=active, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=active, average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y_true, y_pred, labels=active, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=active)
    return {
        "overall": {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(mp),
                    "macro_recall": float(mr), "macro_f1": float(mf), "weighted_f1": float(wf),
                    "num_samples": int(len(y_true))},
        "per_class": [{"class": class_names[c], "precision": float(p[i]), "recall": float(r[i]),
                       "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(active)],
        "confusion_matrix": {"labels": [class_names[c] for c in active], "matrix": cm.tolist()},
    }


def subset_accuracy(y_true, y_pred, class_indices):
    """Accuracy over test samples whose TRUE class is in class_indices. None if no such samples."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    m = np.isin(y_true, list(class_indices))
    if not m.any():
        return None
    return float((y_true[m] == y_pred[m]).mean())
