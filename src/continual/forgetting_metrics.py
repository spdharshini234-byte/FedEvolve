"""Forgetting metrics (Chaudhry et al.). R[t][j] = accuracy on task j after training stage t."""


def per_task_forgetting(R):
    """For every task j except the last: max accuracy it ever had (before the end) minus final accuracy."""
    T = len(R)
    out = []
    for j in range(T - 1):
        hist = [R[t][j] for t in range(j, T - 1) if R[t][j] is not None]
        final = R[T - 1][j]
        if not hist or final is None:
            out.append(None)
        else:
            out.append(max(hist) - final)
    return out


def average_forgetting(R):
    vals = [v for v in per_task_forgetting(R) if v is not None]
    return sum(vals) / len(vals) if vals else None
