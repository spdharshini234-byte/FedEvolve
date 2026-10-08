"""Replay (rehearsal) buffer. Stores REFERENCES to old local samples (path, label); the images stay on the factory."""
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.capacity = int(capacity)
        self.items = []          # list of (path, label)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def class_counts(self):
        out = {}
        for _, y in self.items:
            out[y] = out.get(y, 0) + 1
        return out

    def update(self, new_items):
        """Add samples of the task that was just learned, then shrink to capacity keeping classes balanced."""
        pool = list(self.items) + list(new_items)
        classes = sorted({y for _, y in pool})
        if not classes or self.capacity <= 0:
            self.items = []
            return
        if len(pool) <= self.capacity:
            self.items = pool
            return
        by_class = {c: [it for it in pool if it[1] == c] for c in classes}
        quota = {c: 0 for c in classes}
        remaining = self.capacity
        # fair share: give each class an equal quota, redistribute what small classes cannot use
        active = list(classes)
        while remaining > 0 and active:
            share = max(remaining // len(active), 1)
            for c in list(active):
                take = min(share, len(by_class[c]) - quota[c], remaining)
                quota[c] += take
                remaining -= take
                if quota[c] >= len(by_class[c]):
                    active.remove(c)
                if remaining <= 0:
                    break
        kept = []
        for c in classes:
            items = by_class[c]
            idx = self.rng.permutation(len(items))[:quota[c]]
            kept.extend(items[i] for i in idx)
        self.items = kept

    def sample_all(self):
        return list(self.items)
