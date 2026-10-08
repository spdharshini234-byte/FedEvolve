"""Keeps track of which classes belong to which continual-learning task."""


class TaskManager:
    def __init__(self, tasks, task_names=None):
        self.tasks = [list(t) for t in tasks]          # list of lists of class indices
        self.names = task_names or [f"Task {i + 1}" for i in range(len(self.tasks))]

    def __len__(self):
        return len(self.tasks)

    def new_classes(self, t):
        return list(self.tasks[t])

    def seen_classes(self, t):
        return sorted(c for k in range(t + 1) for c in self.tasks[k])

    def old_classes(self, t):
        return sorted(c for k in range(t) for c in self.tasks[k])

    def all_classes(self):
        return self.seen_classes(len(self.tasks) - 1)

    def task_of_class(self, c):
        for i, t in enumerate(self.tasks):
            if c in t:
                return i
        raise KeyError(c)
