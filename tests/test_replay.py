from src.continual.replay_buffer import ReplayBuffer
from src.continual.forgetting_metrics import average_forgetting, per_task_forgetting
from src.continual.task_manager import TaskManager


def test_buffer_capacity_and_balance():
    b = ReplayBuffer(10, seed=0)
    b.update([(f"a{i}", 0) for i in range(30)])
    assert len(b) == 10
    b.update([(f"b{i}", 1) for i in range(30)])
    assert len(b) == 10 and b.class_counts() == {0: 5, 1: 5}


def test_buffer_keeps_all_when_small():
    b = ReplayBuffer(100)
    b.update([("x", 0), ("y", 1)])
    assert len(b) == 2


def test_task_manager():
    tm = TaskManager([[0, 1], [2], [3]])
    assert tm.seen_classes(1) == [0, 1, 2] and tm.old_classes(2) == [0, 1, 2] and tm.new_classes(2) == [3]


def test_forgetting_metric():
    R = [[0.9, None], [0.5, 0.8]]       # task 0 dropped from 0.9 to 0.5 -> forgetting 0.4
    assert per_task_forgetting(R) == [0.4] and abs(average_forgetting(R) - 0.4) < 1e-9
