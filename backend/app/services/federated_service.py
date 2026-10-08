"""Starts training in a separate Python process and reports live progress from results/status.json."""
import json
import subprocess
import sys
from pathlib import Path
from src.utils.config import load_config
from src.utils.paths import project_root
from src.utils.status import read_status
from . import analytics_service

COMMANDS = {"centralized": "train-centralized", "local_only": "train-local", "fedavg": "train-fedavg",
            "federated_continual": "train-continual", "full": "full-experiment", "prepare_demo": "prepare-demo"}
_PROC = {"p": None}


def _running():
    p = _PROC["p"]
    return p is not None and p.poll() is None


def start_training(experiment: str, demo: bool = True):
    if experiment not in COMMANDS:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose one of {list(COMMANDS)}")
    if _running():
        raise RuntimeError("A training job is already running. Wait for it to finish.")
    root = project_root()
    cmd = [sys.executable, str(root / "run.py"), COMMANDS[experiment]]
    if demo and experiment != "prepare_demo":
        cmd.append("--demo")
    log_dir = root / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / "train.log", "w", encoding="utf-8")
    _PROC["p"] = subprocess.Popen(cmd, cwd=str(root), stdout=log, stderr=subprocess.STDOUT)
    return {"started": True, "command": " ".join(Path(c).name if i == 1 else c for i, c in enumerate(cmd[1:], 1)),
            "experiment": experiment}


def training_status():
    root = project_root()
    st = read_status(root / "results" / "status.json") or {"state": "idle", "message": "No training started yet.", "clients": {}}
    log_f = root / "results" / "logs" / "train.log"
    tail = log_f.read_text(encoding="utf-8", errors="ignore").splitlines()[-12:] if log_f.exists() else []
    st = dict(st)
    st["process_running"] = _running()
    if st.get("state") == "running" and not _running() and _PROC["p"] is not None:
        st["state"] = "failed"
        st["message"] = "Training process ended unexpectedly. See log."
    st["log_tail"] = tail
    return st


def clients_info():
    root = project_root()
    cfg = load_config(root)
    st = read_status(root / "results" / "status.json") or {}
    live = st.get("clients", {})
    out = []
    for i in range(int(cfg["federated"]["num_clients"])):
        f = root / cfg["paths"]["clients"] / f"client_{i + 1}" / "client_info.json"
        if f.exists():
            info = json.loads(f.read_text())
        else:
            names = cfg["federated"].get("client_names", [])
            info = {"client_id": i + 1, "name": names[i] if i < len(names) else f"Factory {i + 1}",
                    "num_images": 0, "class_distribution": {}, "ready": False}
        info.setdefault("ready", True)
        lv = live.get(str(i + 1), {})
        info["training_status"] = lv.get("status", "idle")
        info["last_loss"] = lv.get("loss")
        info["raw_images_leave_factory"] = False
        out.append(info)
    return {"clients": out, "non_iid_mode": cfg["non_iid"].get("mode")}


def federated_rounds(experiment="fedavg"):
    r = analytics_service.all_results().get(experiment)
    live = read_status(project_root() / "results" / "status.json") or {}
    return {"experiment": experiment, "history": r["history"] if r else [],
            "communication": r["communication"] if r else None,
            "live": {"state": live.get("state"), "experiment": live.get("experiment"),
                     "current_round": live.get("current_round", 0), "total_rounds": live.get("total_rounds", 0),
                     "fedavg_status": live.get("fedavg_status"), "history": live.get("history", [])}}
