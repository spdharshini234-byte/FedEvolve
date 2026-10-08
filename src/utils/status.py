"""Live training status written to results/status.json so the web app can show real progress."""
import json
import os
import time
from pathlib import Path


class StatusTracker:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.data = {"state": "idle", "experiment": None, "phase": None, "current_round": 0, "total_rounds": 0,
                     "current_task": None, "fedavg_status": "idle", "message": "No training started yet.",
                     "clients": {}, "started_at": None, "updated_at": None}

    def update(self, **kw):
        self.data.update(kw)
        self._save()

    def set_client(self, cid, **kw):
        self.data["clients"].setdefault(str(cid), {}).update(kw)
        self._save()

    def start(self, experiment, total_rounds=0):
        self.data.update(state="running", experiment=experiment, total_rounds=total_rounds, current_round=0,
                         started_at=time.strftime("%Y-%m-%d %H:%M:%S"), message=f"Started {experiment}",
                         fedavg_status="idle", clients={})
        self._save()

    def finish(self, message="Finished"):
        self.update(state="completed", message=message, fedavg_status="idle")

    def fail(self, message):
        self.update(state="failed", message=message)

    def _save(self):
        if not self.path:
            return
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)  # atomic replace so readers never see half-written JSON


def read_status(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
