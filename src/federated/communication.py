"""Communication cost accounting.

Only model weights travel between factories and the server. The tracker has a `raw_image_bytes`
counter that is never increased: the server API (see server.py) refuses anything except tensors.
"""


class CommunicationTracker:
    def __init__(self):
        self.download_bytes = 0   # server -> clients (global model)
        self.upload_bytes = 0     # clients -> server (local models)
        self.raw_image_bytes = 0  # always 0: raw images are never transferred
        self.rounds = 0
        self.per_round = []

    def record_round(self, num_clients, model_bytes, num_uploads=None):
        up = model_bytes * (num_clients if num_uploads is None else num_uploads)
        down = model_bytes * num_clients
        self.upload_bytes += up
        self.download_bytes += down
        self.rounds += 1
        self.per_round.append({"round": self.rounds, "upload_bytes": up, "download_bytes": down})

    @property
    def total_bytes(self):
        return self.upload_bytes + self.download_bytes

    def summary(self):
        return {"rounds": self.rounds, "upload_bytes": self.upload_bytes, "download_bytes": self.download_bytes,
                "total_bytes": self.total_bytes, "total_mb": round(self.total_bytes / 1024 ** 2, 3),
                "raw_image_bytes_transferred": self.raw_image_bytes, "model_updates_transferred": self.rounds > 0,
                "per_round": self.per_round}
