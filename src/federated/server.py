"""Central federated server. It only ever sees model weights, never images."""
import torch
from .fedavg import fedavg
from .communication import CommunicationTracker
from ..models.model_utils import get_weights, set_weights, state_nbytes


class Server:
    def __init__(self, global_model):
        self.global_model = global_model
        self.round = 0
        self.comm = CommunicationTracker()
        self._inbox = []

    def broadcast(self):
        """Global weights sent to the clients."""
        return get_weights(self.global_model)

    def receive_update(self, client_id, state, num_samples):
        """Accept ONLY a dict of tensors + a sample count. Anything else (e.g. images) is rejected."""
        if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
            raise TypeError("Server accepts only model weights (dict of tensors), not raw data.")
        self._inbox.append((state, int(num_samples)))

    def aggregate(self, num_clients_downloading):
        """FedAvg over received updates; updates the global model; records communication cost."""
        if not self._inbox:
            raise RuntimeError("No client updates received this round")
        new_state = fedavg(self._inbox)
        set_weights(self.global_model, new_state)
        self.comm.record_round(num_clients_downloading, state_nbytes(new_state), num_uploads=len(self._inbox))
        self._inbox = []
        self.round += 1
        return new_state
