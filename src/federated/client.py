"""A simulated factory. It owns its images (listed in a local manifest) and trains locally."""
from collections import Counter
from ..data.augmentation import get_transforms
from ..data.dataset import make_loader
from ..models.model_utils import clone_model, get_weights, set_weights
from ..training.loss import class_mask_tensor
from ..training.optimizer import build_optimizer
from ..training.train_local import train_epochs


class Client:
    def __init__(self, client_id, name, items, root, cfg, device, template_model, seed=0):
        self.id, self.name, self.items = client_id, name, list(items)
        self.root, self.cfg, self.device, self.seed = root, cfg, device, seed
        self.model = clone_model(template_model)      # local model
        self.replay = None                              # set by the continual trainer

    @property
    def num_samples(self):
        return len(self.items)

    def class_distribution(self, names):
        c = Counter(y for _, y in self.items)
        return {n: c.get(i, 0) for i, n in enumerate(names)}

    def receive_global(self, state):
        set_weights(self.model, state)

    def train_local(self, epochs, active_classes, items=None, rnd=0):
        """Train on `items` (default: all local items). Returns (weights, num_samples, stats)."""
        items = self.items if items is None else items
        tcfg = self.cfg["training"]
        n_cls = len(self.cfg["dataset"]["classes"])
        loader = make_loader(items, self.root, get_transforms(self.cfg["dataset"]["image_size"], True,
                                                              tcfg.get("augmentation", True)),
                             tcfg["batch_size"], shuffle=True, num_workers=tcfg.get("num_workers", 0),
                             seed=self.seed + 1000 * self.id + rnd)
        opt = build_optimizer(self.model, tcfg)
        stats = train_epochs(self.model, loader, opt, self.device, epochs, class_mask_tensor(active_classes, n_cls))
        return get_weights(self.model), len(items), stats
