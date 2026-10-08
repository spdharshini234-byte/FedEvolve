"""Orchestrates every step: data preparation, the four experiments, the no-replay baseline and evaluation.

All numbers saved to results/ come from real training runs executed by this code.
"""
import json
import time
from pathlib import Path
import numpy as np

from .utils.config import load_config, get_paths, task_class_indices, num_epochs
from .utils.device import get_device
from .utils.logger import get_logger
from .utils.seed import set_seed
from .utils.status import StatusTracker
from .utils.paths import project_root, ensure_dir
from .data.augmentation import get_transforms
from .data.client_partition import create_clients
from .data.dataset import make_loader, read_manifest, filter_items, list_images
from .data.preprocessing import verify_dataset, split_dataset
from .data.synthetic import generate_demo_dataset
from .models.resnet_model import build_resnet18
from .models.checkpoint import save_checkpoint, load_checkpoint
from .models.resnet_model import build_model
from .training.evaluate import predict
from .training.loss import class_mask_tensor
from .training.optimizer import build_optimizer
from .training.train_local import train_epochs
from .evaluation.metrics import compute_metrics, subset_accuracy
from .evaluation.confusion_matrix import save_confusion_csv, plot_confusion_matrix
from .evaluation.compare_models import build_comparison, ORDER
from .evaluation.communication_metrics import summarize_communication
from .federated.client import Client
from .federated.server import Server
from .continual.replay_buffer import ReplayBuffer
from .continual.task_manager import TaskManager
from .continual.forgetting_metrics import average_forgetting, per_task_forgetting

DISPLAY = {"centralized": "Centralized", "local_only": "Local Only", "fedavg": "FedAvg",
           "federated_no_replay": "Federated sequential (no replay)",
           "federated_continual": "Federated Continual (FedAvg + Replay)"}


class Context:
    def __init__(self, root=None, overrides=None):
        self.root = Path(root) if root else project_root()
        self.cfg = load_config(self.root, overrides)
        self.paths = get_paths(self.cfg, self.root)
        self.classes = self.cfg["dataset"]["classes"]
        self.device = get_device(self.cfg["project"].get("device", "auto"))
        self.results = ensure_dir(self.paths["results"])
        self.log = get_logger("fcdd", self.results / "logs" / "run.log")
        self.status = StatusTracker(self.results / "status.json")
        names = [t.get("name") for t in self.cfg["continual"]["tasks"]]
        self.tm = TaskManager(task_class_indices(self.cfg), names)
        self.active = self.tm.all_classes()
        self.old = self.tm.old_classes(len(self.tm) - 1)
        self.new = self.tm.new_classes(len(self.tm) - 1)
        self.n_cls = len(self.classes)
        self.seed = int(self.cfg["project"]["seed"])

    def mask(self, classes=None):
        return class_mask_tensor(self.active if classes is None else classes, self.n_cls)


# ----------------------------------------------------------------------------- data steps
def run_verify(ctx):
    rep = verify_dataset(ctx.paths["raw"], ctx.classes, ctx.cfg["dataset"]["min_images_per_class"])
    return rep


def run_prepare(ctx):
    rep = run_verify(ctx)
    if not rep["ok"]:
        raise RuntimeError("Dataset check failed:\n  " + "\n  ".join(rep["errors"]) +
                           "\nPut images in data/raw/<class_name>/ or run: python run.py prepare-demo")
    d = ctx.cfg["dataset"]
    info = split_dataset(ctx.paths["raw"], ctx.paths["processed"], ctx.classes, d["image_size"], d["split"],
                         ctx.seed, ctx.root)
    ctx.log.info(f"Prepared dataset: {info['counts']}")
    return info


def run_create_clients(ctx):
    infos = create_clients(ctx.cfg, ctx.paths["processed"], ctx.paths["clients"])
    for i in infos:
        ctx.log.info(f"{i['name']}: {i['num_images']} images {i['class_distribution']}")
    return infos


def prepare_demo(ctx, force=False):
    """Generate the SYNTHETIC demo dataset (if data/raw is empty), then split it and create the clients."""
    raw = Path(ctx.paths["raw"])
    has_images = any(list_images(raw / c) for c in ctx.classes)
    if force or not has_images:
        ctx.log.info("Generating SYNTHETIC demo dataset (not real factory data)...")
        generate_demo_dataset(raw, ctx.classes, ctx.cfg["dataset"]["demo"]["images_per_class"], ctx.seed)
    run_prepare(ctx)
    return run_create_clients(ctx)


def ensure_ready(ctx, demo=False):
    ready = (Path(ctx.paths["processed"]) / "train.csv").exists() and \
            (Path(ctx.paths["clients"]) / "client_1" / "train.csv").exists()
    if not ready:
        if demo:
            prepare_demo(ctx)
        else:
            raise FileNotFoundError("Data not prepared. Run: python run.py prepare-data && python run.py create-clients "
                                    "(or use --demo to generate and use the synthetic demo dataset).")


# ----------------------------------------------------------------------------- helpers
def _items(ctx, split):
    return filter_items(read_manifest(Path(ctx.paths["processed"]) / f"{split}.csv"), ctx.active)


def _loader(ctx, items, train=False, shuffle=False):
    t = ctx.cfg["training"]
    return make_loader(items, ctx.root, get_transforms(ctx.cfg["dataset"]["image_size"], train, t.get("augmentation", True)),
                       t["batch_size"], shuffle=shuffle, num_workers=t.get("num_workers", 0), seed=ctx.seed)


def _predict(ctx, model, items, classes):
    return predict(model, _loader(ctx, items), ctx.device, ctx.mask(classes))[:2]


def _quick(ctx, model, items, classes):
    yt, yp = _predict(ctx, model, items, classes)
    if len(yt) == 0:
        return 0.0, 0.0
    m = compute_metrics(yt, yp, ctx.classes, classes)["overall"]
    return m["accuracy"], m["macro_f1"]


def _bundle(ctx, yt, yp, classes=None):
    classes = ctx.active if classes is None else classes
    b = compute_metrics(yt, yp, ctx.classes, classes)
    b["old_new"] = {"old_classes": [ctx.classes[c] for c in ctx.old], "new_classes": [ctx.classes[c] for c in ctx.new],
                    "old_class_accuracy": subset_accuracy(yt, yp, ctx.old) if ctx.old else None,
                    "new_class_accuracy": subset_accuracy(yt, yp, ctx.new)}
    return b


def _dataset_info(ctx):
    f = Path(ctx.paths["processed"]) / "dataset_info.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_result(ctx, name, bundle, **extra):
    res = {"experiment": name, "display_name": DISPLAY[name], "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "dataset": _dataset_info(ctx),
           "settings": {"image_size": ctx.cfg["dataset"]["image_size"], "batch_size": ctx.cfg["training"]["batch_size"],
                        "learning_rate": ctx.cfg["training"]["learning_rate"], "seed": ctx.seed,
                        "device": str(ctx.device), "pretrained_requested": ctx.cfg["model"]["pretrained"],
                        "pretrained_weights_loaded": bool(build_resnet18.last_pretrained_loaded),
                        "num_clients": ctx.cfg["federated"]["num_clients"],
                        "classes": ctx.classes, "active_classes": [ctx.classes[c] for c in ctx.active]}}
    res.update(bundle)
    res.update(extra)
    (ctx.results / "metrics").mkdir(parents=True, exist_ok=True)
    (ctx.results / "metrics" / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    save_confusion_csv(res["confusion_matrix"], ctx.results / "confusion_matrices" / f"{name}.csv")
    plot_confusion_matrix(res["confusion_matrix"], ctx.results / "confusion_matrices" / f"{name}.png", DISPLAY[name])
    build_comparison(ctx.results)
    return res


def _ckpt_meta(ctx, name, seen=None):
    return {"experiment": name, "display_name": DISPLAY[name], "image_size": ctx.cfg["dataset"]["image_size"],
            "seen_classes": [ctx.classes[c] for c in (ctx.active if seen is None else seen)],
            "dataset_source": _dataset_info(ctx).get("dataset_source")}


def _setup_fed(ctx):
    template = build_model(ctx.cfg)
    server = Server(template)
    clients = []
    names = ctx.cfg["federated"].get("client_names", [])
    for i in range(int(ctx.cfg["federated"]["num_clients"])):
        f = Path(ctx.paths["clients"]) / f"client_{i + 1}" / "train.csv"
        items = filter_items(read_manifest(f), ctx.active)
        nm = names[i] if i < len(names) else f"Factory {i + 1}"
        clients.append(Client(i + 1, nm, items, ctx.root, ctx.cfg, ctx.device, template, ctx.seed))
    return server, clients


def _init_client_status(ctx, clients):
    for c in clients:
        ctx.status.set_client(c.id, name=c.name, status="waiting", samples=c.num_samples, loss=None,
                              distribution=c.class_distribution(ctx.classes))


# ----------------------------------------------------------------------------- experiments
def run_centralized(ctx, demo=False):
    ensure_ready(ctx, demo)
    set_seed(ctx.seed)
    epochs = num_epochs(ctx.cfg)
    ctx.status.start("centralized", epochs)
    t0 = time.time()
    model = build_model(ctx.cfg)
    train, val, test = _items(ctx, "train"), _items(ctx, "val"), _items(ctx, "test")
    loader = _loader(ctx, train, True, True)
    opt = build_optimizer(model, ctx.cfg["training"])
    hist = []
    for ep in range(epochs):
        st = train_epochs(model, loader, opt, ctx.device, 1, ctx.mask())
        va, vf = _quick(ctx, model, val, ctx.active)
        hist.append({"epoch": ep + 1, "train_loss": st["loss"], "val_accuracy": va, "val_macro_f1": vf})
        ctx.status.update(current_round=ep + 1, message=f"Centralized epoch {ep + 1}/{epochs}: val acc {va:.3f}")
        ctx.log.info(f"[centralized] epoch {ep + 1}/{epochs} loss={st['loss']:.4f} val_acc={va:.3f}")
    yt, yp = _predict(ctx, model, test, ctx.active)
    save_checkpoint(ctx.paths["models"] / "global" / "centralized.pt", model, ctx.classes, _ckpt_meta(ctx, "centralized"))
    res = _save_result(ctx, "centralized", _bundle(ctx, yt, yp), history=hist, forgetting_score=None,
                       communication={"rounds": 0, "total_bytes": 0, "total_mb": 0.0, "raw_image_bytes_transferred": 0,
                                      "note": "Centralized training needs all raw images in ONE place (no privacy)."},
                       duration_seconds=round(time.time() - t0, 1),
                       model_checkpoint="models/global/centralized.pt")
    ctx.status.finish("Centralized training finished")
    return res


def run_local_only(ctx, demo=False):
    ensure_ready(ctx, demo)
    set_seed(ctx.seed)
    epochs = num_epochs(ctx.cfg)
    ctx.status.start("local_only", epochs)
    t0 = time.time()
    server, clients = _setup_fed(ctx)
    _init_client_status(ctx, clients)
    test = _items(ctx, "test")
    all_t, all_p, per_client = [], [], []
    for c in clients:
        ctx.status.set_client(c.id, status="training alone")
        loader = _loader(ctx, c.items, True, True)
        opt = build_optimizer(c.model, ctx.cfg["training"])
        # Local model can only learn classes its factory has seen: mask to those classes.
        local_classes = sorted({y for _, y in c.items}) or ctx.active
        st = train_epochs(c.model, loader, opt, ctx.device, epochs, ctx.mask(local_classes))
        yt, yp = _predict(ctx, c.model, test, local_classes)
        all_t.append(yt); all_p.append(yp)
        m = compute_metrics(yt, yp, ctx.classes, ctx.active)
        per_client.append({"client": c.name, "num_images": c.num_samples,
                           "local_classes": [ctx.classes[k] for k in local_classes], "accuracy": m["overall"]["accuracy"],
                           "macro_f1": m["overall"]["macro_f1"], "train_loss": st["loss"]})
        save_checkpoint(ctx.paths["models"] / "client" / f"local_only_client_{c.id}.pt", c.model, ctx.classes,
                        _ckpt_meta(ctx, "local_only", local_classes))
        ctx.status.set_client(c.id, status="done", loss=st["loss"])
        ctx.log.info(f"[local_only] {c.name}: test acc={m['overall']['accuracy']:.3f}")
    yt, yp = np.concatenate(all_t), np.concatenate(all_p)
    res = _save_result(ctx, "local_only", _bundle(ctx, yt, yp), per_client=per_client, history=[], forgetting_score=None,
                       communication={"rounds": 0, "total_bytes": 0, "total_mb": 0.0, "raw_image_bytes_transferred": 0,
                                      "note": "No collaboration: nothing is sent. Metrics pool the predictions of the 3 local models."},
                       duration_seconds=round(time.time() - t0, 1))
    ctx.status.finish("Local-only training finished")
    return res


def _fed_round(ctx, server, clients, rnd_label, global_round, epochs, active, items_fn):
    """One federated round: broadcast -> local training -> upload weights -> FedAvg."""
    g = server.broadcast()
    losses = {}
    ctx.status.update(fedavg_status="broadcasting global model")
    for c in clients:
        c.receive_global(g)
        items = items_fn(c)
        if len(items) < 2:
            ctx.status.set_client(c.id, status="no data this round")
            continue
        ctx.status.set_client(c.id, status="training locally", samples=len(items))
        w, n, st = c.train_local(epochs, active, items=items, rnd=global_round)
        server.receive_update(c.id, w, n)  # ONLY weights are sent
        losses[c.name] = st["loss"]
        ctx.status.set_client(c.id, status="uploaded weights", loss=st["loss"])
        ctx.log.info(f"  {c.name}: {n} local samples, loss={st['loss']:.4f}")
    ctx.status.update(fedavg_status="aggregating (FedAvg)")
    server.aggregate(len(clients))
    return losses


def run_fedavg(ctx, demo=False):
    ensure_ready(ctx, demo)
    set_seed(ctx.seed)
    rounds, epochs = int(ctx.cfg["federated"]["rounds"]), int(ctx.cfg["federated"]["local_epochs"])
    ctx.status.start("fedavg", rounds)
    t0 = time.time()
    server, clients = _setup_fed(ctx)
    _init_client_status(ctx, clients)
    val, test = _items(ctx, "val"), _items(ctx, "test")
    hist = []
    for r in range(rounds):
        ctx.status.update(current_round=r + 1, message=f"FedAvg round {r + 1}/{rounds}")
        ctx.log.info(f"[fedavg] round {r + 1}/{rounds}")
        losses = _fed_round(ctx, server, clients, r, r + 1, epochs, ctx.active, lambda c: c.items)
        va, vf = _quick(ctx, server.global_model, val, ctx.active)
        hist.append({"round": r + 1, "val_accuracy": va, "val_macro_f1": vf, "client_losses": losses,
                     "communication_mb": round(server.comm.total_bytes / 1024 ** 2, 3)})
        ctx.status.update(fedavg_status="done", global_accuracy=va, global_macro_f1=vf, history=hist)
        ctx.log.info(f"[fedavg] round {r + 1}: global val acc={va:.3f}")
    yt, yp = _predict(ctx, server.global_model, test, ctx.active)
    save_checkpoint(ctx.paths["models"] / "global" / "fedavg.pt", server.global_model, ctx.classes, _ckpt_meta(ctx, "fedavg"))
    res = _save_result(ctx, "fedavg", _bundle(ctx, yt, yp), history=hist, forgetting_score=None,
                       communication=summarize_communication(server.comm), duration_seconds=round(time.time() - t0, 1),
                       clients=[{"name": c.name, "num_images": c.num_samples, "distribution": c.class_distribution(ctx.classes)} for c in clients],
                       model_checkpoint="models/global/fedavg.pt")
    ctx.status.finish("FedAvg finished")
    return res


def run_federated_continual(ctx, use_replay=True, demo=False):
    name = "federated_continual" if use_replay else "federated_no_replay"
    ensure_ready(ctx, demo)
    set_seed(ctx.seed)
    cc = ctx.cfg["continual"]
    T, rpt, epochs = len(ctx.tm), int(cc["rounds_per_task"]), int(cc["local_epochs"])
    ctx.status.start(name, T * rpt)
    t0 = time.time()
    server, clients = _setup_fed(ctx)
    buf = int(cc["replay"]["buffer_size"])
    for c in clients:
        c.replay = ReplayBuffer(buf, seed=ctx.seed + c.id) if use_replay else None
    _init_client_status(ctx, clients)
    val, test = _items(ctx, "val"), _items(ctx, "test")
    R, stages, hist, g_round = [], [], [], 0
    for t in range(T):
        new, seen = ctx.tm.new_classes(t), ctx.tm.seen_classes(t)
        ctx.status.update(current_task=f"{ctx.tm.names[t]}: new classes {[ctx.classes[c] for c in new]}")
        ctx.log.info(f"[{name}] {ctx.tm.names[t]} new={[ctx.classes[c] for c in new]} seen={[ctx.classes[c] for c in seen]}")
        new_items = {c.id: [it for it in c.items if it[1] in new] for c in clients}
        val_seen = filter_items(val, seen)

        def items_fn(c, _n=new_items):
            return _n[c.id] + (c.replay.sample_all() if c.replay else [])

        for r in range(rpt):
            g_round += 1
            ctx.status.update(current_round=g_round, message=f"{ctx.tm.names[t]} round {r + 1}/{rpt}")
            losses = _fed_round(ctx, server, clients, r, g_round, epochs, seen, items_fn)
            va, vf = _quick(ctx, server.global_model, val_seen, seen)
            hist.append({"round": g_round, "task": t + 1, "val_accuracy": va, "val_macro_f1": vf, "client_losses": losses})
            ctx.status.update(fedavg_status="done", global_accuracy=va, global_macro_f1=vf, history=hist)
        if use_replay:  # remember samples of the task just learned (kept locally on each factory)
            for c in clients:
                c.replay.update(new_items[c.id])
        yt, yp = _predict(ctx, server.global_model, filter_items(test, seen), seen)
        R.append([subset_accuracy(yt, yp, ctx.tm.tasks[j]) if j <= t else None for j in range(T)])
        m = compute_metrics(yt, yp, ctx.classes, seen)["overall"]
        old = ctx.tm.old_classes(t)
        stages.append({"task_index": t, "task_name": ctx.tm.names[t], "new_classes": [ctx.classes[c] for c in new],
                       "seen_classes": [ctx.classes[c] for c in seen], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                       "old_class_accuracy": subset_accuracy(yt, yp, old) if old else None,
                       "new_class_accuracy": subset_accuracy(yt, yp, new),
                       "replay_buffers": [{"client": c.name, "size": len(c.replay) if c.replay else 0,
                                           "per_class": {ctx.classes[k]: v for k, v in (c.replay.class_counts() if c.replay else {}).items()}}
                                          for c in clients]})
        ctx.log.info(f"[{name}] after {ctx.tm.names[t]}: acc={m['accuracy']:.3f} old={stages[-1]['old_class_accuracy']} new={stages[-1]['new_class_accuracy']}")
    yt, yp = _predict(ctx, server.global_model, test, ctx.active)
    forgetting = average_forgetting(R)
    save_checkpoint(ctx.paths["models"] / "global" / f"{name}.pt", server.global_model, ctx.classes, _ckpt_meta(ctx, name))
    res = _save_result(ctx, name, _bundle(ctx, yt, yp), history=hist, forgetting_score=forgetting,
                       continual={"replay_enabled": use_replay, "buffer_size_per_client": buf if use_replay else 0,
                                  "tasks": [{"name": ctx.tm.names[i], "classes": [ctx.classes[c] for c in ctx.tm.tasks[i]]} for i in range(T)],
                                  "accuracy_matrix": R, "per_task_forgetting": per_task_forgetting(R), "stages": stages},
                       communication=summarize_communication(server.comm), duration_seconds=round(time.time() - t0, 1),
                       model_checkpoint=f"models/global/{name}.pt")
    ctx.status.finish(f"{DISPLAY[name]} finished")
    return res


def run_continual_both(ctx, demo=False):
    """Plain sequential FedAvg (shows catastrophic forgetting) and FedAvg + replay (reduces it)."""
    a = run_federated_continual(ctx, use_replay=False, demo=demo)
    b = run_federated_continual(ctx, use_replay=True, demo=demo)
    return a, b


# ----------------------------------------------------------------------------- evaluation
def evaluate_all(ctx):
    """Re-evaluate every saved checkpoint on the test split and rebuild the comparison table/plots."""
    test = _items(ctx, "test")
    done = []
    for name in ORDER:
        f = ctx.results / "metrics" / f"{name}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text(encoding="utf-8"))
        if name == "local_only":
            yt_l, yp_l = [], []
            for i in range(int(ctx.cfg["federated"]["num_clients"])):
                model, classes, meta = load_checkpoint(ctx.paths["models"] / "client" / f"local_only_client_{i + 1}.pt")
                seen = [classes.index(c) for c in meta["seen_classes"]]
                yt, yp = _predict(ctx, model, test, seen)
                yt_l.append(yt); yp_l.append(yp)
            yt, yp = np.concatenate(yt_l), np.concatenate(yp_l)
        else:
            model, classes, meta = load_checkpoint(ctx.paths["models"] / "global" / f"{name}.pt")
            yt, yp = _predict(ctx, model, test, [classes.index(c) for c in meta["seen_classes"]])
        res.update(_bundle(ctx, yt, yp))
        f.write_text(json.dumps(res, indent=2), encoding="utf-8")
        save_confusion_csv(res["confusion_matrix"], ctx.results / "confusion_matrices" / f"{name}.csv")
        plot_confusion_matrix(res["confusion_matrix"], ctx.results / "confusion_matrices" / f"{name}.png", DISPLAY[name])
        ctx.log.info(f"[evaluate] {DISPLAY[name]}: acc={res['overall']['accuracy']:.3f} macro-F1={res['overall']['macro_f1']:.3f}")
        done.append(name)
    df = build_comparison(ctx.results)
    return done, df


def full_experiment(ctx, demo=False):
    ensure_ready(ctx, demo)
    run_centralized(ctx)
    run_local_only(ctx)
    run_fedavg(ctx)
    run_continual_both(ctx)
    return evaluate_all(ctx)
