import torch


def build_optimizer(model, tcfg):
    params = [p for p in model.parameters() if p.requires_grad]
    lr, wd = float(tcfg["learning_rate"]), float(tcfg.get("weight_decay", 0.0))
    if tcfg.get("optimizer", "adam") == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)
