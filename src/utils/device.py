import torch


def get_device(pref: str = "auto") -> torch.device:
    """CPU by default. CUDA is only used if requested (or auto and available)."""
    pref = (pref or "auto").lower()
    if pref == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available -> using CPU")
        return torch.device("cpu")
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
