"""Path helpers. Everything is relative to the project root so the project works on any OS."""
import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Project root. Can be overridden with the FCDD_ROOT environment variable (used by tests)."""
    env = os.environ.get("FCDD_ROOT")
    return Path(env).resolve() if env else _DEFAULT_ROOT


def resolve(rel, root=None) -> Path:
    """Turn a (relative) path from the config into an absolute path under the project root."""
    p = Path(rel)
    return p if p.is_absolute() else (Path(root) if root else project_root()) / p


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def rel_posix(path, root) -> str:
    """Store paths in manifests as relative POSIX strings (portable between Windows and Linux)."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
