import sys
from pathlib import Path

# make `src` importable when running `uvicorn backend.app.main:app` from the project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
