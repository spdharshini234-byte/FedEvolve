"""Thin wrapper:  python scripts/prepare_dataset.py   ==   python run.py prepare-data"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main

if __name__ == "__main__":
    sys.exit(main(["prepare-data"]))
