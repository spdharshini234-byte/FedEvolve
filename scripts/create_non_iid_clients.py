"""Thin wrapper:  python scripts/create_non_iid_clients.py   ==   python run.py create-clients"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main

if __name__ == "__main__":
    sys.exit(main(["create-clients"]))
