"""Command-line interface.   python run.py <command> [--demo]

Commands: verify-data, prepare-demo, prepare-data, create-clients, train-centralized, train-local,
          train-fedavg, train-continual, evaluate, full-experiment
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

COMMANDS = ["verify-data", "prepare-demo", "prepare-data", "create-clients", "train-centralized", "train-local",
            "train-fedavg", "train-continual", "evaluate", "full-experiment"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Federated Continual Defect Detection (software-only simulation)")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--demo", action="store_true",
                    help="if data is not prepared, generate + prepare the SYNTHETIC demo dataset automatically")
    args = ap.parse_args(argv)

    from src import pipeline as pl
    ctx = pl.Context(ROOT)
    ctx.log.info(f"Command: {args.command} | device: {ctx.device} | root: {ctx.root}")
    try:
        if args.command == "verify-data":
            rep = pl.run_verify(ctx)
            for c, v in rep["classes"].items():
                print(f"  {c:15s} images={v['images']:5d} unreadable={v['unreadable']}")
            for w in rep["warnings"]:
                print("  [warn]", w)
            for e in rep["errors"]:
                print("  [error]", e)
            print("Dataset OK" if rep["ok"] else "Dataset NOT ready (try: python run.py prepare-demo)")
            return 0 if rep["ok"] else 1
        if args.command == "prepare-demo":
            pl.prepare_demo(ctx)
        elif args.command == "prepare-data":
            pl.run_prepare(ctx)
        elif args.command == "create-clients":
            pl.run_create_clients(ctx)
        elif args.command == "train-centralized":
            pl.run_centralized(ctx, args.demo)
        elif args.command == "train-local":
            pl.run_local_only(ctx, args.demo)
        elif args.command == "train-fedavg":
            pl.run_fedavg(ctx, args.demo)
        elif args.command == "train-continual":
            pl.run_continual_both(ctx, args.demo)
        elif args.command == "evaluate":
            done, df = pl.evaluate_all(ctx)
            if df is None or len(df) == 0:
                print("No experiment results found. Run a training command first.")
                return 1
            print(df[["display_name", "accuracy", "macro_f1", "old_class_accuracy", "new_class_accuracy",
                      "forgetting_score", "communication_mb"]].to_string(index=False))
        elif args.command == "full-experiment":
            _, df = pl.full_experiment(ctx, args.demo)
            print(df[["display_name", "accuracy", "macro_f1", "old_class_accuracy", "new_class_accuracy",
                      "forgetting_score", "communication_mb"]].to_string(index=False))
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        ctx.status.fail(str(e))
        print(f"\n[ERROR] {e}")
        return 1
    except Exception as e:  # keep the web app informed, then re-raise
        ctx.status.fail(f"{type(e).__name__}: {e}")
        raise
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
