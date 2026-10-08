"""Shows / writes the continual-learning task plan defined in configs/continual_config.yaml.

   python scripts/create_continual_tasks.py   -> writes data/processed/continual_tasks.json
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import Context


def main():
    ctx = Context()
    plan = []
    for i, t in enumerate(ctx.tm.tasks):
        plan.append({"task": ctx.tm.names[i], "new_classes": [ctx.classes[c] for c in t],
                     "seen_after_task": [ctx.classes[c] for c in ctx.tm.seen_classes(i)]})
    out = Path(ctx.paths["processed"]) / "continual_tasks.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2))
    for p in plan:
        print(f"{p['task']}: learn {p['new_classes']} (model knows {p['seen_after_task']})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
