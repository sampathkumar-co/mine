from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V22 = HERE.parent / "v22"
for folder in (HERE, V22):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
from portable_runtime_v22 import run


def inventory():
    yield {"op": "pack_columns"}
    for marker in tuple(range(10)):
        for paint in tuple(range(10)):
            if marker != paint:
                yield {"op": "connect_aligned", "marker_colour": marker, "paint_colour": paint}
    for axis in ("vertical", "horizontal"):
        for marker in tuple(range(10)):
            for equal_colour in tuple(range(10)):
                for unequal_colour in tuple(range(10)):
                    if equal_colour != unequal_colour:
                        yield {
                            "op": "classify_reflection", "axis": axis,
                            "marker_colour": marker, "equal_colour": equal_colour,
                            "unequal_colour": unequal_colour,
                        }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.arcgen_root))
    precommit = json.loads((HERE / "V23_PRECOMMIT.json").read_text())
    reports = []
    for task in precommit["selected_validation_task_ids"]:
        module = importlib.import_module(f"tasks.task_{task}")
        examples = module.validate()["train"]
        exact = 0
        tested = 0
        for program in inventory():
            tested += 1
            try:
                if all(run(program, item["input"]) == item["output"] for item in examples):
                    exact += 1
            except Exception:
                pass
        reports.append({
            "task": task,
            "published_examples": len(examples),
            "candidates_tested": tested,
            "exact_program_count": exact,
        })
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
    result = {
        "schema": "lexigen-v23-independent-search-report-v1",
        "task_count": len(reports),
        "tasks_with_exact_program": sum(r["exact_program_count"] > 0 for r in reports),
        "total_candidates_tested": sum(r["candidates_tested"] for r in reports),
        "reports": reports,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps({
        "task_count": result["task_count"],
        "tasks_with_exact_program": result["tasks_with_exact_program"],
        "total_candidates_tested": result["total_candidates_tested"],
    }, sort_keys=True))

if __name__ == "__main__":
    main()
