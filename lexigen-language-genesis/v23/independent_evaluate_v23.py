from __future__ import annotations
import argparse, importlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V22 = HERE.parent / "v22"
if str(V22) not in sys.path:
    sys.path.insert(0, str(V22))
from portable_runtime_v22 import run


def candidates():
    yield {"op": "pack_columns"}
    for marker in range(10):
        for paint in range(10):
            if paint != marker:
                yield {"op": "connect_aligned", "marker_colour": marker, "paint_colour": paint}
    for axis in ("vertical", "horizontal"):
        for marker in range(10):
            for yes in range(10):
                for no in range(10):
                    if yes != no:
                        yield {"op": "classify_reflection", "axis": axis,
                               "marker_colour": marker, "equal_colour": yes,
                               "unequal_colour": no}

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
        examples = [(x["input"], x["output"]) for x in module.validate()["train"]]
        tested = exact = invalid = 0
        for program in candidates():
            tested += 1
            try:
                if all(run(program, source) == target for source, target in examples):
                    exact += 1
            except Exception:
                invalid += 1
        reports.append({"task": task, "candidates_tested": tested,
                        "exact_program_count": exact,
                        "runtime_invalid_candidates": invalid})
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
    result = {
        "schema": "lexigen-v23-independent-heldout-report-v1",
        "task_count": len(reports),
        "total_candidates_tested": sum(x["candidates_tested"] for x in reports),
        "tasks_with_exact_program": sum(x["exact_program_count"] > 0 for x in reports),
        "total_exact_programs": sum(x["exact_program_count"] for x in reports),
        "reports": reports,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps({k: result[k] for k in (
        "task_count", "total_candidates_tested", "tasks_with_exact_program",
        "total_exact_programs")}, sort_keys=True))


if __name__ == "__main__":
    main()
