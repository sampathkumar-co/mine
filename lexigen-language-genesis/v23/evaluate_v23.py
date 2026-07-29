from __future__ import annotations
import argparse, hashlib, importlib, json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V22 = HERE.parent / "v22"
for folder in (HERE, V22):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
from runtime_v22 import as_grid, execute
from portable_runtime_v22 import run as portable_run
from synthesizer_v22 import synthesize


def seed(task: str, index: int) -> int:
    return int(hashlib.sha256(f"lexigen-v23:{task}:{index}".encode()).hexdigest()[:8], 16)


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
        published = module.validate()["train"]
        examples = [(as_grid(x["input"]), as_grid(x["output"])) for x in published]
        result = synthesize(examples)
        report = {
            "task": task,
            "published_examples": len(examples),
            "candidates_tested": result.candidates_tested,
            "exact_program_count": result.exact_count,
            "program": result.program,
            "fresh_cases": 0,
            "primary_exact": 0,
            "portable_exact": 0,
            "runtime_agreement": 0,
            "generator_rejections": 0,
        }
        if result.program is not None:
            accepted = attempts = 0
            target_cases = int(precommit["fresh_accepted_cases_per_survivor"])
            while accepted < target_cases and attempts < target_cases * 4:
                random.seed(seed(task, attempts)); attempts += 1
                try:
                    pair = module.generate()
                    source, target = as_grid(pair["input"]), as_grid(pair["output"])
                except (ValueError, IndexError, TypeError, RuntimeError):
                    report["generator_rejections"] += 1
                    continue
                primary = execute(result.program, source)
                portable = tuple(tuple(row) for row in portable_run(result.program, source))
                report["primary_exact"] += primary == target
                report["portable_exact"] += portable == target
                report["runtime_agreement"] += primary == portable
                accepted += 1
            report["fresh_cases"] = accepted
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    survivors = [r for r in reports if r["program"] is not None]
    result = {
        "schema": "lexigen-v23-heldout-horizon-report-v1",
        "task_count": len(reports),
        "tasks_with_exact_demonstration_program": len(survivors),
        "total_fresh_cases": sum(r["fresh_cases"] for r in reports),
        "primary_failures": sum(r["fresh_cases"] - r["primary_exact"] for r in reports),
        "portable_failures": sum(r["fresh_cases"] - r["portable_exact"] for r in reports),
        "runtime_disagreements": sum(r["fresh_cases"] - r["runtime_agreement"] for r in reports),
        "reports": reports,
        "external_real_world_discovery_produced": False,
        "independent_outside_human_reproduction": False,
        "world_level_breakthrough": False,
        "horizon_verdict": "No held-out transfer and no external real-world discovery through v23.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps({k: result[k] for k in (
        "task_count", "tasks_with_exact_demonstration_program", "total_fresh_cases",
        "primary_failures", "portable_failures", "runtime_disagreements",
        "external_real_world_discovery_produced", "world_level_breakthrough")}, sort_keys=True))

if __name__ == "__main__":
    main()
