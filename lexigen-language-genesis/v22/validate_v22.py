from __future__ import annotations
import argparse, hashlib, importlib, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for folder in (HERE,):
    if str(folder) not in sys.path: sys.path.insert(0, str(folder))
from runtime_v22 import as_grid, execute
from portable_runtime_v22 import run as portable_run
from synthesizer_v22 import synthesize


def seed(task: str, index: int) -> int:
    text = f"lexigen-v22-fresh:{task}:{index}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=HERE / "V22_REPORT.json")
    args = parser.parse_args()
    sys.path.insert(0, str(args.arcgen_root))
    pool = json.loads((HERE.parent / "v21" / "V21_PRECOMMIT.json").read_text())[
        "selected_discovery_task_ids"
    ]
    reports = []
    for task in pool:
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
        if result.program is None:
            reports.append(report)
            continue
        accepted = attempts = 0
        while accepted < args.cases and attempts < args.cases * 4:
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

    solved = [r for r in reports if r["program"] is not None]
    summary = {
        "schema": "lexigen-v22-relational-sketch-report-v1",
        "pool_tasks": len(pool),
        "solved_task_families": len(solved),
        "total_fresh_cases": sum(r["fresh_cases"] for r in reports),
        "primary_failures": sum(r["fresh_cases"] - r["primary_exact"] for r in reports),
        "portable_failures": sum(r["fresh_cases"] - r["portable_exact"] for r in reports),
        "runtime_disagreements": sum(r["fresh_cases"] - r["runtime_agreement"] for r in reports),
        "reports": reports,
        "integrity_verdict": "human_authored_search_schemas_after_visible_task_inspection",
        "completion_gate_passed": False,
        "world_level_breakthrough": False,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("SUMMARY", json.dumps({k: summary[k] for k in (
        "pool_tasks", "solved_task_families", "total_fresh_cases",
        "primary_failures", "portable_failures", "runtime_disagreements",
        "completion_gate_passed")}, sort_keys=True))

if __name__ == "__main__":
    main()
