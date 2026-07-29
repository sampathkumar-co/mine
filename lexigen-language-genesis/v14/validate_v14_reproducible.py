from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from portable_scene_runtime_v14 import execute_portable_pipeline
from scene_runtime_v14 import as_grid, execute_pipeline
from scene_synthesizer_v14 import synthesize_scene

GATES = (1, 2, 3, 4, 5, 6, 8, 9, 12)


def load_examples(evidence_root: Path, gate: int):
    path = evidence_root / f"v13-campaign-{gate:02d}" / "redacted-task.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in data["train"]]
    return data["selected_task_id"], examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=HERE / "v14-reproducible-report.json")
    args = parser.parse_args()
    arcgen_root = args.arcgen_root.resolve()
    evidence_root = args.evidence_root.resolve()
    sys.path.insert(0, str(arcgen_root))

    reports = []
    failures = []
    for gate in GATES:
        task, examples = load_examples(evidence_root, gate)
        synthesis = synthesize_scene(examples, max_depth=2)
        if synthesis.pipeline is None:
            raise RuntimeError(f"gate {gate} synthesis failed")
        module = importlib.import_module(f"tasks.task_{task}")
        accepted = attempts = 0
        exact_primary = exact_portable = agreement = 0
        rejected: list[dict[str, object]] = []
        started = time.perf_counter()
        base_seed = 1_900_000 + gate * 50_000
        while accepted < args.cases:
            seed = base_seed + attempts
            attempts += 1
            random.seed(seed)
            try:
                pair = module.generate()
                source = as_grid(pair["input"])
                target = as_grid(pair["output"])
            except Exception as exc:
                if len(rejected) < 20:
                    rejected.append({"seed": seed, "type": type(exc).__name__, "message": str(exc)})
                continue
            primary = execute_pipeline(synthesis.pipeline, source)
            portable = execute_portable_pipeline(synthesis.pipeline, source)
            exact_primary += primary == target
            exact_portable += portable == target
            agreement += primary == portable
            if (primary != target or portable != target or primary != portable) and len(failures) < 20:
                failures.append({"gate": gate, "task": task, "accepted_index": accepted, "seed": seed})
            accepted += 1

        report = {
            "gate": gate,
            "task": task,
            "pipeline": synthesis.pipeline,
            "accepted_cases": accepted,
            "generator_attempts": attempts,
            "generator_rejections": attempts - accepted,
            "rejection_examples": rejected,
            "primary_exact": exact_primary,
            "portable_exact": exact_portable,
            "runtime_agreement": agreement,
            "seconds": round(time.perf_counter() - started, 4),
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = {
        "families": len(reports),
        "accepted_cases_per_family": args.cases,
        "total_accepted_cases": sum(item["accepted_cases"] for item in reports),
        "total_generator_attempts": sum(item["generator_attempts"] for item in reports),
        "total_generator_rejections": sum(item["generator_rejections"] for item in reports),
        "primary_failures": sum(item["accepted_cases"] - item["primary_exact"] for item in reports),
        "portable_failures": sum(item["accepted_cases"] - item["portable_exact"] for item in reports),
        "runtime_disagreements": sum(item["accepted_cases"] - item["runtime_agreement"] for item in reports),
        "failure_examples": failures,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = (
        "families", "total_accepted_cases", "total_generator_attempts",
        "total_generator_rejections", "primary_failures",
        "portable_failures", "runtime_disagreements",
    )
    print("SUMMARY", json.dumps({key: summary[key] for key in keys}, sort_keys=True))
    if summary["primary_failures"] or summary["portable_failures"] or summary["runtime_disagreements"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
