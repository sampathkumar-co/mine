from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V14 = HERE.parent / "v14"
for path in (HERE, V14):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compiler_v15 import compile_pipeline
from induce_language_v15 import load_programs
from ir_runtime_v15 import as_grid, execute
from macro_miner_v15 import compress, expand, mine_macros
from portable_scene_runtime_v14 import execute_portable_pipeline


def generate_accepted(task: str, arcgen_root: Path, seed: int):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    random.seed(seed)
    pair = importlib.import_module(f"tasks.task_{task}").generate()
    return as_grid(pair["input"]), as_grid(pair["output"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=HERE / "V15_TRANSFER_REPORT.json")
    args = parser.parse_args()

    programs, _, metadata = load_programs(args.v14_evidence, args.package_root)
    macros = mine_macros(programs, limit=8)
    compressed = [compress(program, macros) for program in programs]
    expanded = [expand(program, macros) for program in compressed]
    reports = []
    total_ir_failures = 0
    total_portable_failures = 0
    total_disagreements = 0
    total_rejections = 0

    for ast, item in zip(expanded, metadata):
        gate, task = item["gate"], item["task"]
        accepted = attempts = rejected = 0
        ir_exact = portable_exact = agreement = 0
        started = time.perf_counter()
        while accepted < args.cases:
            seed = 2_500_000 + gate * 100_000 + attempts
            attempts += 1
            try:
                source, target = generate_accepted(task, args.arcgen_root, seed)
            except (ValueError, IndexError, TypeError, RuntimeError):
                rejected += 1
                if attempts > args.cases * 5 + 1000:
                    raise RuntimeError(f"too many generator rejections for gate {gate}")
                continue
            ir_output = execute(ast, source)
            portable_output = as_grid(execute_portable_pipeline(item["pipeline"], source))
            ir_exact += ir_output == target
            portable_exact += portable_output == target
            agreement += ir_output == portable_output
            accepted += 1

        report = {
            "gate": gate,
            "task": task,
            "accepted_cases": accepted,
            "generator_attempts": attempts,
            "generator_rejections": rejected,
            "ir_exact": ir_exact,
            "portable_exact": portable_exact,
            "runtime_agreement": agreement,
            "seconds": round(time.perf_counter() - started, 4),
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        total_ir_failures += accepted - ir_exact
        total_portable_failures += accepted - portable_exact
        total_disagreements += accepted - agreement
        total_rejections += rejected

    summary = {
        "schema": "lexigen-v15-transfer-report-v1",
        "families": len(reports),
        "cases_per_family": args.cases,
        "total_cases": sum(item["accepted_cases"] for item in reports),
        "total_generator_rejections": total_rejections,
        "ir_failures": total_ir_failures,
        "portable_failures": total_portable_failures,
        "runtime_disagreements": total_disagreements,
        "induced_macro_count": len(macros),
        "reports": reports,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps({key: summary[key] for key in (
        "families",
        "total_cases",
        "total_generator_rejections",
        "ir_failures",
        "portable_failures",
        "runtime_disagreements",
        "induced_macro_count",
    )}, sort_keys=True))
    if total_ir_failures or total_portable_failures or total_disagreements:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
