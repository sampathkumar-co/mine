from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V14 = HERE.parent / "v14"
for folder in (HERE, V14):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from induce_language_v15 import load_programs
from ir_runtime_v15 import as_grid, execute
from macro_miner_v15 import compress, expand, mine_macros
from portable_ir_runtime_v15 import execute_portable_ir
from portable_scene_runtime_v14 import execute_portable_pipeline


def generate(task: str, arcgen_root: Path, seed: int):
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
    parser.add_argument("--output", type=Path, default=HERE / "V15_INDEPENDENT_REPORT.json")
    args = parser.parse_args()

    programs, examples_by_gate, metadata = load_programs(args.v14_evidence, args.package_root)
    macros = mine_macros(programs, limit=8)
    expanded = [expand(compress(program, macros), macros) for program in programs]
    reports = []
    totals = {"primary": 0, "portable_ir": 0, "portable_v14": 0, "agreement": 0, "rejections": 0}

    for ast, item in zip(expanded, metadata):
        gate, task = item["gate"], item["task"]
        demo_primary = sum(execute(ast, source) == target for source, target in examples_by_gate[gate])
        demo_portable = sum(execute_portable_ir(ast, source) == target for source, target in examples_by_gate[gate])
        if demo_primary != len(examples_by_gate[gate]) or demo_portable != len(examples_by_gate[gate]):
            raise RuntimeError(f"demonstration replay failed for gate {gate}")

        accepted = attempts = rejected = 0
        primary_exact = portable_ir_exact = portable_v14_exact = agreement = 0
        while accepted < args.cases:
            seed = 3_100_000 + gate * 100_000 + attempts
            attempts += 1
            try:
                source, target = generate(task, args.arcgen_root, seed)
            except (ValueError, IndexError, TypeError, RuntimeError):
                rejected += 1
                if attempts > args.cases * 5 + 1000:
                    raise RuntimeError(f"too many generator rejections for gate {gate}")
                continue
            primary = execute(ast, source)
            portable_ir = execute_portable_ir(ast, source)
            portable_v14 = as_grid(execute_portable_pipeline(item["pipeline"], source))
            primary_exact += primary == target
            portable_ir_exact += portable_ir == target
            portable_v14_exact += portable_v14 == target
            agreement += primary == portable_ir == portable_v14
            accepted += 1

        report = {
            "gate": gate,
            "task": task,
            "accepted_cases": accepted,
            "generator_attempts": attempts,
            "generator_rejections": rejected,
            "primary_exact": primary_exact,
            "portable_ir_exact": portable_ir_exact,
            "portable_v14_exact": portable_v14_exact,
            "three_runtime_agreement": agreement,
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        totals["primary"] += accepted - primary_exact
        totals["portable_ir"] += accepted - portable_ir_exact
        totals["portable_v14"] += accepted - portable_v14_exact
        totals["agreement"] += accepted - agreement
        totals["rejections"] += rejected

    summary = {
        "schema": "lexigen-v15-independent-ir-report-v1",
        "families": len(reports),
        "cases_per_family": args.cases,
        "total_cases": sum(item["accepted_cases"] for item in reports),
        "induced_macro_count": len(macros),
        "primary_failures": totals["primary"],
        "portable_ir_failures": totals["portable_ir"],
        "portable_v14_failures": totals["portable_v14"],
        "three_runtime_disagreements": totals["agreement"],
        "generator_rejections": totals["rejections"],
        "reports": reports,
    }
    args.output.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("SUMMARY", json.dumps({key: summary[key] for key in (
        "families",
        "total_cases",
        "induced_macro_count",
        "primary_failures",
        "portable_ir_failures",
        "portable_v14_failures",
        "three_runtime_disagreements",
        "generator_rejections",
    )}, sort_keys=True))
    if any(summary[key] for key in (
        "primary_failures",
        "portable_ir_failures",
        "portable_v14_failures",
        "three_runtime_disagreements",
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
