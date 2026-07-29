from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V14 = HERE.parent / "v14"
for path in (HERE, V14):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compiler_v15 import compile_stage
from induce_language_v15 import load_programs
from ir_runtime_v15 import execute
from macro_miner_v15 import canonical, compress, expand, mine_macros, tree_size
from scene_synthesizer_v14 import candidate_stages


def candidate_orders(examples, macros):
    raw = {}
    induced = {}
    for stage in candidate_stages(examples):
        try:
            ast = compile_stage(stage)
        except (ValueError, KeyError):
            continue
        raw[canonical(ast)] = ast
        compressed = compress(ast, macros)
        representation = compressed if tree_size(compressed) < tree_size(ast) else ast
        induced[canonical(representation)] = representation
    key = lambda value: (tree_size(value), hashlib.sha256(canonical(value).encode()).digest())
    return sorted(raw.values(), key=key), sorted(induced.values(), key=key)


def portfolio_order(baseline, induced, macros):
    order = []
    seen = set()
    width = max(len(baseline), len(induced))
    for index in range(width):
        for source in (induced, baseline):
            if index >= len(source):
                continue
            representation = source[index]
            semantic_key = canonical(expand(representation, macros))
            if semantic_key in seen:
                continue
            seen.add(semantic_key)
            order.append(representation)
    return order


def solve(order, examples, macros):
    for tested, representation in enumerate(order, 1):
        ast = expand(representation, macros)
        try:
            if all(execute(ast, source) == target for source, target in examples):
                return tested
        except Exception:
            continue
    return len(order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V15_PORTFOLIO_BENCHMARK.json")
    args = parser.parse_args()

    programs, examples_by_gate, metadata = load_programs(args.v14_evidence, args.package_root)
    macros = mine_macros(programs, limit=8)
    reports = []
    for item in metadata:
        examples = examples_by_gate[item["gate"]]
        baseline, induced = candidate_orders(examples, macros)
        portfolio = portfolio_order(baseline, induced, macros)
        baseline_tested = solve(baseline, examples, macros)
        induced_tested = solve(induced, examples, macros)
        portfolio_tested = solve(portfolio, examples, macros)
        report = {
            "gate": item["gate"],
            "task": item["task"],
            "baseline_tested": baseline_tested,
            "induced_tested": induced_tested,
            "portfolio_tested": portfolio_tested,
            "baseline_to_portfolio": baseline_tested / portfolio_tested,
            "portfolio_slowdown_vs_baseline": portfolio_tested / baseline_tested,
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    baseline_total = sum(item["baseline_tested"] for item in reports)
    portfolio_total = sum(item["portfolio_tested"] for item in reports)
    summary = {
        "schema": "lexigen-v15-portfolio-benchmark-v1",
        "families": len(reports),
        "induced_macro_count": len(macros),
        "baseline_total": baseline_total,
        "portfolio_total": portfolio_total,
        "aggregate_speedup": baseline_total / portfolio_total,
        "worst_slowdown": max(item["portfolio_slowdown_vs_baseline"] for item in reports),
        "families_faster": sum(item["portfolio_tested"] < item["baseline_tested"] for item in reports),
        "families_equal": sum(item["portfolio_tested"] == item["baseline_tested"] for item in reports),
        "families_slower": sum(item["portfolio_tested"] > item["baseline_tested"] for item in reports),
        "macro_training_includes_benchmark_programs": True,
        "claim_boundary": "in-sample safe portfolio; not blind transfer",
        "reports": reports,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps({key: summary[key] for key in (
        "families",
        "baseline_total",
        "portfolio_total",
        "aggregate_speedup",
        "worst_slowdown",
        "families_faster",
        "families_equal",
        "families_slower",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
