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

from compiler_v15 import compile_pipeline, compile_stage
from induce_language_v15 import load_programs
from ir_runtime_v15 import execute
from macro_miner_v15 import canonical, compress, expand, mine_macros, tree_size
from scene_synthesizer_v14 import candidate_stages


def ordered_candidates(examples, macros, induced: bool):
    candidates = {}
    for stage in candidate_stages(examples):
        try:
            ast = compile_stage(stage)
        except (ValueError, KeyError):
            continue
        try:
            if any(
                (len(execute(ast, source)), len(execute(ast, source)[0]))
                != (len(target), len(target[0]))
                for source, target in examples
            ):
                continue
        except Exception:
            continue
        if induced:
            compressed = compress(ast, macros)
            representation = compressed if tree_size(compressed) < tree_size(ast) else ast
        else:
            representation = ast
        candidates[canonical(representation)] = representation
    return sorted(
        candidates.values(),
        key=lambda value: (
            tree_size(value),
            hashlib.sha256(canonical(value).encode()).digest(),
        ),
    )


def solve(candidates, examples, macros):
    tested = 0
    for representation in candidates:
        tested += 1
        ast = expand(representation, macros)
        try:
            if all(execute(ast, source) == target for source, target in examples):
                return tested, representation, ast
        except Exception:
            continue
    return tested, None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V15_SEARCH_BENCHMARK.json")
    args = parser.parse_args()

    programs, examples_by_gate, metadata = load_programs(args.v14_evidence, args.package_root)
    macros = mine_macros(programs, limit=8)
    reports = []
    for item in metadata:
        examples = examples_by_gate[item["gate"]]
        baseline_candidates = ordered_candidates(examples, macros, False)
        induced_candidates = ordered_candidates(examples, macros, True)
        baseline_tested, _, baseline_ast = solve(baseline_candidates, examples, macros)
        induced_tested, induced_repr, induced_ast = solve(induced_candidates, examples, macros)
        if baseline_ast is None or induced_ast is None:
            raise RuntimeError(f"search failed on gate {item['gate']}")
        report = {
            "gate": item["gate"],
            "task": item["task"],
            "baseline_candidate_count": len(baseline_candidates),
            "induced_candidate_count": len(induced_candidates),
            "baseline_tested": baseline_tested,
            "induced_tested": induced_tested,
            "candidate_reduction": baseline_tested / induced_tested,
            "induced_representation_size": tree_size(induced_repr),
            "expanded_ast_size": tree_size(induced_ast),
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    total_baseline = sum(item["baseline_tested"] for item in reports)
    total_induced = sum(item["induced_tested"] for item in reports)
    improved = sum(item["induced_tested"] < item["baseline_tested"] for item in reports)
    worsened = sum(item["induced_tested"] > item["baseline_tested"] for item in reports)
    summary = {
        "schema": "lexigen-v15-search-benchmark-v1",
        "families": len(reports),
        "induced_macro_count": len(macros),
        "total_baseline_candidates_tested": total_baseline,
        "total_induced_candidates_tested": total_induced,
        "aggregate_candidate_reduction": total_baseline / total_induced,
        "families_improved": improved,
        "families_unchanged": len(reports) - improved - worsened,
        "families_worsened": worsened,
        "macro_training_includes_benchmark_programs": True,
        "claim_boundary": "in-sample induced-language search ordering; not blind transfer",
        "reports": reports,
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps({key: summary[key] for key in (
        "families",
        "induced_macro_count",
        "total_baseline_candidates_tested",
        "total_induced_candidates_tested",
        "aggregate_candidate_reduction",
        "families_improved",
        "families_unchanged",
        "families_worsened",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
