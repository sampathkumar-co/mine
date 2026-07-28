from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from portable_runtime_v7 import as_grid as portable_grid
from portable_runtime_v7 import execute_portable
from semantic_ast_v7 import as_grid, canonical_json, execute_ast, synthesize_ast


def load_published(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(as_grid(x["input"]), as_grid(x["output"])) for x in payload["train"]], payload


def generate_cases(arcgen_root: Path, seeds: range):
    sys.path.insert(0, str(arcgen_root))
    task = importlib.import_module("tasks.task_228f6490")
    cases = []
    for seed in seeds:
        random.seed(seed)
        item = task.generate()
        cases.append((seed, (as_grid(item["input"]), as_grid(item["output"]))))
    return cases


def run(training: Path, arcgen_root: Path, output_dir: Path) -> dict[str, Any]:
    examples, source = load_published(training)
    discovery = generate_cases(arcgen_root, range(10000, 10100))
    heldout = generate_cases(arcgen_root, range(10100, 10300))
    counterexamples: list[int] = []

    for _ in range(8):
        result = synthesize_ast(examples)
        if result.ast is None or result.ambiguous:
            raise AssertionError("v7 CEGIS did not produce a unique minimum-description AST")
        failure = next(
            ((seed, pair) for seed, pair in discovery if execute_ast(result.ast, pair[0]) != pair[1]),
            None,
        )
        if failure is None:
            break
        counterexamples.append(failure[0])
        examples.append(failure[1])
        discovery = [item for item in discovery if item[0] != failure[0]]
    else:
        raise AssertionError("v7 CEGIS exceeded frozen counterexample budget")

    result = synthesize_ast(examples)
    assert result.ast is not None
    heldout_failures = [seed for seed, pair in heldout if execute_ast(result.ast, pair[0]) != pair[1]]
    portable_ok = all(
        execute_ast(result.ast, pair[0]) == execute_portable(result.ast, portable_grid(pair[0])) == pair[1]
        for _, pair in heldout
    )
    report = {
        "version": "v7",
        "status": "retrospective external-family mechanism validation; no breakthrough claim",
        "arcgen_commit": "a15cbdb44c776610aeeb9f487a06af875d3d0878",
        "source_task_id": source.get("selected_task_id"),
        "source_sealed_outputs_accessed": False,
        "discovery_seed_range": [10000, 10099],
        "heldout_seed_range": [10100, 10299],
        "counterexample_seeds": counterexamples,
        "semantic_ast": result.ast,
        "candidates_tested_final": result.candidates_tested,
        "exact_candidate_count_final": result.exact_candidate_count,
        "heldout_case_count": len(heldout),
        "heldout_failures": heldout_failures,
        "heldout_accuracy": 1.0 - len(heldout_failures) / len(heldout),
        "portable_runtime_agreement": portable_ok,
        "claim_boundary": (
            "The task family and seed ranges were examined during v7 development, so this is not a blind result. "
            "It demonstrates counterexample-guided refinement from area matching to a source-role predicate."
        ),
    }
    if heldout_failures or not portable_ok:
        raise AssertionError("v7 failed deterministic held-out family validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "v7-cegis-report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "counterexamples": counterexamples,
        "heldout_accuracy": report["heldout_accuracy"],
        "semantic_ast_sha256": hashlib.sha256(canonical_json(result.ast).encode()).hexdigest(),
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7"))
    args = parser.parse_args()
    run(args.training, args.arcgen_root, args.output_dir)


if __name__ == "__main__":
    main()
