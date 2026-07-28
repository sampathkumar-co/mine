from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from meta_runtime_v8 import as_grid, canonical_json, execute_extension, to_json_grid
from meta_synthesizer_v8 import synthesize_meta_extension
from portable_runtime_v8 import as_grid as portable_grid
from portable_runtime_v8 import execute_portable

ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
TASK_ID = "1d61978c"


def load_redacted(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_task_id") != TASK_ID:
        raise AssertionError("unexpected gate-6 task identity")
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in payload["train"]]
    tests = [as_grid(item["input"]) for item in payload["test"]]
    return payload, examples, tests


def v7_baseline_fits(examples) -> bool:
    v7_root = Path(__file__).resolve().parents[1] / "v7"
    sys.path.insert(0, str(v7_root))
    try:
        module = importlib.import_module("semantic_ast_v7")
        converted = [(module.as_grid(source), module.as_grid(target)) for source, target in examples]
        result = module.synthesize_ast(converted)
        return result.ast is not None
    except (ValueError, AssertionError, RuntimeError):
        return False
    finally:
        if str(v7_root) in sys.path:
            sys.path.remove(str(v7_root))


def generated_cases(arcgen_root: Path, start: int, count: int):
    sys.path.insert(0, str(arcgen_root))
    task = importlib.import_module("tasks.task_1d61978c")
    cases = []
    for seed in range(start, start + count):
        random.seed(seed)
        item = task.generate()
        cases.append((seed, as_grid(item["input"]), as_grid(item["output"])))
    return cases


def run(redacted: Path, arcgen_root: Path, output_dir: Path) -> dict[str, Any]:
    payload, examples, tests = load_redacted(redacted)
    result = synthesize_meta_extension(examples)
    if result.extension is None:
        raise AssertionError("v8 failed to synthesize a meta-grammar extension")
    extension = result.extension
    exact_training = all(execute_extension(extension, source) == target for source, target in examples)
    portable_training = all(
        execute_portable(extension, portable_grid(source)) == target for source, target in examples
    )
    v7_found = v7_baseline_fits(examples)

    holdout = generated_cases(arcgen_root, 30_000, 10_000)
    failures = []
    portable_failures = []
    for seed, source, target in holdout:
        primary = execute_extension(extension, source)
        portable = execute_portable(extension, portable_grid(source))
        if primary != target:
            failures.append(seed)
        if portable != target or portable != primary:
            portable_failures.append(seed)
    predictions = [to_json_grid(execute_extension(extension, source)) for source in tests]

    extension_sha = hashlib.sha256(canonical_json(extension).encode()).hexdigest()
    training_sha = hashlib.sha256(
        canonical_json(payload["train"]).encode()
    ).hexdigest()
    task_source = arcgen_root / "tasks" / "task_1d61978c.py"
    certificate = {
        "schema": "lexigen-v8-meta-extension-certificate-v1",
        "extension_sha256": extension_sha,
        "training_sha256": training_sha,
        "task_source_sha256": hashlib.sha256(task_source.read_bytes()).hexdigest(),
        "training_exact": exact_training,
        "portable_training_exact": portable_training,
        "holdout_count": len(holdout),
        "holdout_exact": len(holdout) - len(failures),
        "portable_holdout_exact": len(holdout) - len(portable_failures),
        "v7_fixed_grammar_found": v7_found,
        "simple_fixed_baseline_found": result.fixed_grammar_baseline_found,
        "extension_ablation_training_exact": all(source == target for source, target in examples),
    }
    report = {
        "version": "v8",
        "benchmark": "ARC-GEN 1d61978c post-failure meta-grammar growth",
        "status": "autonomous meta-grammar mechanism candidate; not a blind breakthrough claim",
        "arcgen_commit": ARCGEN_COMMIT,
        "source_task_id": TASK_ID,
        "source_sealed_outputs_accessed": False,
        "candidate_extensions_tested": result.candidates_tested,
        "exact_extension_count": result.exact_candidate_count,
        "generated_extension": extension,
        "certificate": certificate,
        "holdout_seed_range": [30_000, 39_999],
        "holdout_failures": failures,
        "portable_holdout_failures": portable_failures,
        "test_prediction_count": len(predictions),
        "claim_boundary": (
            "The extension is synthesized as an arithmetic/graph/fold AST and was absent from v7. "
            "The generic substrate and post-failure task family remain human selected; a fresh sealed gate is required."
        ),
    }
    gate = {
        "new_production_absent_from_v7": not v7_found,
        "no_finished_task_operator_in_substrate": not extension["provenance"]["human_supplied_finished_task_operator"],
        "exact_training": exact_training,
        "portable_training": portable_training,
        "holdout_exact": not failures,
        "portable_holdout_exact": not portable_failures,
        "ablation_fails": not certificate["extension_ablation_training_exact"],
        "sealed_outputs_untouched": True,
    }
    report["gate"] = gate
    if not all(gate.values()):
        raise AssertionError(f"v8 gate failed: {gate}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v8-meta-extension.json"
    report_path = output_dir / "v8-report.json"
    prediction_path = output_dir / "gate6-postfailure-predictions.json"
    artifact_path.write_text(json.dumps(extension, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prediction_path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "extension": extension["name"],
        "candidate_extensions_tested": result.candidates_tested,
        "exact_extension_count": result.exact_candidate_count,
        "holdout_accuracy": certificate["holdout_exact"] / certificate["holdout_count"],
        "v7_baseline_found": v7_found,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redacted", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v8"))
    args = parser.parse_args()
    run(args.redacted, args.arcgen_root, args.output_dir)


if __name__ == "__main__":
    main()
