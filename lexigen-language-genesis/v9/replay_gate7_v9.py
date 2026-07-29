from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from constructive_arcgen_v9 import generate as generate_constructive
from object_motion_runtime_v9 import as_grid, canonical_json, execute_extension, to_json_grid
from object_motion_synthesizer_v9 import synthesize_object_motion
from portable_runtime_v9 import as_grid as portable_grid
from portable_runtime_v9 import execute_portable

ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
TASK_ID = "470c91de"


def load_redacted(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_task_id") != TASK_ID:
        raise AssertionError("unexpected gate-7 task")
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in payload["train"]]
    tests = [as_grid(item["input"]) for item in payload["test"]]
    return payload, examples, tests


def v8_baseline_fits(examples) -> bool:
    v8_root = Path(__file__).resolve().parents[1] / "v8"
    sys.path.insert(0, str(v8_root))
    try:
        module = importlib.import_module("meta_synthesizer_v8")
        result = module.synthesize_meta_extension(examples)
        return result.extension is not None
    except (ValueError, RuntimeError, KeyError, IndexError):
        return False
    finally:
        if str(v8_root) in sys.path:
            sys.path.remove(str(v8_root))


def generate_holdout(arcgen_root: Path, start: int, count: int):
    sys.path.insert(0, str(arcgen_root))
    task = importlib.import_module("tasks.task_470c91de")
    result = []
    for offset, seed in enumerate(range(start, start + count), start=1):
        item = generate_constructive(task, seed)
        result.append((seed, as_grid(item["input"]), as_grid(item["output"])))
        if offset % 1_000 == 0:
            print(json.dumps({"generated_holdout_cases": offset}), flush=True)
    return result


def run(redacted: Path, arcgen_root: Path, output_dir: Path) -> dict[str, Any]:
    payload, examples, tests = load_redacted(redacted)
    result = synthesize_object_motion(examples)
    if result.extension is None:
        raise AssertionError("v9 failed to synthesize an object-motion extension")
    extension = result.extension
    training_exact = all(execute_extension(extension, source) == target for source, target in examples)
    portable_training = all(execute_portable(extension, portable_grid(source)) == target for source, target in examples)
    v8_found = v8_baseline_fits(examples)
    print(
        json.dumps(
            {
                "candidates_tested": result.candidates_tested,
                "exact_candidates": result.exact_candidate_count,
                "extension": extension["name"],
            }
        ),
        flush=True,
    )

    holdout = generate_holdout(arcgen_root, 40_000, 10_000)
    failures = []
    portable_failures = []
    for offset, (seed, source, target) in enumerate(holdout, start=1):
        primary = execute_extension(extension, source)
        portable = execute_portable(extension, portable_grid(source))
        if primary != target:
            failures.append(seed)
        if portable != target or portable != primary:
            portable_failures.append(seed)
        if offset % 1_000 == 0:
            print(
                json.dumps(
                    {
                        "validated_holdout_cases": offset,
                        "primary_failures": len(failures),
                        "portable_failures": len(portable_failures),
                    }
                ),
                flush=True,
            )
    predictions = [to_json_grid(execute_extension(extension, source)) for source in tests]
    extension_sha = hashlib.sha256(canonical_json(extension).encode()).hexdigest()
    task_source = arcgen_root / "tasks" / "task_470c91de.py"
    certificate = {
        "schema": "lexigen-v9-object-motion-certificate-v1",
        "extension_sha256": extension_sha,
        "task_source_sha256": hashlib.sha256(task_source.read_bytes()).hexdigest(),
        "training_exact": training_exact,
        "portable_training_exact": portable_training,
        "holdout_count": len(holdout),
        "holdout_exact": len(holdout) - len(failures),
        "portable_holdout_exact": len(holdout) - len(portable_failures),
        "v8_fixed_meta_grammar_found": v8_found,
        "extension_ablation_training_exact": all(source == target for source, target in examples),
    }
    report = {
        "version": "v9",
        "benchmark": "ARC-GEN 470c91de post-failure object-motion grammar growth",
        "status": "object-marker-vector grammar mechanism candidate; not a blind breakthrough claim",
        "arcgen_commit": ARCGEN_COMMIT,
        "source_task_id": TASK_ID,
        "source_sealed_outputs_accessed": False,
        "candidate_extensions_tested": result.candidates_tested,
        "exact_extension_count": result.exact_candidate_count,
        "generated_extension": extension,
        "certificate": certificate,
        "holdout_seed_range": [40_000, 49_999],
        "holdout_failures": failures,
        "portable_holdout_failures": portable_failures,
        "test_prediction_count": len(predictions),
        "claim_boundary": (
            "The object-motion extension is synthesized from generic object, marker, point-set, vector and rendering choices. "
            "The grammar was designed post-failure; a fresh sealed task is required before any external breakthrough claim."
        ),
    }
    gate = {
        "new_production_absent_from_v8": not v8_found,
        "no_finished_task_operator": not extension["provenance"]["human_supplied_finished_task_operator"],
        "training_exact": training_exact,
        "portable_training_exact": portable_training,
        "holdout_exact": not failures,
        "portable_holdout_exact": not portable_failures,
        "ablation_fails": not certificate["extension_ablation_training_exact"],
        "sealed_outputs_untouched": True,
    }
    report["gate"] = gate
    if not all(gate.values()):
        raise AssertionError(f"v9 gate failed: {gate}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v9-object-motion-extension.json"
    report_path = output_dir / "v9-report.json"
    predictions_path = output_dir / "gate7-postfailure-predictions.json"
    artifact_path.write_text(json.dumps(extension, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "extension": extension["name"],
        "candidates_tested": result.candidates_tested,
        "exact_extension_count": result.exact_candidate_count,
        "holdout_accuracy": certificate["holdout_exact"] / certificate["holdout_count"],
        "v8_baseline_found": v8_found,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redacted", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v9"))
    args = parser.parse_args()
    run(args.redacted, args.arcgen_root, args.output_dir)


if __name__ == "__main__":
    main()
