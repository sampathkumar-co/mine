from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from compositional_runtime_v11 import as_grid, canonical_json, execute_pipeline, to_json_grid
from compositional_synthesizer_v11 import synthesize_pipeline
from constructive_arcgen_v11 import generate as generate_constructive
from portable_runtime_v11 import as_grid as portable_grid
from portable_runtime_v11 import execute_portable

ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
TASK_ID = "33067df9"


def load_redacted(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected_task_id") != TASK_ID:
        raise AssertionError("unexpected gate-9 task")
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in payload["train"]]
    tests = [as_grid(item["input"]) for item in payload["test"]]
    return payload, examples, tests


def previous_baselines(examples) -> dict[str, bool]:
    root = Path(__file__).resolve().parents[1]
    checks = [
        ("v7", "semantic_ast_v7", "synthesize_ast", "ast"),
        ("v8", "meta_synthesizer_v8", "synthesize_meta_extension", "extension"),
        ("v9", "object_motion_synthesizer_v9", "synthesize_object_motion", "extension"),
        ("v10", "state_machine_synthesizer_v10", "synthesize_state_machine", "machine"),
    ]
    results = {}
    for folder, module_name, function_name, result_field in checks:
        folder_path = root / folder
        sys.path.insert(0, str(folder_path))
        try:
            module = importlib.import_module(module_name)
            result = getattr(module, function_name)(examples)
            results[folder] = getattr(result, result_field) is not None
        except Exception:
            results[folder] = False
        finally:
            if str(folder_path) in sys.path:
                sys.path.remove(str(folder_path))
            sys.modules.pop(module_name, None)
    return results


def generate_cases(arcgen_root: Path, start: int, count: int, label: str):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    task = importlib.import_module("tasks.task_33067df9")
    cases = []
    for offset, seed in enumerate(range(start, start + count), start=1):
        item = generate_constructive(task, seed)
        cases.append((seed, as_grid(item["input"]), as_grid(item["output"])))
        if offset % 1_000 == 0:
            print(json.dumps({f"generated_{label}_cases": offset}), flush=True)
    return cases


def stage_summary(program: dict[str, Any]) -> dict[str, Any]:
    stages = program["stages"]
    return {
        "extract": stages[0]["mode"],
        "background": stages[0]["background"],
        "output_shape": [stages[1]["output_height"], stages[1]["output_width"]],
        "margin": stages[1]["margin"],
        "gap": stages[1]["gap"],
        "relation": stages[2]["predicate"],
        "precedence": stages[3]["mode"],
        "canvas_background": stages[4]["canvas_background"],
        "skip_background_tiles": stages[4]["skip_background_tiles"],
    }


def run(redacted: Path, arcgen_root: Path, output_dir: Path) -> dict[str, Any]:
    payload, examples, tests = load_redacted(redacted)
    initial_count = len(examples)
    discovery = generate_cases(arcgen_root, 60_000, 10_000, "discovery")
    rounds: list[dict[str, Any]] = []
    used_counterexamples: set[int] = set()

    for round_index in range(8):
        result = synthesize_pipeline(examples)
        if result.program is None:
            raise AssertionError("v11 pipeline grammar became inconsistent with accumulated evidence")
        program = result.program
        failure = next(
            (
                (seed, source, target)
                for seed, source, target in discovery
                if seed not in used_counterexamples and execute_pipeline(program, source) != target
            ),
            None,
        )
        rounds.append(
            {
                "round": round_index,
                "program_sha256": hashlib.sha256(canonical_json(program).encode()).hexdigest(),
                "stages": stage_summary(program),
                "exact_candidate_count": result.exact_candidate_count,
                "counterexample_seed": None if failure is None else failure[0],
            }
        )
        if failure is None:
            break
        used_counterexamples.add(failure[0])
        examples.append((failure[1], failure[2]))
    else:
        raise AssertionError("v11 exceeded frozen counterexample round budget")

    result = synthesize_pipeline(examples)
    if result.program is None:
        raise AssertionError("v11 final compositional pipeline is missing")
    program = result.program
    training_exact = all(execute_pipeline(program, source) == target for source, target in examples)
    portable_training = all(execute_portable(program, portable_grid(source)) == target for source, target in examples)
    baselines = previous_baselines(examples)
    print(
        json.dumps(
            {
                "cegis_rounds": len(rounds),
                "counterexamples": sorted(used_counterexamples),
                "candidates_tested_final": result.candidates_tested,
                "exact_candidates_final": result.exact_candidate_count,
                "pipeline": program["name"],
                "prior_baselines": baselines,
                "stages": stage_summary(program),
            }
        ),
        flush=True,
    )

    holdout = generate_cases(arcgen_root, 70_000, 10_000, "holdout")
    failures = []
    portable_failures = []
    for offset, (seed, source, target) in enumerate(holdout, start=1):
        primary = execute_pipeline(program, source)
        portable = execute_portable(program, portable_grid(source))
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
    predictions = [to_json_grid(execute_pipeline(program, source)) for source in tests]
    program_sha = hashlib.sha256(canonical_json(program).encode()).hexdigest()
    task_source = arcgen_root / "tasks" / "task_33067df9.py"
    certificate = {
        "schema": "lexigen-v11-compositional-certificate-v1",
        "program_sha256": program_sha,
        "task_source_sha256": hashlib.sha256(task_source.read_bytes()).hexdigest(),
        "training_exact": training_exact,
        "portable_training_exact": portable_training,
        "holdout_count": len(holdout),
        "holdout_exact": len(holdout) - len(failures),
        "portable_holdout_exact": len(holdout) - len(portable_failures),
        "prior_baselines": baselines,
        "pipeline_ablation_training_exact": all(source == target for source, target in examples),
    }
    report = {
        "version": "v11",
        "benchmark": "ARC-GEN 33067df9 post-failure unified compositional pipeline",
        "status": "counterexample-guided unified pipeline mechanism candidate; not a blind breakthrough claim",
        "arcgen_commit": ARCGEN_COMMIT,
        "source_task_id": TASK_ID,
        "source_sealed_outputs_accessed": False,
        "initial_demonstration_count": initial_count,
        "counterexample_count": len(used_counterexamples),
        "counterexample_seeds": sorted(used_counterexamples),
        "rounds": rounds,
        "candidate_pipelines_tested_final": result.candidates_tested,
        "exact_pipeline_count_final": result.exact_candidate_count,
        "generated_pipeline": program,
        "certificate": certificate,
        "discovery_seed_range": [60_000, 69_999],
        "holdout_seed_range": [70_000, 79_999],
        "holdout_failures": failures,
        "portable_holdout_failures": portable_failures,
        "test_prediction_count": len(predictions),
        "claim_boundary": (
            "v11 composes generic stages and uses public post-failure counterexamples to resolve underdetermined precedence. "
            "The stage inventory remains human supplied; a fresh sealed gate and cross-family transfer are required."
        ),
    }
    gate = {
        "new_pipeline_absent_from_prior_versions": not any(baselines.values()),
        "no_finished_task_operator": not program["provenance"]["human_supplied_finished_task_operator"],
        "training_exact": training_exact,
        "portable_training_exact": portable_training,
        "holdout_exact": not failures,
        "portable_holdout_exact": not portable_failures,
        "ablation_fails": not certificate["pipeline_ablation_training_exact"],
        "sealed_outputs_untouched": True,
    }
    report["gate"] = gate
    if not all(gate.values()):
        raise AssertionError(f"v11 gate failed: {gate}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_dir / "v11-compositional-pipeline.json"
    report_path = output_dir / "v11-report.json"
    predictions_path = output_dir / "gate9-postfailure-predictions.json"
    pipeline_path.write_text(json.dumps(program, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps({"predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    summary = {
        "pipeline": program["name"],
        "counterexamples": sorted(used_counterexamples),
        "cegis_rounds": len(rounds),
        "candidates_tested_final": result.candidates_tested,
        "exact_pipeline_count_final": result.exact_candidate_count,
        "holdout_accuracy": certificate["holdout_exact"] / certificate["holdout_count"],
        "prior_baselines": baselines,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redacted", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v11"))
    args = parser.parse_args()
    run(args.redacted, args.arcgen_root, args.output_dir)


if __name__ == "__main__":
    main()
