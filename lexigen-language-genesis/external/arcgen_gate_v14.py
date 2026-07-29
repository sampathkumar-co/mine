from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import arcgen_gate as base
from arc_language_v6 import synthesize as synthesize_v6

ROOT = Path(__file__).resolve().parents[1]
VERSION_ROOTS = {
    name: ROOT / name
    for name in ("v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14")
}
sys.path.insert(0, str(VERSION_ROOTS["v14"]))

from portable_scene_runtime_v14 import execute_portable_pipeline  # noqa: E402
from scene_runtime_v14 import as_grid, execute_pipeline  # noqa: E402
from scene_synthesizer_v14 import canonical, synthesize_scene  # noqa: E402

PROTOCOL = "arcgen-gate-v14"
base.PROTOCOL = PROTOCOL


def _baseline(folder: Path, module_name: str, function_name: str, field: str, examples) -> bool:
    sys.path.insert(0, str(folder))
    try:
        module = importlib.import_module(module_name)
        result = getattr(module, function_name)(examples)
        return getattr(result, field) is not None
    except Exception:
        return False
    finally:
        if str(folder) in sys.path:
            sys.path.remove(str(folder))
        sys.modules.pop(module_name, None)


def command_solve(args) -> None:
    package = json.loads(args.redacted.read_text(encoding="utf-8"))
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in package["train"]]
    v6 = synthesize_v6(examples, max_depth=3, candidate_budget=75_000)
    baselines = {
        "v7": _baseline(VERSION_ROOTS["v7"], "semantic_ast_v7", "synthesize_ast", "ast", examples),
        "v8": _baseline(VERSION_ROOTS["v8"], "meta_synthesizer_v8", "synthesize_meta_extension", "extension", examples),
        "v9": _baseline(VERSION_ROOTS["v9"], "object_motion_synthesizer_v9", "synthesize_object_motion", "extension", examples),
        "v10": _baseline(VERSION_ROOTS["v10"], "state_machine_synthesizer_v10", "synthesize_state_machine", "machine", examples),
        "v11": _baseline(VERSION_ROOTS["v11"], "compositional_synthesizer_v11", "synthesize_pipeline", "program", examples),
        "v12": _baseline(VERSION_ROOTS["v12"], "hierarchical_synthesizer_v12", "synthesize_hierarchical", "program", examples),
        "v13": _baseline(VERSION_ROOTS["v13"], "latent_synthesizer_v13_final", "synthesize_latent_final", "program", examples),
    }
    try:
        generated = synthesize_scene(examples, max_depth=2, candidate_budget=200_000)
    except Exception:
        generated = None

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "selected_task_id": package["selected_task_id"],
        "redacted_task_sha256": base.sha256_file(args.redacted),
        "v6_candidate_budget": 75_000,
        "v6_candidates_tested": v6.candidates_tested,
        "v6_signatures_seen": v6.signatures_seen,
        "v6_fixed_language_found": v6.program is not None,
        "v7_relational_language_found": baselines["v7"],
        "v8_meta_grammar_found": baselines["v8"],
        "v9_motion_grammar_found": baselines["v9"],
        "v10_state_machine_found": baselines["v10"],
        "v11_generated_pipeline_found": baselines["v11"],
        "v12_hierarchical_program_found": baselines["v12"],
        "v13_latent_program_found": baselines["v13"],
        "v14_scene_pipeline_found": bool(generated and generated.pipeline is not None),
        "v14_only_candidate": False,
        "breakthrough_candidate": False,
        "world_level_claim_eligible": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if generated and generated.pipeline is not None:
        pipeline = generated.pipeline
        training_exact = all(execute_pipeline(pipeline, source) == target for source, target in examples)
        portable_training_exact = all(
            execute_portable_pipeline(pipeline, source) == target
            for source, target in examples
        )
        predictions = [
            [list(row) for row in execute_pipeline(pipeline, as_grid(item["input"]))]
            for item in package["test"]
        ]
        portable_predictions = [
            [list(row) for row in execute_portable_pipeline(pipeline, item["input"])]
            for item in package["test"]
        ]
        portable_agreement = predictions == portable_predictions
        pipeline_text = canonical(list(pipeline))
        pipeline_sha256 = hashlib.sha256(pipeline_text.encode()).hexdigest()
        artifact = {
            "schema": "lexigen-v14-sealed-scene-pipeline-v1",
            "name": "generated_scene_pipeline_" + pipeline_sha256[:12],
            "generated_pipeline": pipeline,
            "pipeline_sha256": pipeline_sha256,
            "training_exact": training_exact,
            "portable_training_exact": portable_training_exact,
            "portable_prediction_agreement": portable_agreement,
            "human_supplied_finished_task_operator": False,
            "human_supplied_generic_scene_algebra": True,
            "claim_boundary": (
                "A v14-only blind win is external capability evidence, not autonomous "
                "language-genesis evidence, because the generic scene atoms were human-authored."
            ),
        }
        artifact_path = args.output_dir / "candidate-scene-pipeline.json"
        prediction_path = args.output_dir / "candidate-predictions.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prediction_path.write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "selected_task_id": package["selected_task_id"],
                    "artifact_sha256": base.sha256_file(artifact_path),
                    "predictions": predictions,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        older_found = v6.program is not None or any(baselines.values())
        v14_only = bool(
            training_exact
            and portable_training_exact
            and portable_agreement
            and not older_found
        )
        report.update(
            {
                "v14_candidates_tested": generated.candidates_tested,
                "v14_signatures_seen": generated.signatures_seen,
                "v14_inventory_size": generated.inventory_size,
                "v14_exact_pipeline_count": generated.exact_pipeline_count,
                "artifact_name": artifact["name"],
                "artifact_sha256": base.sha256_file(artifact_path),
                "predictions_sha256": base.sha256_file(prediction_path),
                "training_exact": training_exact,
                "portable_training_exact": portable_training_exact,
                "portable_prediction_agreement": portable_agreement,
                "v14_only_candidate": v14_only,
            }
        )
    report_path = args.output_dir / "candidate-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = base.build_parser()
    args = parser.parse_args()
    if args.command == "solve":
        command_solve(args)
    else:
        args.function(args)


if __name__ == "__main__":
    main()
