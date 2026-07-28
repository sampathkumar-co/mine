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
V7_ROOT, V8_ROOT, V9_ROOT, V10_ROOT, V11_ROOT = (
    ROOT / name for name in ("v7", "v8", "v9", "v10", "v11")
)
sys.path.insert(0, str(V11_ROOT))

from compositional_runtime_v11 import as_grid, canonical_json, execute_pipeline, to_json_grid  # noqa: E402
from compositional_synthesizer_v11 import synthesize_pipeline  # noqa: E402
from portable_runtime_v11 import as_grid as portable_grid  # noqa: E402
from portable_runtime_v11 import execute_portable  # noqa: E402

PROTOCOL = "arcgen-gate-v11-gate10"
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
        "v7": _baseline(V7_ROOT, "semantic_ast_v7", "synthesize_ast", "ast", examples),
        "v8": _baseline(V8_ROOT, "meta_synthesizer_v8", "synthesize_meta_extension", "extension", examples),
        "v9": _baseline(V9_ROOT, "object_motion_synthesizer_v9", "synthesize_object_motion", "extension", examples),
        "v10": _baseline(V10_ROOT, "state_machine_synthesizer_v10", "synthesize_state_machine", "machine", examples),
    }
    try:
        generated = synthesize_pipeline(examples)
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
        "v11_generated_pipeline_found": bool(generated and generated.program is not None),
        "breakthrough_candidate": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if generated and generated.program is not None:
        program = generated.program
        training_exact = all(execute_pipeline(program, source) == target for source, target in examples)
        portable_training = all(
            execute_portable(program, portable_grid(source)) == target for source, target in examples
        )
        predictions = [
            to_json_grid(execute_pipeline(program, as_grid(item["input"])))
            for item in package["test"]
        ]
        portable_predictions = [
            to_json_grid(execute_portable(program, portable_grid(item["input"])))
            for item in package["test"]
        ]
        portable_agreement = predictions == portable_predictions
        artifact = {
            "schema": "lexigen-v11-sealed-pipeline-artifact-v1",
            "name": program["name"],
            "generated_pipeline": program,
            "program_sha256": hashlib.sha256(canonical_json(program).encode()).hexdigest(),
            "training_exact": training_exact,
            "portable_training_exact": portable_training,
            "portable_prediction_agreement": portable_agreement,
            "human_supplied_finished_task_operator": False,
        }
        artifact_path = args.output_dir / "candidate-compositional-pipeline.json"
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
            )
            + "\n",
            encoding="utf-8",
        )
        breakthrough = bool(
            training_exact
            and portable_training
            and portable_agreement
            and v6.program is None
            and not any(baselines.values())
        )
        report.update(
            {
                "v11_candidates_tested": generated.candidates_tested,
                "v11_exact_pipeline_count": generated.exact_candidate_count,
                "artifact_name": program["name"],
                "artifact_sha256": base.sha256_file(artifact_path),
                "predictions_sha256": base.sha256_file(prediction_path),
                "training_exact": training_exact,
                "portable_training_exact": portable_training,
                "portable_prediction_agreement": portable_agreement,
                "breakthrough_candidate": breakthrough,
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
