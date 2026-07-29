from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import arcgen_gate as base
from arc_language_v6 import synthesize as synthesize_v6

V8_ROOT = Path(__file__).resolve().parents[1] / "v8"
V7_ROOT = Path(__file__).resolve().parents[1] / "v7"
sys.path.insert(0, str(V8_ROOT))

from meta_runtime_v8 import as_grid, canonical_json, execute_extension, to_json_grid  # noqa: E402
from meta_synthesizer_v8 import synthesize_meta_extension  # noqa: E402
from portable_runtime_v8 import as_grid as portable_grid  # noqa: E402
from portable_runtime_v8 import execute_portable  # noqa: E402

PROTOCOL = "arcgen-gate-v8-gate7"
base.PROTOCOL = PROTOCOL


def v7_relational_baseline(examples) -> bool:
    sys.path.insert(0, str(V7_ROOT))
    try:
        module = importlib.import_module("semantic_ast_v7")
        converted = [(module.as_grid(source), module.as_grid(target)) for source, target in examples]
        result = module.synthesize_ast(converted)
        return result.ast is not None
    except (ValueError, AssertionError, RuntimeError, KeyError):
        return False
    finally:
        if str(V7_ROOT) in sys.path:
            sys.path.remove(str(V7_ROOT))


def command_solve(args) -> None:
    package = json.loads(args.redacted.read_text(encoding="utf-8"))
    examples = [(as_grid(pair["input"]), as_grid(pair["output"])) for pair in package["train"]]

    v6_result = synthesize_v6(examples, max_depth=3, candidate_budget=75_000)
    v7_found = v7_relational_baseline(examples)
    try:
        meta_result = synthesize_meta_extension(examples)
    except (ValueError, RuntimeError, KeyError, IndexError):
        meta_result = None

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "selected_task_id": package["selected_task_id"],
        "redacted_task_sha256": base.sha256_file(args.redacted),
        "v6_candidate_budget": 75_000,
        "v6_candidates_tested": v6_result.candidates_tested,
        "v6_signatures_seen": v6_result.signatures_seen,
        "v6_fixed_language_found": v6_result.program is not None,
        "v7_relational_language_found": v7_found,
        "v8_generated_extension_found": bool(meta_result and meta_result.extension is not None),
        "breakthrough_candidate": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if meta_result and meta_result.extension is not None:
        extension = meta_result.extension
        primary_training = all(execute_extension(extension, source) == target for source, target in examples)
        portable_training = all(
            execute_portable(extension, portable_grid(source)) == target for source, target in examples
        )
        predictions = [
            to_json_grid(execute_extension(extension, as_grid(pair["input"])))
            for pair in package["test"]
        ]
        portable_predictions = [
            to_json_grid(execute_portable(extension, portable_grid(pair["input"])))
            for pair in package["test"]
        ]
        portable_prediction_agreement = predictions == portable_predictions
        artifact = {
            "schema": "lexigen-v8-sealed-candidate-artifact-v1",
            "name": extension["name"],
            "generated_extension": extension,
            "training_demonstration_count": len(examples),
            "training_exact": primary_training,
            "portable_training_exact": portable_training,
            "portable_prediction_agreement": portable_prediction_agreement,
            "extension_sha256": hashlib.sha256(canonical_json(extension).encode()).hexdigest(),
            "human_supplied_finished_task_operator": False,
        }
        artifact_path = args.output_dir / "candidate-meta-extension.json"
        prediction_path = args.output_dir / "candidate-predictions.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prediction_payload = {
            "protocol": PROTOCOL,
            "selected_task_id": package["selected_task_id"],
            "artifact_sha256": base.sha256_file(artifact_path),
            "predictions": predictions,
        }
        prediction_path.write_text(
            json.dumps(prediction_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        breakthrough_candidate = bool(
            primary_training
            and portable_training
            and portable_prediction_agreement
            and v6_result.program is None
            and not v7_found
        )
        report.update(
            {
                "v8_candidate_extensions_tested": meta_result.candidates_tested,
                "v8_exact_extension_count": meta_result.exact_candidate_count,
                "artifact_name": extension["name"],
                "artifact_sha256": base.sha256_file(artifact_path),
                "predictions_sha256": base.sha256_file(prediction_path),
                "training_exact": primary_training,
                "portable_training_exact": portable_training,
                "portable_prediction_agreement": portable_prediction_agreement,
                "breakthrough_candidate": breakthrough_candidate,
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
