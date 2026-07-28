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
V7_ROOT = ROOT / "v7"
V8_ROOT = ROOT / "v8"
V9_ROOT = ROOT / "v9"
sys.path.insert(0, str(V9_ROOT))

from object_motion_runtime_v9 import as_grid, canonical_json, execute_extension, to_json_grid  # noqa: E402
from object_motion_synthesizer_v9 import synthesize_object_motion  # noqa: E402
from portable_runtime_v9 import as_grid as portable_grid  # noqa: E402
from portable_runtime_v9 import execute_portable  # noqa: E402

PROTOCOL = "arcgen-gate-v9-gate8"
base.PROTOCOL = PROTOCOL


def v7_found(examples) -> bool:
    sys.path.insert(0, str(V7_ROOT))
    try:
        module = importlib.import_module("semantic_ast_v7")
        converted = [(module.as_grid(source), module.as_grid(target)) for source, target in examples]
        return module.synthesize_ast(converted).ast is not None
    except (ValueError, RuntimeError, KeyError, IndexError, AssertionError):
        return False
    finally:
        if str(V7_ROOT) in sys.path:
            sys.path.remove(str(V7_ROOT))


def v8_found(examples) -> bool:
    sys.path.insert(0, str(V8_ROOT))
    try:
        module = importlib.import_module("meta_synthesizer_v8")
        return module.synthesize_meta_extension(examples).extension is not None
    except (ValueError, RuntimeError, KeyError, IndexError, AssertionError):
        return False
    finally:
        if str(V8_ROOT) in sys.path:
            sys.path.remove(str(V8_ROOT))


def command_solve(args) -> None:
    package = json.loads(args.redacted.read_text(encoding="utf-8"))
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in package["train"]]
    v6 = synthesize_v6(examples, max_depth=3, candidate_budget=75_000)
    fixed_v7 = v7_found(examples)
    fixed_v8 = v8_found(examples)
    try:
        generated = synthesize_object_motion(examples)
    except (ValueError, RuntimeError, KeyError, IndexError, AssertionError):
        generated = None

    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "selected_task_id": package["selected_task_id"],
        "redacted_task_sha256": base.sha256_file(args.redacted),
        "v6_candidate_budget": 75_000,
        "v6_candidates_tested": v6.candidates_tested,
        "v6_signatures_seen": v6.signatures_seen,
        "v6_fixed_language_found": v6.program is not None,
        "v7_relational_language_found": fixed_v7,
        "v8_meta_grammar_found": fixed_v8,
        "v9_generated_motion_found": bool(generated and generated.extension is not None),
        "breakthrough_candidate": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if generated and generated.extension is not None:
        extension = generated.extension
        training_exact = all(execute_extension(extension, source) == target for source, target in examples)
        portable_training = all(
            execute_portable(extension, portable_grid(source)) == target for source, target in examples
        )
        predictions = [
            to_json_grid(execute_extension(extension, as_grid(item["input"])))
            for item in package["test"]
        ]
        portable_predictions = [
            to_json_grid(execute_portable(extension, portable_grid(item["input"])))
            for item in package["test"]
        ]
        portable_agreement = predictions == portable_predictions
        artifact = {
            "schema": "lexigen-v9-sealed-candidate-artifact-v1",
            "name": extension["name"],
            "generated_extension": extension,
            "extension_sha256": hashlib.sha256(canonical_json(extension).encode()).hexdigest(),
            "training_exact": training_exact,
            "portable_training_exact": portable_training,
            "portable_prediction_agreement": portable_agreement,
            "human_supplied_finished_task_operator": False,
        }
        artifact_path = args.output_dir / "candidate-motion-extension.json"
        predictions_path = args.output_dir / "candidate-predictions.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        predictions_path.write_text(
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
            and not fixed_v7
            and not fixed_v8
        )
        report.update(
            {
                "v9_candidates_tested": generated.candidates_tested,
                "v9_exact_extension_count": generated.exact_candidate_count,
                "artifact_name": extension["name"],
                "artifact_sha256": base.sha256_file(artifact_path),
                "predictions_sha256": base.sha256_file(predictions_path),
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
