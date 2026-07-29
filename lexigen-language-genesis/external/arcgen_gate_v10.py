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
V7_ROOT, V8_ROOT, V9_ROOT, V10_ROOT = (ROOT / name for name in ("v7", "v8", "v9", "v10"))
sys.path.insert(0, str(V10_ROOT))

from portable_runtime_v10 import as_grid as portable_grid  # noqa: E402
from portable_runtime_v10 import execute_portable  # noqa: E402
from state_machine_runtime_v10 import as_grid, canonical_json, execute_machine, to_json_grid  # noqa: E402
from state_machine_synthesizer_v10 import synthesize_state_machine  # noqa: E402

PROTOCOL = "arcgen-gate-v10-gate9"
base.PROTOCOL = PROTOCOL


def _v7(examples) -> bool:
    sys.path.insert(0, str(V7_ROOT))
    try:
        module = importlib.import_module("semantic_ast_v7")
        converted = [(module.as_grid(source), module.as_grid(target)) for source, target in examples]
        return module.synthesize_ast(converted).ast is not None
    except Exception:
        return False
    finally:
        if str(V7_ROOT) in sys.path:
            sys.path.remove(str(V7_ROOT))


def _v8(examples) -> bool:
    sys.path.insert(0, str(V8_ROOT))
    try:
        module = importlib.import_module("meta_synthesizer_v8")
        return module.synthesize_meta_extension(examples).extension is not None
    except Exception:
        return False
    finally:
        if str(V8_ROOT) in sys.path:
            sys.path.remove(str(V8_ROOT))


def _v9(examples) -> bool:
    sys.path.insert(0, str(V9_ROOT))
    try:
        module = importlib.import_module("object_motion_synthesizer_v9")
        return module.synthesize_object_motion(examples).extension is not None
    except Exception:
        return False
    finally:
        if str(V9_ROOT) in sys.path:
            sys.path.remove(str(V9_ROOT))


def command_solve(args) -> None:
    package = json.loads(args.redacted.read_text(encoding="utf-8"))
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in package["train"]]
    v6 = synthesize_v6(examples, max_depth=3, candidate_budget=75_000)
    v7, v8, v9 = _v7(examples), _v8(examples), _v9(examples)
    try:
        generated = synthesize_state_machine(examples)
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
        "v7_relational_language_found": v7,
        "v8_meta_grammar_found": v8,
        "v9_motion_grammar_found": v9,
        "v10_generated_state_machine_found": bool(generated and generated.machine is not None),
        "breakthrough_candidate": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if generated and generated.machine is not None:
        machine = generated.machine
        training_exact = all(execute_machine(machine, source) == target for source, target in examples)
        portable_training = all(execute_portable(machine, portable_grid(source)) == target for source, target in examples)
        predictions = [to_json_grid(execute_machine(machine, as_grid(item["input"]))) for item in package["test"]]
        portable_predictions = [
            to_json_grid(execute_portable(machine, portable_grid(item["input"]))) for item in package["test"]
        ]
        portable_agreement = predictions == portable_predictions
        artifact = {
            "schema": "lexigen-v10-sealed-state-machine-artifact-v1",
            "name": machine["name"],
            "generated_machine": machine,
            "machine_sha256": hashlib.sha256(canonical_json(machine).encode()).hexdigest(),
            "training_exact": training_exact,
            "portable_training_exact": portable_training,
            "portable_prediction_agreement": portable_agreement,
            "human_supplied_finished_task_operator": False,
        }
        artifact_path = args.output_dir / "candidate-state-machine.json"
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
            training_exact and portable_training and portable_agreement
            and v6.program is None and not v7 and not v8 and not v9
        )
        report.update(
            {
                "v10_candidates_tested": generated.candidates_tested,
                "v10_exact_machine_count": generated.exact_candidate_count,
                "artifact_name": machine["name"],
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
