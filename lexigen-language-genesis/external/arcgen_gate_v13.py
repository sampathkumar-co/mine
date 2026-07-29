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
VERSION_ROOTS = {name: ROOT / name for name in ("v7", "v8", "v9", "v10", "v11", "v12", "v13")}
sys.path.insert(0, str(VERSION_ROOTS["v13"]))

from latent_runtime_v13 import as_grid, to_json_grid  # noqa: E402
from latent_runtime_v13_ext4 import execute_program  # noqa: E402
from latent_synthesizer_v13 import canonical_json  # noqa: E402
from latent_synthesizer_v13_final import synthesize_latent_final  # noqa: E402
from portable_runtime_v13_ext4 import execute_portable  # noqa: E402

PROTOCOL = "arcgen-gate-v13"
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
    }
    try:
        generated = synthesize_latent_final(examples)
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
        "v13_latent_program_found": bool(generated and generated.program is not None),
        "breakthrough_candidate": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if generated and generated.program is not None:
        program = generated.program
        training_exact = all(execute_program(program, source) == target for source, target in examples)
        portable_training = all(
            execute_portable(program, source) == [list(row) for row in target]
            for source, target in examples
        )
        predictions = [
            to_json_grid(execute_program(program, as_grid(item["input"])))
            for item in package["test"]
        ]
        portable_predictions = [execute_portable(program, item["input"]) for item in package["test"]]
        portable_agreement = predictions == portable_predictions
        artifact = {
            "schema": "lexigen-v13-sealed-latent-artifact-v1",
            "name": program["name"],
            "generated_program": program,
            "program_sha256": hashlib.sha256(canonical_json(program).encode()).hexdigest(),
            "training_exact": training_exact,
            "portable_training_exact": portable_training,
            "portable_prediction_agreement": portable_agreement,
            "human_supplied_finished_task_operator": False,
        }
        artifact_path = args.output_dir / "candidate-latent-program.json"
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
                "v13_candidates_tested": generated.candidates_tested,
                "v13_exact_program_count": generated.exact_candidate_count,
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
