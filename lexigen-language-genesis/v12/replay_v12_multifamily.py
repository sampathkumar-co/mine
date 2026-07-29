from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from hierarchical_runtime_v12 import as_grid, canonical_json, execute_program, to_json_grid
from hierarchical_synthesizer_v12 import synthesize_hierarchical
from portable_runtime_v12 import execute_portable

ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
GATES = ("v11-campaign-01", "v11-campaign-05", "v11-campaign-07")
HOLDOUT_START = 180_000
HOLDOUT_COUNT = 10_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gate(campaign_root: Path, gate: str):
    folder = campaign_root / gate
    payload = json.loads((folder / "redacted-task.json").read_text(encoding="utf-8"))
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in payload["train"]]
    tests = [as_grid(item["input"]) for item in payload["test"]]
    prior = json.loads((folder / "candidate-report.json").read_text(encoding="utf-8"))
    return folder, payload, examples, tests, prior


def run(campaign_root: Path, arcgen_root: Path, output_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(arcgen_root))
    import task_list  # type: ignore

    registry = task_list.task_list()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_programs = {}
    all_predictions = {}
    for gate in GATES:
        folder, payload, examples, tests, prior = load_gate(campaign_root, gate)
        result = synthesize_hierarchical(examples)
        if result.program is None:
            raise AssertionError(f"v12 found no program for {gate}")
        program = result.program
        training_exact = all(execute_program(program, source) == target for source, target in examples)
        portable_training = all(
            execute_portable(program, source) == [list(row) for row in target]
            for source, target in examples
        )
        no_op_exact = all(source == target for source, target in examples)
        generator, _ = registry[payload["selected_task_id"]]
        primary_failures: list[int] = []
        portable_failures: list[int] = []
        for offset in range(HOLDOUT_COUNT):
            seed = HOLDOUT_START + offset
            random.seed(seed)
            item = generator()
            source = as_grid(item["input"])
            target = as_grid(item["output"])
            primary = execute_program(program, source)
            portable = execute_portable(program, source)
            if primary != target:
                primary_failures.append(seed)
            if portable != [list(row) for row in primary] or primary != target:
                portable_failures.append(seed)
            if (offset + 1) % 1_000 == 0:
                print(json.dumps({
                    "gate": gate,
                    "validated": offset + 1,
                    "primary_failures": len(primary_failures),
                    "portable_failures": len(portable_failures),
                }), flush=True)
        prior_false = not any(bool(prior.get(key)) for key in (
            "v6_fixed_language_found",
            "v7_relational_language_found",
            "v8_meta_grammar_found",
            "v9_motion_grammar_found",
            "v10_state_machine_found",
            "v11_generated_pipeline_found",
        ))
        row = {
            "gate_id": gate,
            "task_id": payload["selected_task_id"],
            "redacted_sha256": sha256_file(folder / "redacted-task.json"),
            "candidates_tested": result.candidates_tested,
            "exact_candidate_count": result.exact_candidate_count,
            "program_name": program["name"],
            "program_sha256": hashlib.sha256(canonical_json(program).encode()).hexdigest(),
            "training_exact": training_exact,
            "portable_training_exact": portable_training,
            "no_op_ablation_exact": no_op_exact,
            "prior_v6_v11_all_failed": prior_false,
            "holdout_count": HOLDOUT_COUNT,
            "primary_holdout_exact": HOLDOUT_COUNT - len(primary_failures),
            "portable_holdout_exact": HOLDOUT_COUNT - len(portable_failures),
            "primary_failures": primary_failures,
            "portable_failures": portable_failures,
        }
        rows.append(row)
        all_programs[gate] = program
        all_predictions[gate] = [to_json_grid(execute_program(program, source)) for source in tests]

    gate_checks = {
        "three_unrelated_task_families": len(rows) == 3,
        "all_prior_versions_failed": all(row["prior_v6_v11_all_failed"] for row in rows),
        "all_training_exact": all(row["training_exact"] for row in rows),
        "all_portable_training_exact": all(row["portable_training_exact"] for row in rows),
        "all_holdouts_exact": all(row["primary_holdout_exact"] == HOLDOUT_COUNT for row in rows),
        "all_portable_holdouts_exact": all(row["portable_holdout_exact"] == HOLDOUT_COUNT for row in rows),
        "all_ablations_fail": all(not row["no_op_ablation_exact"] for row in rows),
        "sealed_outputs_untouched": True,
    }
    report = {
        "version": "v12",
        "status": "post-failure hierarchical multi-family mechanism candidate; not a blind breakthrough",
        "arcgen_commit": ARCGEN_COMMIT,
        "source_gates": list(GATES),
        "source_sealed_outputs_accessed": False,
        "holdout_seed_range": [HOLDOUT_START, HOLDOUT_START + HOLDOUT_COUNT - 1],
        "rows": rows,
        "gate": gate_checks,
        "claim_boundary": (
            "v12 uses one generic hierarchical scene grammar across three already-preserved v11 failures. "
            "The substrate was designed after those failures and therefore requires a fresh sealed campaign."
        ),
    }
    if not all(gate_checks.values()):
        raise AssertionError(f"v12 multi-family gate failed: {gate_checks}")

    programs_path = output_dir / "v12-programs.json"
    predictions_path = output_dir / "postfailure-predictions.json"
    report_path = output_dir / "v12-report.json"
    programs_path.write_text(json.dumps(all_programs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps(all_predictions, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "tasks": len(rows),
        "fresh_cases_per_runtime": len(rows) * HOLDOUT_COUNT,
        "report_sha256": sha256_file(report_path),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v12"))
    args = parser.parse_args()
    run(args.campaign_root, args.arcgen_root, args.output_dir)


if __name__ == "__main__":
    main()
