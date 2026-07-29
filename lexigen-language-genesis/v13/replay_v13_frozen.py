from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from latent_runtime_v13 import as_grid, to_json_grid
from latent_runtime_v13_ext3 import execute_program
from latent_synthesizer_v13 import canonical_json
from latent_synthesizer_v13_final import VALIDATED_OPERATORS, synthesize_latent_final
from portable_runtime_v13 import execute_portable

ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
HOLDOUT_COUNT = 10_000
GATES = (
    "v12-campaign-01",
    "v12-campaign-02",
    "v12-campaign-03",
    "v12-campaign-06",
    "v12-campaign-09",
    "v12-campaign-10",
    "v12-campaign-11",
    "v12-campaign-12",
    "v12-campaign-13",
    "v12-campaign-14",
    "v12-campaign-15",
    "v12-campaign-17",
    "v12-campaign-19",
    "v12-campaign-20",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(evidence_root: Path, arcgen_root: Path, output_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(arcgen_root))
    import task_list  # type: ignore

    registry = task_list.task_list()
    rows = []
    programs = {}
    predictions = {}
    for gate_index, gate in enumerate(GATES):
        folder = evidence_root / gate
        package = json.loads((folder / "redacted-task.json").read_text(encoding="utf-8"))
        examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in package["train"]]
        tests = [as_grid(item["input"]) for item in package["test"]]
        result = synthesize_latent_final(examples)
        if result.program is None:
            raise AssertionError(f"v13 final language found no program for {gate}")
        generated = result.program
        operator = generated["operator"]
        if operator not in VALIDATED_OPERATORS:
            raise AssertionError(f"unvalidated operator selected: {operator}")
        training_exact = all(execute_program(generated, source) == target for source, target in examples)
        portable_training = all(
            execute_portable(generated, source) == [list(row) for row in target]
            for source, target in examples
        )
        generator, _ = registry[package["selected_task_id"]]
        primary_failures = []
        portable_failures = []
        for offset in range(HOLDOUT_COUNT):
            seed = 500_000 + gate_index * 20_000 + offset
            random.seed(seed)
            item = generator()
            source = as_grid(item["input"])
            target = as_grid(item["output"])
            primary = execute_program(generated, source)
            portable = execute_portable(generated, source)
            if primary != target:
                primary_failures.append(seed)
            if portable != [list(row) for row in primary] or primary != target:
                portable_failures.append(seed)
            if (offset + 1) % 1_000 == 0:
                print(
                    json.dumps(
                        {
                            "gate": gate,
                            "validated": offset + 1,
                            "primary_failures": len(primary_failures),
                            "portable_failures": len(portable_failures),
                        }
                    ),
                    flush=True,
                )
        row = {
            "gate": gate,
            "task": package["selected_task_id"],
            "operator": operator,
            "program_name": generated["name"],
            "program_sha256": hashlib.sha256(canonical_json(generated).encode()).hexdigest(),
            "candidates_tested": result.candidates_tested,
            "exact_candidate_count": result.exact_candidate_count,
            "training_exact": training_exact,
            "portable_training_exact": portable_training,
            "holdout_count": HOLDOUT_COUNT,
            "primary_holdout_exact": HOLDOUT_COUNT - len(primary_failures),
            "portable_holdout_exact": HOLDOUT_COUNT - len(portable_failures),
            "primary_failures": primary_failures,
            "portable_failures": portable_failures,
        }
        rows.append(row)
        programs[gate] = generated
        predictions[gate] = [to_json_grid(execute_program(generated, source)) for source in tests]

    checks = {
        "fourteen_task_families": len(rows) == 14,
        "fourteen_validated_operators": set(row["operator"] for row in rows) == VALIDATED_OPERATORS,
        "all_training_exact": all(row["training_exact"] for row in rows),
        "all_portable_training_exact": all(row["portable_training_exact"] for row in rows),
        "all_primary_holdouts_exact": all(row["primary_holdout_exact"] == HOLDOUT_COUNT for row in rows),
        "all_portable_holdouts_exact": all(row["portable_holdout_exact"] == HOLDOUT_COUNT for row in rows),
        "sealed_outputs_untouched": True,
    }
    report = {
        "version": "v13",
        "status": "post-failure fourteen-family inverse-generator mechanism candidate; not a blind breakthrough",
        "arcgen_commit": ARCGEN_COMMIT,
        "source_gates": list(GATES),
        "source_sealed_outputs_accessed": False,
        "fresh_cases_per_runtime": len(rows) * HOLDOUT_COUNT,
        "validated_operator_inventory": sorted(VALIDATED_OPERATORS),
        "rows": rows,
        "gate": checks,
        "claim_boundary": (
            "The v13 language was designed from already-preserved v12 failures. "
            "A fresh precommitted sealed campaign is required before any breakthrough claim."
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"v13 frozen gate failed: {checks}")
    output_dir.mkdir(parents=True, exist_ok=True)
    programs_path = output_dir / "v13-programs.json"
    predictions_path = output_dir / "postfailure-predictions.json"
    report_path = output_dir / "v13-report.json"
    programs_path.write_text(json.dumps(programs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks": len(rows),
                "operators": len(VALIDATED_OPERATORS),
                "fresh_cases_per_runtime": report["fresh_cases_per_runtime"],
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v13"))
    args = parser.parse_args()
    run(args.evidence_root, args.arcgen_root, args.output_dir)


if __name__ == "__main__":
    main()
