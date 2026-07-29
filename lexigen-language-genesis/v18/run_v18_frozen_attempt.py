from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V17 = ROOT / "v17"
for folder in (V17, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from constructive_dsl_v17 import (
    FORBIDDEN_OPS,
    as_grid,
    canonical,
    execute,
    sha256_json,
    synthesize,
    walk_ops,
)
from portable_constructive_dsl_v17 import execute_portable
from cosynthesize_verifier_v17 import synthesize_contract
from verifier_grammar_v17 import VERIFIER_GRAMMAR_SHA256

FROZEN_V17_COMMIT = "cd89382e38b45d12916e662af052a7aa1a374896"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def task_id_hits(value: Any, task_id: str) -> list[str]:
    text = canonical(value)
    return [task_id] if task_id in text else []


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_v18_frozen_attempt.py <visible-package-root>")
    package_root = Path(sys.argv[1])
    precommit = load_json(HERE / "V18_PRECOMMIT.json")

    programs_dir = HERE / "programs"
    contracts_dir = HERE / "contracts"
    predictions_dir = HERE / "predictions"
    programs_dir.mkdir(exist_ok=True)
    contracts_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)

    constructive_hash = file_sha256(V17 / "constructive_dsl_v17.py")
    portable_hash = file_sha256(V17 / "portable_constructive_dsl_v17.py")
    synthesizer_hash = constructive_hash
    reports: list[dict[str, Any]] = []

    for item in precommit["gates"]:
        gate = int(item["gate"])
        expected_task_id = str(item["task_id"])
        package_path = package_root / f"v13-campaign-{gate:02d}" / "redacted-task.json"
        if file_sha256(package_path) != item["redacted_task_sha256"]:
            raise RuntimeError(f"redacted package hash mismatch for gate {gate}")
        package = load_json(package_path)
        if str(package["selected_task_id"]) != expected_task_id:
            raise RuntimeError(f"task identity mismatch for gate {gate}")
        demonstrations = [
            (as_grid(example["input"]), as_grid(example["output"]))
            for example in package["train"]
        ]
        test_inputs = [as_grid(example["input"]) for example in package["test"]]
        demonstration_hash = sha256_json([
            {"input": source, "output": target}
            for source, target in demonstrations
        ])

        report: dict[str, Any] = {
            "gate": gate,
            "task_id": expected_task_id,
            "demonstrations": len(demonstrations),
            "sealed_test_inputs": len(test_inputs),
            "redacted_task_sha256": item["redacted_task_sha256"],
            "hidden_outputs_opened": False,
            "scored": False,
            "retry_budget_after_scoring": 0,
        }

        try:
            program, search = synthesize(demonstrations)
        except RuntimeError as error:
            report.update({
                "status": "no_program",
                "reason": str(error),
                "predictions_committed": False,
                "permanently_unscored": True,
            })
            reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
            continue

        forbidden_hits = sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
        if forbidden_hits:
            raise RuntimeError(f"forbidden opcodes generated for gate {gate}: {forbidden_hits}")
        if task_id_hits(program, expected_task_id):
            raise RuntimeError(f"task id leaked into generated program for gate {gate}")

        primary_demo = sum(execute(program, source) == target for source, target in demonstrations)
        portable_demo = sum(execute_portable(program, source) == target for source, target in demonstrations)
        if primary_demo != len(demonstrations) or portable_demo != len(demonstrations):
            raise RuntimeError(f"demonstration replay failed for gate {gate}")

        contract, mutation_cases, manifest = synthesize_contract(
            program,
            demonstrations,
            constructive_grammar_sha256=constructive_hash,
            portable_runtime_sha256=portable_hash,
            demonstration_sha256=demonstration_hash,
            mutation_limit=64,
            revision=0,
         )
        if contract["exact_digest_used"]:
            raise RuntimeError(f"learned contract used exact digest for gate {gate}")

        predictions = []
        runtime_agreement = 0
        for index, source in enumerate(test_inputs):
            primary = execute(program, source)
            portable = execute_portable(program, source)
            runtime_agreement += primary == portable
            predictions.append({
                "index": index,
                "input_sha256": sha256_json(source),
                "output": primary,
                "output_sha256": sha256_json(primary),
                "portable_output_sha256": sha256_json(portable),
            })
        if runtime_agreement != len(test_inputs):
            raise RuntimeError(f"test runtime disagreement for gate {gate}")

        program_path = programs_dir / f"v18-gate-{gate:02d}-program.json"
        contract_path = contracts_dir / f"v18-gate-{gate:02d}-contract.json"
        prediction_path = predictions_dir / f"v18-gate-{gate:02d}-predictions.json"
        write_json(program_path, program)
        write_json(contract_path, contract)
        write_json(prediction_path, {
            "schema": "lexigen-v18-sealed-predictions-v1",
            "gate": gate,
            "task_id": expected_task_id,
            "program_sha256": sha256_json(program),
            "contract_sha256": contract["contract_sha256"],
            "predictions": predictions,
            "hidden_outputs_opened": False,
            "scored": False,
        })

        report.update({
            "status": "candidate_committed",
            "search": search,
            "program_sha256": sha256_json(program),
            "program_file_sha256": file_sha256(program_path),
            "contract_sha256": contract["contract_sha256"],
            "contract_file_sha256": file_sha256(contract_path),
            "prediction_file_sha256": file_sha256(prediction_path),
            "prediction_outputs_sha256": sha256_json([entry["output_sha256"] for entry in predictions]),
            "demonstration_primary_exact": primary_demo,
            "demonstration_portable_exact": portable_demo,
            "sealed_test_runtime_agreement": runtime_agreement,
            "training_mutation_cases": len(mutation_cases),
            "mutation_manifest_size": len(manifest),
            "learned_exact_digest_used": contract["exact_digest_used"],
            "forbidden_opcode_hits": forbidden_hits,
            "task_id_hits": task_id_hits(program, expected_task_id),
            "predictions_committed": True,
            "permanently_unscored": False,
        })
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = {
        "schema": "lexigen-v18-frozen-attempt-report-v1",
        "v17_frozen_commit": FROZEN_V17_COMMIT,
        "v17_evidence_sha256": precommit["v17_evidence_sha256"],
        "constructive_grammar_sha256": constructive_hash,
        "portable_runtime_sha256": portable_hash,
        "synthesizer_sha256": synthesizer_hash,
        "verifier_grammar_sha256": VERIFIER_GRAMMAR_SHA256,
        "gates_precommitted": len(reports),
        "candidate_programs": sum(item["status"] == "candidate_committed" for item in reports),
        "no_program_failures": sum(item["status"] == "no_program" for item in reports),
        "hidden_outputs_opened": False,
        "gates_scored": 0,
        "post_score_retries": 0,
        "world_level_breakthrough": False,
        "reports": reports,
    }
    write_json(HERE / "V18_ATTEMPT_REPORT.json", summary)
    print("SUMMARY", json.dumps({
        "gates": summary["gates_precommitted"],
        "candidates": summary["candidate_programs"],
        "no_program": summary["no_program_failures"],
        "hidden_outputs_opened": summary["hidden_outputs_opened"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
