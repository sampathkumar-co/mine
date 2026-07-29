from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V17 = ROOT / "v17"
if str(V17) not in sys.path:
    sys.path.insert(0, str(V17))

from constructive_dsl_v17 import (
    FORBIDDEN_OPS,
    as_grid,
    execute,
    sha256_json,
    synthesize,
    walk_ops,
)
from cosynthesize_verifier_v17 import synthesize_contract
from portable_constructive_dsl_v17 import execute_portable

FROZEN_V17_COMMIT = "cd89382e38b45d12916e662af052a7aa1a374896"
V17_EVIDENCE_SHA256 = "5ecedba17a6fdacfe1e9a6ec0ca1938e60b7474d65502f05d0876fbee6f7f771"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def demonstrations(task: dict[str, Any]):
    return [
        (as_grid(item["input"]), as_grid(item["output"]))
        for item in task["train"]
    ]


def test_inputs(task: dict[str, Any]):
    return [as_grid(item["input"]) for item in task["test"]]


def demonstration_hash(examples) -> str:
    return sha256_json([
        {"input": source, "output": target}
        for source, target in examples
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "V18_PREDICTIONS.json",
    )
    args = parser.parse_args()

    precommit_path = HERE / "V18_PRECOMMIT.json"
    precommit = load_json(precommit_path)
    if precommit["v17_frozen_commit"] != FROZEN_V17_COMMIT:
        raise RuntimeError("v17 frozen commit changed")
    if precommit["v17_evidence_sha256"] != V17_EVIDENCE_SHA256:
        raise RuntimeError("v17 evidence identity changed")
    if precommit["hidden_outputs_opened"]:
        raise RuntimeError("precommit says hidden outputs were opened")

    grammar_path = V17 / "constructive_dsl_v17.py"
    portable_path = V17 / "portable_constructive_dsl_v17.py"
    grammar_hash = file_sha256(grammar_path)
    portable_hash = file_sha256(portable_path)
    programs_dir = HERE / "programs"
    contracts_dir = HERE / "contracts"
    predictions_dir = HERE / "predictions"
    outcomes: list[dict[str, Any]] = []

    for gate_item in precommit["gates"]:
        gate = int(gate_item["gate"])
        package = args.package_root / f"v13-campaign-{gate:02d}"
        redacted_path = package / "redacted-task.json"
        seal_path = package / "seal-manifest.json"
        if file_sha256(redacted_path) != gate_item["redacted_task_sha256"]:
            raise RuntimeError(f"redacted package hash changed for gate {gate}")
        if file_sha256(seal_path) != gate_item["seal_manifest_sha256"]:
            raise RuntimeError(f"seal manifest hash changed for gate {gate}")

        task = load_json(redacted_path)
        examples = demonstrations(task)
        inputs = test_inputs(task)
        if len(examples) != int(gate_item["train_cases"]):
            raise RuntimeError(f"training denominator changed for gate {gate}")
        if len(inputs) != int(gate_item["sealed_test_inputs"]):
            raise RuntimeError(f"test-input denominator changed for gate {gate}")

        outcome: dict[str, Any] = {
            "gate": gate,
            "task_id": str(gate_item["task_id"]),
            "demonstrations": len(examples),
            "sealed_test_inputs": len(inputs),
            "synthesis_attempts": 1,
            "hidden_outputs_opened": False,
        }

        try:
            program, search = synthesize(examples)
        except RuntimeError as exc:
            outcome.update({
                "status": "no_program",
                "scoring_eligible": False,
                "failure": str(exc),
            })
            outcomes.append(outcome)
            print(json.dumps(outcome, sort_keys=True), flush=True)
            continue

        forbidden = sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
        if forbidden:
            raise RuntimeError(f"forbidden opcodes in gate {gate}: {forbidden}")
        if any(execute(program, source) != target for source, target in examples):
            raise RuntimeError(f"primary demonstration replay failed for gate {gate}")
        if any(execute_portable(program, source) != target for source, target in examples):
            raise RuntimeError(f"portable demonstration replay failed for gate {gate}")

        primary_predictions = [execute(program, source) for source in inputs]
        portable_predictions = [execute_portable(program, source) for source in inputs]
        if primary_predictions != portable_predictions:
            raise RuntimeError(f"runtime prediction disagreement for gate {gate}")

        demo_hash = demonstration_hash(examples)
        contract, mutation_cases, manifest = synthesize_contract(
            program,
            examples,
            constructive_grammar_sha256=grammar_hash,
            portable_runtime_sha256=portable_hash,
            demonstration_sha256=demo_hash,
            mutation_limit=64,
            revision=0,
        )
        if contract["exact_digest_used"]:
            raise RuntimeError(f"learned verifier used exact digest for gate {gate}")

        program_path = programs_dir / f"v18-program-{gate:02d}.json"
        contract_path = contracts_dir / f"v18-contract-{gate:02d}.json"
        prediction_path = predictions_dir / f"v18-predictions-{gate:02d}.json"
        prediction_payload = {
            "schema": "lexigen-v18-sealed-predictions-v1",
            "gate": gate,
            "program_sha256": sha256_json(program),
            "contract_sha256": contract["contract_sha256"],
            "test_input_sha256": [sha256_json(source) for source in inputs],
            "predictions": primary_predictions,
            "hidden_outputs_opened": False,
        }
        write_json(program_path, program)
        write_json(contract_path, contract)
        write_json(prediction_path, prediction_payload)

        outcome.update({
            "status": "predictions_committed_locally",
            "scoring_eligible": True,
            "search": search,
            "program_sha256": sha256_json(program),
            "contract_sha256": contract["contract_sha256"],
            "prediction_sha256": file_sha256(prediction_path),
            "training_mutation_cases": len(mutation_cases),
            "mutation_manifest_size": len(manifest),
            "learned_exact_digest_used": False,
            "forbidden_opcode_hits": forbidden,
            "runtime_agreement": len(inputs),
        })
        outcomes.append(outcome)
        print(json.dumps(outcome, sort_keys=True), flush=True)

    report = {
        "schema": "lexigen-v18-sealed-prediction-report-v1",
        "v17_frozen_commit": FROZEN_V17_COMMIT,
        "v17_evidence_sha256": V17_EVIDENCE_SHA256,
        "precommit_sha256": file_sha256(precommit_path),
        "hidden_outputs_opened": False,
        "gates_attempted": len(outcomes),
        "programs_found": sum(item["status"] != "no_program" for item in outcomes),
        "no_program_gates": sum(item["status"] == "no_program" for item in outcomes),
        "scoring_eligible_gates": sum(item["scoring_eligible"] for item in outcomes),
        "world_level_breakthrough": False,
        "outcomes": outcomes,
    }
    write_json(args.output, report)
    print("SUMMARY", json.dumps({
        "gates_attempted": report["gates_attempted"],
        "programs_found": report["programs_found"],
        "no_program_gates": report["no_program_gates"],
        "scoring_eligible_gates": report["scoring_eligible_gates"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
