from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V15 = HERE.parent / "v15"
for folder in (HERE, V15):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from cosynthesize_verifier_v16 import synthesize_contract
from induce_language_v15 import load_programs
from ir_runtime_v15 import as_grid, execute
from mutations_v16 import apply_output_mutation
from portable_ir_runtime_v15 import execute_portable_ir
from portable_verifier_v16 import (
    screening_holds_portable,
    verify_against_reference_portable,
)
from verifier_grammar_v16 import screening_holds, verify_against_reference


def generate(task: str, arcgen_root: Path, seed: int):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    random.seed(seed)
    pair = importlib.import_module(f"tasks.task_{task}").generate()
    return as_grid(pair["input"]), as_grid(pair["output"])


def _fresh_cases(task: str, gate: int, arcgen_root: Path, count: int):
    cases = []
    attempts = rejections = 0
    while len(cases) < count:
        seed = 4_200_000 + gate * 100_000 + attempts
        attempts += 1
        try:
            cases.append(generate(task, arcgen_root, seed))
        except (ValueError, IndexError, TypeError, RuntimeError):
            rejections += 1
            if attempts > count * 5 + 1000:
                raise RuntimeError(f"too many generator rejections for gate {gate}")
    return cases, attempts, rejections


def _evaluate_contract(
    contract: dict[str, Any],
    program: dict[str, Any],
    fresh_cases,
    manifest,
    *,
    mutations_per_case: int,
):
    correct_primary = correct_portable = 0
    mutant_cases = 0
    screening_rejected_primary = screening_rejected_portable = 0
    soundness_rejected_primary = soundness_rejected_portable = 0
    false_accepts: list[dict[str, Any]] = []
    for case_index, (source, target) in enumerate(fresh_cases):
        reference = execute(program, source)
        portable_reference = execute_portable_ir(program, source)
        if reference != target or portable_reference != reference:
            raise RuntimeError("frozen v15 program failed a v16 fresh target")
        correct_primary += verify_against_reference(
            contract, program, source, reference, reference
        )
        correct_portable += verify_against_reference_portable(
            contract, program, source, reference, portable_reference
        )
        used_mutations = 0
        for mutation in manifest:
            if used_mutations >= mutations_per_case:
                break
            try:
                if mutation["kind"] == "ast":
                    candidate = execute(mutation["ast"], source)
                else:
                    candidate = apply_output_mutation(reference, str(mutation["operator"]))
            except Exception:
                continue
            if candidate == reference:
                continue
            used_mutations += 1
            mutant_cases += 1
            primary_screen = screening_holds(contract, source, candidate, reference)
            portable_screen = screening_holds_portable(
                contract, source, candidate, portable_reference
            )
            primary_full = verify_against_reference(
                contract, program, source, candidate, reference
            )
            portable_full = verify_against_reference_portable(
                contract, program, source, candidate, portable_reference
            )
            screening_rejected_primary += not primary_screen
            screening_rejected_portable += not portable_screen
            soundness_rejected_primary += not primary_full
            soundness_rejected_portable += not portable_full
            if primary_screen or portable_screen:
                false_accepts.append({
                    "example_index": 100_000 + case_index,
                    "mutation_sha256": mutation["mutation_sha256"],
                    "source": source,
                    "candidate": candidate,
                    "reference": reference,
                })
    return {
        "correct_primary": correct_primary,
        "correct_portable": correct_portable,
        "mutant_cases": mutant_cases,
        "screening_rejected_primary": screening_rejected_primary,
        "screening_rejected_portable": screening_rejected_portable,
        "soundness_rejected_primary": soundness_rejected_primary,
        "soundness_rejected_portable": soundness_rejected_portable,
        "false_accepts": false_accepts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--mutations-per-case", type=int, default=8)
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=HERE / "V16_REPORT.json")
    args = parser.parse_args()

    programs, examples_by_gate, metadata = load_programs(
        args.v14_evidence,
        args.package_root,
    )
    contracts_dir = HERE / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    reports = []

    for program, item in zip(programs, metadata):
        gate, task = int(item["gate"]), str(item["task"])
        examples = examples_by_gate[gate]
        fresh, attempts, generator_rejections = _fresh_cases(
            task,
            gate,
            args.arcgen_root,
            args.cases,
        )
        counterexamples: list[dict[str, Any]] = []
        revisions = []
        final_contract = None
        final_evaluation = None
        manifest = None
        for revision in range(args.max_revisions + 1):
            contract, training_cases, manifest = synthesize_contract(
                program,
                examples,
                additional_cases=counterexamples,
                revision=revision,
            )
            evaluation = _evaluate_contract(
                contract,
                program,
                fresh,
                manifest,
                mutations_per_case=args.mutations_per_case,
            )
            revisions.append(
                {
                    "revision": revision,
                    "predicates": contract["predicates"],
                    "predicate_cost": contract["predicate_cost"],
                    "training_mutation_cases": len(training_cases),
                    "fresh_mutant_cases": evaluation["mutant_cases"],
                    "fresh_false_accepts": len(evaluation["false_accepts"]),
                    "exact_digest_used": contract["exact_digest_used"],
                }
            )
            final_contract, final_evaluation = contract, evaluation
            if not evaluation["false_accepts"]:
                break
            counterexamples.extend(evaluation["false_accepts"])
        else:
            raise RuntimeError(f"v16 CEGIS budget exhausted for gate {gate}")

        if final_contract is None or final_evaluation is None or manifest is None:
            raise RuntimeError("verifier synthesis produced no contract")
        if final_evaluation["correct_primary"] != len(fresh):
            raise RuntimeError(f"primary verifier rejected correct outputs for gate {gate}")
        if final_evaluation["correct_portable"] != len(fresh):
            raise RuntimeError(f"portable verifier rejected correct outputs for gate {gate}")
        for key in (
            "screening_rejected_primary",
            "screening_rejected_portable",
            "soundness_rejected_primary",
            "soundness_rejected_portable",
        ):
            if final_evaluation[key] != final_evaluation["mutant_cases"]:
                raise RuntimeError(f"{key} missed mutations for gate {gate}")

        contract_path = contracts_dir / f"v16-contract-{gate:02d}.json"
        contract_path.write_bytes(
            (json.dumps(final_contract, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        report = {
            "gate": gate,
            "task": task,
            "accepted_cases": len(fresh),
            "generator_attempts": attempts,
            "generator_rejections": generator_rejections,
            "mutation_manifest_size": len(manifest),
            "fresh_mutant_cases": final_evaluation["mutant_cases"],
            "correct_primary": final_evaluation["correct_primary"],
            "correct_portable": final_evaluation["correct_portable"],
            "screening_rejected_primary": final_evaluation["screening_rejected_primary"],
            "screening_rejected_portable": final_evaluation["screening_rejected_portable"],
            "soundness_rejected_primary": final_evaluation["soundness_rejected_primary"],
            "soundness_rejected_portable": final_evaluation["soundness_rejected_portable"],
            "shape_only_survivors": final_contract["shape_only_survivors"],
            "final_predicates": final_contract["predicates"],
            "final_predicate_cost": final_contract["predicate_cost"],
            "exact_digest_used": final_contract["exact_digest_used"],
            "contract_sha256": final_contract["contract_sha256"],
            "revisions": revisions,
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = {
        "schema": "lexigen-v16-verifier-cosynthesis-report-v1",
        "families": len(reports),
        "cases_per_family": args.cases,
        "mutations_per_case": args.mutations_per_case,
        "correct_outputs_checked": sum(item["accepted_cases"] for item in reports),
        "mutant_outputs_checked": sum(item["fresh_mutant_cases"] for item in reports),
        "screening_rejections": sum(item["screening_rejected_primary"] for item in reports),
        "portable_screening_rejections": sum(item["screening_rejected_portable"] for item in reports),
        "soundness_rejections": sum(item["soundness_rejected_primary"] for item in reports),
        "portable_soundness_rejections": sum(item["soundness_rejected_portable"] for item in reports),
        "contracts_using_exact_digest": sum(item["exact_digest_used"] for item in reports),
        "contracts_requiring_revision": sum(len(item["revisions"]) > 1 for item in reports),
        "target_used_by_verifier": False,
        "world_level_breakthrough": False,
        "reports": reports,
    }
    args.output.write_bytes((json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("SUMMARY", json.dumps({key: summary[key] for key in (
        "families",
        "correct_outputs_checked",
        "mutant_outputs_checked",
        "screening_rejections",
        "portable_screening_rejections",
        "soundness_rejections",
        "portable_soundness_rejections",
        "contracts_using_exact_digest",
        "contracts_requiring_revision",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
