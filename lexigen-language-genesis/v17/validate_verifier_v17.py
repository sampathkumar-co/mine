from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from constructive_dsl_v17 import as_grid, execute, sha256_json
from cosynthesize_verifier_v17 import synthesize_contract
from mutations_v17 import apply_output_mutation
from portable_constructive_dsl_v17 import execute_portable
from portable_verifier_v17 import (
    screening_holds_portable,
    verify_against_reference_portable,
)
from verifier_grammar_v17 import (
    screening_holds,
    verify_against_reference,
)

TARGET_GATES = (1, 2, 3)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_families(v14_evidence: Path) -> dict[int, dict[str, Any]]:
    evidence = json.loads(v14_evidence.read_text(encoding="utf-8"))
    return {
        int(item["gate"]): item
        for item in evidence["families_report"]
        if int(item["gate"]) in TARGET_GATES
    }


def load_demonstrations(package_root: Path, gate: int):
    package = json.loads(
        (
            package_root
            / f"v13-campaign-{gate:02d}"
            / "redacted-task.json"
        ).read_text(encoding="utf-8")
    )
    return [
        (as_grid(item["input"]), as_grid(item["output"]))
        for item in package["train"]
    ]


def generate(task: str, arcgen_root: Path, seed: int):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    random.seed(seed)
    pair = importlib.import_module(f"tasks.task_{task}").generate()
    return as_grid(pair["input"]), as_grid(pair["output"])


def fresh_cases(task: str, gate: int, arcgen_root: Path, count: int):
    cases = []
    attempts = rejections = 0
    while len(cases) < count:
        seed = 6_200_000 + gate * 100_000 + attempts
        attempts += 1
        try:
            cases.append(generate(task, arcgen_root, seed))
        except (ValueError, IndexError, TypeError, RuntimeError):
            rejections += 1
            if attempts > count * 5 + 1000:
                raise RuntimeError(f"too many generator rejections for gate {gate}")
    return cases, attempts, rejections


def evaluate_contract(
    contract: dict[str, Any],
    program: dict[str, Any],
    cases,
    manifest,
    *,
    mutations_per_case: int,
):
    correct_primary = correct_portable = 0
    mutant_cases = runtime_invalid_mutations = 0
    screening_rejected_primary = screening_rejected_portable = 0
    soundness_rejected_primary = soundness_rejected_portable = 0
    false_accepts: list[dict[str, Any]] = []
    for case_index, (source, target) in enumerate(cases):
        reference = execute(program, source)
        portable_reference = execute_portable(program, source)
        if reference != target or portable_reference != reference:
            raise RuntimeError("frozen v17 program failed a fresh target")
        correct_primary += verify_against_reference(
            contract, program, source, reference, reference
        )
        correct_portable += verify_against_reference_portable(
            contract, program, source, reference, portable_reference
        )
        used = 0
        for mutation in manifest:
            if used >= mutations_per_case:
                break
            try:
                if mutation["kind"] == "ast":
                    candidate = execute(mutation["ast"], source)
                else:
                    candidate = apply_output_mutation(reference, str(mutation["operator"]))
            except Exception:
                runtime_invalid_mutations += 1
                continue
            if candidate == reference:
                continue
            used += 1
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
                false_accepts.append(
                    {
                        "example_index": 100_000 + case_index,
                        "mutation_sha256": mutation["mutation_sha256"],
                        "source": source,
                        "candidate": candidate,
                        "reference": reference,
                    }
                )
    return {
        "correct_primary": correct_primary,
        "correct_portable": correct_portable,
        "mutant_cases": mutant_cases,
        "runtime_invalid_mutations": runtime_invalid_mutations,
        "screening_rejected_primary": screening_rejected_primary,
        "screening_rejected_portable": screening_rejected_portable,
        "soundness_rejected_primary": soundness_rejected_primary,
        "soundness_rejected_portable": soundness_rejected_portable,
        "false_accepts": false_accepts,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--mutations-per-case", type=int, default=8)
    parser.add_argument("--max-revisions", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=HERE / "V17_VERIFIER_REPORT.json"
    )
    args = parser.parse_args()

    families = load_families(args.v14_evidence)
    if sorted(families) != list(TARGET_GATES):
        raise RuntimeError("frozen target-family set is incomplete")

    constructive_hash = file_sha256(HERE / "constructive_dsl_v17.py")
    portable_hash = file_sha256(HERE / "portable_constructive_dsl_v17.py")
    contracts_dir = HERE / "contracts"
    revisions_dir = contracts_dir / "revisions"
    revisions_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for gate in TARGET_GATES:
        task = str(families[gate]["task"])
        program = json.loads(
            (HERE / "programs" / f"v17-program-{gate:02d}.json").read_text(
                encoding="utf-8"
            )
        )
        examples = load_demonstrations(args.package_root, gate)
        demonstration_hash = sha256_json(
            [{"input": source, "output": target} for source, target in examples]
        )
        cases, attempts, generator_rejections = fresh_cases(
            task, gate, args.arcgen_root, args.cases
        )
        counterexamples: list[dict[str, Any]] = []
        revisions = []
        final_contract = final_evaluation = manifest = None

        for revision in range(args.max_revisions + 1):
            contract, training_cases, manifest = synthesize_contract(
                program,
                examples,
                constructive_grammar_sha256=constructive_hash,
                portable_runtime_sha256=portable_hash,
                demonstration_sha256=demonstration_hash,
                additional_cases=counterexamples,
                revision=revision,
            )
            evaluation = evaluate_contract(
                contract,
                program,
                cases,
                manifest,
                mutations_per_case=args.mutations_per_case,
            )
            revision_record = {
                "revision": revision,
                "contract_sha256": contract["contract_sha256"],
                "predicates": contract["predicates"],
                "predicate_cost": contract["predicate_cost"],
                "training_mutation_cases": len(training_cases),
                "training_runtime_invalid_mutations": contract[
                    "training_runtime_invalid_mutations"
                ],
                "fresh_mutant_cases": evaluation["mutant_cases"],
                "fresh_runtime_invalid_mutations": evaluation[
                    "runtime_invalid_mutations"
                ],
                "fresh_false_accepts": len(evaluation["false_accepts"]),
                "exact_digest_used": contract["exact_digest_used"],
            }
            revisions.append(revision_record)
            write_json(
                revisions_dir / f"v17-gate-{gate:02d}-revision-{revision}.json",
                contract,
            )
            write_json(
                revisions_dir
                / f"v17-gate-{gate:02d}-revision-{revision}-false-accepts.json",
                evaluation["false_accepts"],
            )
            final_contract, final_evaluation = contract, evaluation
            if not evaluation["false_accepts"]:
                break
            counterexamples.extend(evaluation["false_accepts"])
        else:
            raise RuntimeError(f"v17 CEGIS budget exhausted for gate {gate}")

        if final_contract is None or final_evaluation is None or manifest is None:
            raise RuntimeError("verifier synthesis produced no contract")
        if final_contract["constructive_grammar_sha256"] != constructive_hash:
            raise RuntimeError("constructive grammar binding mismatch")
        if final_contract["portable_runtime_sha256"] != portable_hash:
            raise RuntimeError("portable runtime binding mismatch")
        if final_contract["demonstration_sha256"] != demonstration_hash:
            raise RuntimeError("demonstration binding mismatch")
        if final_evaluation["correct_primary"] != len(cases):
            raise RuntimeError(f"primary verifier rejected correct outputs for gate {gate}")
        if final_evaluation["correct_portable"] != len(cases):
            raise RuntimeError(f"portable verifier rejected correct outputs for gate {gate}")
        for key in (
            "screening_rejected_primary",
            "screening_rejected_portable",
            "soundness_rejected_primary",
            "soundness_rejected_portable",
        ):
            if final_evaluation[key] != final_evaluation["mutant_cases"]:
                raise RuntimeError(f"{key} missed mutations for gate {gate}")
        if final_contract["exact_digest_used"]:
            raise RuntimeError(f"learned screen used exact digest for gate {gate}")

        write_json(contracts_dir / f"v17-contract-{gate:02d}.json", final_contract)
        report = {
            "gate": gate,
            "task": task,
            "accepted_cases": len(cases),
            "generator_attempts": attempts,
            "generator_rejections": generator_rejections,
            "mutation_manifest_size": len(manifest),
            "fresh_mutant_cases": final_evaluation["mutant_cases"],
            "fresh_runtime_invalid_mutations": final_evaluation[
                "runtime_invalid_mutations"
            ],
            "correct_primary": final_evaluation["correct_primary"],
            "correct_portable": final_evaluation["correct_portable"],
            "screening_rejected_primary": final_evaluation[
                "screening_rejected_primary"
            ],
            "screening_rejected_portable": final_evaluation[
                "screening_rejected_portable"
            ],
            "soundness_rejected_primary": final_evaluation[
                "soundness_rejected_primary"
            ],
            "soundness_rejected_portable": final_evaluation[
                "soundness_rejected_portable"
            ],
            "shape_only_survivors": final_contract["shape_only_survivors"],
            "final_predicates": final_contract["predicates"],
            "final_predicate_cost": final_contract["predicate_cost"],
            "exact_digest_used": final_contract["exact_digest_used"],
            "contract_sha256": final_contract["contract_sha256"],
            "program_sha256": final_contract["program_sha256"],
            "demonstration_sha256": demonstration_hash,
            "revisions": revisions,
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = {
        "schema": "lexigen-v17-verifier-cosynthesis-report-v1",
        "families": len(reports),
        "cases_per_family": args.cases,
        "mutations_per_case": args.mutations_per_case,
        "correct_outputs_checked": sum(item["accepted_cases"] for item in reports),
        "mutant_outputs_checked": sum(item["fresh_mutant_cases"] for item in reports),
        "runtime_invalid_mutations": sum(
            item["fresh_runtime_invalid_mutations"] for item in reports
        ),
        "screening_rejections": sum(
            item["screening_rejected_primary"] for item in reports
        ),
        "portable_screening_rejections": sum(
            item["screening_rejected_portable"] for item in reports
        ),
        "soundness_rejections": sum(
            item["soundness_rejected_primary"] for item in reports
        ),
        "portable_soundness_rejections": sum(
            item["soundness_rejected_portable"] for item in reports
        ),
        "contracts_using_exact_digest": sum(
            item["exact_digest_used"] for item in reports
        ),
        "contracts_requiring_revision": sum(
            len(item["revisions"]) > 1 for item in reports
        ),
        "constructive_grammar_sha256": constructive_hash,
        "portable_runtime_sha256": portable_hash,
        "target_used_by_verifier": False,
        "world_level_breakthrough": False,
        "reports": reports,
    }
    write_json(args.output, summary)
    print(
        "SUMMARY",
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "families",
                    "correct_outputs_checked",
                    "mutant_outputs_checked",
                    "runtime_invalid_mutations",
                    "screening_rejections",
                    "portable_screening_rejections",
                    "soundness_rejections",
                    "portable_soundness_rejections",
                    "contracts_using_exact_digest",
                    "contracts_requiring_revision",
                )
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
