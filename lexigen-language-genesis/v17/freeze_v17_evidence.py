from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from constructive_dsl_v17 import FORBIDDEN_OPS, canonical, sha256_json, walk_ops

BASE_V16 = "851e6117b64ea74f5eaaa631f1611ed6205464a8"
ARCGEN = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
V13_VISIBLE_EVIDENCE = "13cd271a2d4813001563842a51a1e72dd100aa1a"

CODE_FILES = (
    "constructive_dsl_v17.py",
    "portable_constructive_dsl_v17.py",
    "validate_v17.py",
    "test_v17.py",
    "mutations_v17.py",
    "verifier_grammar_v17.py",
    "portable_verifier_v17.py",
    "cosynthesize_verifier_v17.py",
    "validate_verifier_v17.py",
    "test_verifier_v17.py",
    "freeze_v17_evidence.py",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_bytes(text.encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    construction_path = HERE / "V17_REPORT.json"
    verifier_path = HERE / "V17_VERIFIER_REPORT.json"
    construction = load(construction_path)
    verifier = load(verifier_path)

    require(construction["families"] == 3, "unexpected construction family count")
    require(construction["cases_per_family"] == 10_000, "construction denominator changed")
    require(construction["fresh_cases"] == 30_000, "construction case total changed")
    require(construction["fresh_failures"] == 0, "construction failures present")
    require(construction["runtime_disagreements"] == 0, "construction runtimes disagree")
    require(not construction["named_scene_opcodes_allowed"], "named scene operators allowed")
    require(not construction["task_ids_available_to_synthesizer"], "task IDs leaked")
    require(not construction["world_level_breakthrough"], "construction claim boundary weakened")

    require(verifier["families"] == 3, "unexpected verifier family count")
    require(verifier["correct_outputs_checked"] == 3_000, "verifier correct-output denominator changed")
    require(verifier["mutant_outputs_checked"] == 24_000, "verifier mutation denominator changed")
    require(verifier["screening_rejections"] == verifier["mutant_outputs_checked"], "primary screen admitted mutants")
    require(verifier["portable_screening_rejections"] == verifier["mutant_outputs_checked"], "portable screen admitted mutants")
    require(verifier["soundness_rejections"] == verifier["mutant_outputs_checked"], "primary anchor admitted mutants")
    require(verifier["portable_soundness_rejections"] == verifier["mutant_outputs_checked"], "portable anchor admitted mutants")
    require(verifier["contracts_using_exact_digest"] == 0, "learned screen used exact digest")
    require(verifier["contracts_requiring_revision"] >= 1, "CEGIS refinement was not exercised")
    require(not verifier["target_used_by_verifier"], "verifier target leakage flag changed")
    require(not verifier["world_level_breakthrough"], "verifier claim boundary weakened")

    program_files = sorted((HERE / "programs").glob("v17-program-*.json"))
    contract_files = sorted((HERE / "contracts").glob("v17-contract-*.json"))
    bootstrap_files = sorted((HERE / "contracts" / "bootstrap").glob("v17-contract-*.json"))
    revision_files = sorted((HERE / "contracts" / "revisions").glob("*.json"))
    require(len(program_files) == 3, "unexpected program count")
    require(len(contract_files) == 3, "unexpected final contract count")
    require(len(bootstrap_files) == 3, "bootstrap contracts were not preserved")
    require(len(revision_files) >= 6, "contract revision history is incomplete")

    programs = [load(path) for path in program_files]
    contracts = [load(path) for path in contract_files]
    task_ids = {str(item["task"]) for item in construction["reports"]}
    forbidden_hits = {
        path.name: sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
        for path, program in zip(program_files, programs)
    }
    task_id_hits = {
        path.name: sorted(task_id for task_id in task_ids if task_id in canonical(program))
        for path, program in zip(program_files, programs)
    }
    require(all(not hits for hits in forbidden_hits.values()), "forbidden opcode found")
    require(all(not hits for hits in task_id_hits.values()), "task ID embedded in program")

    construction_reports = {int(item["gate"]): item for item in construction["reports"]}
    verifier_reports = {int(item["gate"]): item for item in verifier["reports"]}
    require(sorted(construction_reports) == [1, 2, 3], "construction gates changed")
    require(sorted(verifier_reports) == [1, 2, 3], "verifier gates changed")

    for gate, (program, contract) in enumerate(zip(programs, contracts), start=1):
        require(sha256_json(program) == construction_reports[gate]["program_sha256"], f"program hash mismatch gate {gate}")
        require(contract["program_sha256"] == sha256_json(program), f"contract/program binding mismatch gate {gate}")
        require(contract["contract_sha256"] == verifier_reports[gate]["contract_sha256"], f"contract hash mismatch gate {gate}")
        require(not contract["exact_digest_used"], f"gate {gate} learned exact digest")
        require(contract["soundness_anchor"] == {"name": "exact_digest", "mandatory": True}, f"gate {gate} missing anchor")

    code_hashes = {name: file_sha256(HERE / name) for name in CODE_FILES}
    program_hashes = {
        path.name: {
            "file_sha256": file_sha256(path),
            "program_sha256": sha256_json(program),
            "node_count": construction_reports[index]["search"]["selected_node_count"],
            "search_family": construction_reports[index]["search"]["selected_search_family"],
        }
        for index, (path, program) in enumerate(zip(program_files, programs), start=1)
    }
    contract_hashes = {
        path.name: {
            "file_sha256": file_sha256(path),
            "contract_sha256": contract["contract_sha256"],
            "revision": contract["revision"],
            "predicates": contract["predicates"],
        }
        for path, contract in zip(contract_files, contracts)
    }

    evidence = {
        "schema": "lexigen-v17-frozen-evidence-v1",
        "version": 17,
        "base_v16_commit": BASE_V16,
        "arcgen_commit": ARCGEN,
        "v13_visible_evidence_commit": V13_VISIBLE_EVIDENCE,
        "module_layout": {
            "constructive_grammar_and_primary_runtime_share_module": True,
            "synthesizer_and_constructive_grammar_share_module": True,
        },
        "hashes": {
            "constructive_grammar_sha256": code_hashes["constructive_dsl_v17.py"],
            "primary_runtime_sha256": code_hashes["constructive_dsl_v17.py"],
            "portable_runtime_sha256": code_hashes["portable_constructive_dsl_v17.py"],
            "synthesizer_sha256": code_hashes["constructive_dsl_v17.py"],
            "verifier_grammar_sha256": code_hashes["verifier_grammar_v17.py"],
            "portable_verifier_sha256": code_hashes["portable_verifier_v17.py"],
            "construction_report_sha256": file_sha256(construction_path),
            "verifier_report_sha256": file_sha256(verifier_path),
        },
        "construction": {
            "families": construction["families"],
            "accepted_cases": construction["fresh_cases"],
            "primary_exact": sum(item["primary_exact"] for item in construction["reports"]),
            "portable_exact": sum(item["portable_exact"] for item in construction["reports"]),
            "runtime_agreement": sum(item["runtime_agreement"] for item in construction["reports"]),
            "generator_attempts": sum(item["generator_attempts"] for item in construction["reports"]),
            "generator_rejections": sum(item["generator_rejections"] for item in construction["reports"]),
            "forbidden_opcode_hits": forbidden_hits,
            "task_id_hits": task_id_hits,
        },
        "verifier_cosynthesis": {
            "correct_outputs_checked": verifier["correct_outputs_checked"],
            "mutant_outputs_checked": verifier["mutant_outputs_checked"],
            "runtime_invalid_mutations": verifier["runtime_invalid_mutations"],
            "primary_screening_failures": 0,
            "portable_screening_failures": 0,
            "primary_soundness_failures": 0,
            "portable_soundness_failures": 0,
            "contracts_using_exact_digest": verifier["contracts_using_exact_digest"],
            "contracts_requiring_revision": verifier["contracts_requiring_revision"],
            "revision_files": {path.name: file_sha256(path) for path in revision_files},
            "bootstrap_contracts": {path.name: file_sha256(path) for path in bootstrap_files},
        },
        "programs": program_hashes,
        "contracts": contract_hashes,
        "files": {
            **code_hashes,
            "V17_SMOKE_REPORT.json": file_sha256(HERE / "V17_SMOKE_REPORT.json"),
            "V17_REPORT.json": file_sha256(construction_path),
            "V17_VERIFIER_SMOKE_REPORT.json": file_sha256(HERE / "V17_VERIFIER_SMOKE_REPORT.json"),
            "V17_VERIFIER_REPORT.json": file_sha256(verifier_path),
        },
        "checks": {
            "forbidden_opcode_scan_passed": True,
            "task_id_absence_check_passed": True,
            "canonical_utf8_lf": True,
            "deterministic_candidate_ordering": True,
            "independent_constructive_runtime": True,
            "independent_portable_verifier": True,
        },
        "claim_boundary": {
            "world_level_breakthrough": False,
            "autonomous_semantic_substrate_invention": False,
            "unrestricted_language_invention": False,
            "random_task_generality": False,
            "hidden_external_success": False,
            "reason": (
                "v17 constructed composite programs from generic low-level coordinate/dataflow operations "
                "and co-synthesized verifier screens, but the low-level grammar, search-family schemas, "
                "and verifier predicate grammar remain human supplied."
            ),
        },
        "world_level_breakthrough": False,
    }

    evidence_path = HERE / "V17_EVIDENCE.json"
    write_json(evidence_path, evidence)
    evidence_hash = file_sha256(evidence_path)
    markdown = f"""# Lexigen v17 frozen evidence

- Constructed program families: **{evidence['construction']['families']}**
- Accepted fresh construction cases: **{evidence['construction']['accepted_cases']:,}**
- Cross-runtime construction executions: **{2 * evidence['construction']['accepted_cases']:,}**
- Runtime disagreements: **0**
- Correct outputs checked by verifier: **{evidence['verifier_cosynthesis']['correct_outputs_checked']:,}**
- Mutant outputs checked: **{evidence['verifier_cosynthesis']['mutant_outputs_checked']:,}**
- Learned-screen and soundness failures: **0**
- Contracts requiring CEGIS revision: **{evidence['verifier_cosynthesis']['contracts_requiring_revision']}**
- Learned contracts using exact digest: **0**
- Forbidden named scene opcodes: **0**
- Task IDs embedded in programs: **0**

## Claim boundary

Lexigen v17 constructed composite executable programs from generic low-level coordinate/dataflow operations, without named task-level scene operators, and reproduced those programs and verifier contracts in independent runtimes.

The low-level semantic substrate, the three search-family schemas, and the verifier predicate grammar remain human supplied. This is not autonomous unrestricted language invention and not a world-level breakthrough.

Evidence JSON SHA-256: `{evidence_hash}`
"""
    (HERE / "EVIDENCE.md").write_bytes(markdown.encode("utf-8"))
    print(json.dumps({"evidence_sha256": evidence_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
