from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE_V19R3 = "baf4ca5"
CEGIS_CHECKPOINT = "ed3494b4851a6f5e35df7513e096ee2330bd68fd"
PRODUCTION_CHECKPOINT = "87f99380759815d97db0968582fcaadd6e51e950"
VERIFIER_CHECKPOINT = "20208d5f440008b177eaf020021d75a38efd9929"
FROZEN_V17 = "cd89382e38b45d12916e662af052a7aa1a374896"
ARCGEN = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
VISIBLE_EVIDENCE = "13cd271a2d4813001563842a51a1e72dd100aa1a"
PRODUCTION_SHA = "751e258d9dbdcdf2641503bacdc32f927fb2872fddaadbbbe348d668f5f25523"
PROGRAM_SHA = "c887750e98d2518c9d79d20a972753a1bc800f58a6168ffd10a02deb4fa48846"

FILES = (
    "v19r4/README.md", "v19r4/V19R4_PRECOMMIT.json",
    "v19r4/V19R4_CEGIS_REPORT.json", "v19r4/V19R4_INTERMEDIATE_REPORT.json",
    "v19r4/V19R4_REPORT.json", "v19r4/V19R4_VERIFIER_SMOKE_REPORT.json",
    "v19r4/V19R4_VERIFIER_REPORT.json", "v19r4/V19R4_INTEGRITY_AUDIT.json",
    "v19r4/run_cegis_v19r4.py", "v19r4/validate_v19r4.py",
    "v19r4/validate_verifier_v19r4.py", "v19r4/test_v19r4.py",
    "v19r4/freeze_v19r4_evidence.py",
    "v19r4/production/v19r4-production.json",
    "v19r4/production/v19r4-arguments.json",
    "v19r4/production/v19r4-concrete-program.json",
    "v19r4/contracts/gate-10-contract.json",
    "v19r4/contracts/revisions/gate-10-contract-r0.json",
    "v19r4/contracts/revisions/gate-10-false-accepts-r0.json",
    "v19r3/enumerate_v19r3.py", "v19r3/V19R3_ATTEMPT_REPORT.json",
    "v19r3/V19R3_EVIDENCE.json", "v19r2/runtime_v19r2.py",
    "v19r2/portable_runtime_v19r2.py", "v19r2/mutations_v19r2.py",
    "v19r2/verifier_grammar_v19r2.py", "v19r2/portable_verifier_v19r2.py",
    "v19r2/cosynthesize_verifier_v19r2.py",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    pre = load(HERE / "V19R4_PRECOMMIT.json")
    cegis = load(HERE / "V19R4_CEGIS_REPORT.json")
    report = load(HERE / "V19R4_REPORT.json")
    verifier = load(HERE / "V19R4_VERIFIER_REPORT.json")
    contract = load(HERE / "contracts" / "gate-10-contract.json")
    audit = load(HERE / "V19R4_INTEGRITY_AUDIT.json")
    v19r3 = load(ROOT / "v19r3" / "V19R3_ATTEMPT_REPORT.json")

    require(pre["base_v19r3_commit"] == BASE_V19R3, "v19r3 base changed")
    require(pre["candidate_compositions_per_gate"] == 250000, "candidate budget changed")
    require(pre["fixed_development_gates"] == [7, 10, 13], "fixed pool changed")
    require(not pre["cegis"]["human_survivor_selection_allowed"], "human selection enabled")
    require(v19r3["unique_productions"] == 0, "v19r3 ambiguity history changed")
    require(v19r3["ambiguous_program_sets"] == 1, "v19r3 ambiguity count changed")
    require(v19r3["no_program_failures"] == 2, "v19r3 negative denominator changed")

    require(cegis["status"] == "unique_survivor", "CEGIS did not finish uniquely")
    require(cegis["initial_survivors"] == 5, "initial survivor count changed")
    require(cegis["accepted_refinement_cases"] == 7, "CEGIS denominator changed")
    require(cegis["attempted_seeds"] == 7 and cegis["generator_rejections"] == 0, "CEGIS seed history changed")
    require(cegis["final_survivors"] == 1, "final survivor count changed")
    require(cegis["selected_descriptor_removal_survivors"] == 0, "removal ablation failed")
    require(cegis["selected_descriptor"]["actions"] == ["reflect_left", "reflect_right", "reflect_top", "reflect_bottom"], "selected composition changed")
    require(cegis["production_sha256"] == PRODUCTION_SHA, "production identity changed")
    require(cegis["selected_concrete_program_sha256"] == PROGRAM_SHA, "program identity changed")

    require(report["production_sha256"] == PRODUCTION_SHA, "validation production mismatch")
    require(report["concrete_program_sha256"] == PROGRAM_SHA, "validation program mismatch")
    require(report["accepted_fresh_cases"] == 10000, "fresh denominator changed")
    require(report["fresh_primary_exact"] == report["fresh_portable_exact"] == report["fresh_runtime_agreement"] == 10000, "fresh gate mismatch")
    require(report["generator_rejections"] == 0, "fresh generator rejections changed")
    require(report["fixed_pool_negative_gates"] == [7, 13], "negative gates changed")
    require(report["frozen_v17_ablation_failed"], "v17 ablation changed")
    require(not report["task_id_hits"] and not report["hidden_outputs_opened"], "leakage flag changed")

    require(verifier["production_sha256"] == PRODUCTION_SHA, "verifier production mismatch")
    require(verifier["concrete_program_sha256"] == PROGRAM_SHA, "verifier program mismatch")
    require(verifier["accepted_cases"] == 1000, "verifier correct denominator changed")
    require(verifier["mutant_cases"] == 8000, "verifier mutant denominator changed")
    for key in ("screening_rejected_primary", "screening_rejected_portable", "soundness_rejected_primary", "soundness_rejected_portable"):
        require(verifier[key] == 8000, key + " changed")
    require(not verifier["exact_digest_used_by_learned_screen"], "learned exact digest used")
    require(contract["production_sha256"] == PRODUCTION_SHA and contract["concrete_program_sha256"] == PROGRAM_SHA, "contract binding changed")
    require(not contract["exact_digest_used"], "contract exact digest flag changed")
    require(all(item["name"] != "exact_digest" for item in contract["predicates"]), "exact digest in learned predicates")

    require(audit["audit_verdict"] == "credible_finite_meta_grammar_composition_invention_candidate", "audit verdict weakened")
    limits = audit["claim_limits"]
    require(limits["human_supplied_finite_meta_grammar"], "human bias hidden")
    require(not any(limits[key] for key in ("unrestricted_semantic_substrate_invention", "transfer_demonstrated", "sealed_external_success", "external_discovery", "world_level_breakthrough")), "claim boundary weakened")

    evidence = {
        "schema": "lexigen-v19r4-frozen-evidence-v1",
        "version": "19r4",
        "base_v19r3_commit": BASE_V19R3,
        "cegis_checkpoint_commit": CEGIS_CHECKPOINT,
        "production_checkpoint_commit": PRODUCTION_CHECKPOINT,
        "verifier_checkpoint_commit": VERIFIER_CHECKPOINT,
        "frozen_v17_commit": FROZEN_V17,
        "arcgen_commit": ARCGEN,
        "visible_evidence_commit": VISIBLE_EVIDENCE,
        "fixed_development_gates": [7, 10, 13],
        "fixed_no_program_gates": [7, 13],
        "candidate_compositions_per_gate": 250000,
        "initial_demonstration_survivors": 5,
        "cegis_accepted_cases": 7,
        "cegis_generator_rejections": 0,
        "final_survivors": 1,
        "selected_descriptor_removal_survivors": 0,
        "selected_actions": ["reflect_left", "reflect_right", "reflect_top", "reflect_bottom"],
        "production_sha256": PRODUCTION_SHA,
        "concrete_program_sha256": PROGRAM_SHA,
        "arguments_sha256": report["arguments_sha256"],
        "fresh_cases": 10000,
        "fresh_primary_exact": 10000,
        "fresh_portable_exact": 10000,
        "fresh_runtime_agreement": 10000,
        "generator_rejections": 0,
        "frozen_v17_ablation_failed": True,
        "verifier": {
            "correct_outputs": 1000,
            "mutant_outputs": 8000,
            "primary_screening_failures": 0,
            "portable_screening_failures": 0,
            "primary_soundness_failures": 0,
            "portable_soundness_failures": 0,
            "learned_exact_digest": False,
            "contract_sha256": verifier["contract_sha256"],
            "predicates": contract["predicates"],
            "contract_revision": contract["revision"],
            "training_runtime_invalid_mutations": contract["training_runtime_invalid_mutations"],
            "fresh_runtime_invalid_mutations": verifier["fresh_runtime_invalid_mutations"],
        },
        "files": {name: sha(ROOT / name) for name in FILES},
        "claim_boundary": {
            "complete_composition_selected_within_frozen_finite_meta_grammar": True,
            "human_supplied_finite_meta_grammar": True,
            "unrestricted_semantic_substrate_invention": False,
            "transfer_demonstrated": False,
            "sealed_external_success": False,
            "external_discovery": False,
            "world_level_breakthrough": False,
        },
    }
    output = HERE / "V19R4_EVIDENCE.json"
    output.write_bytes((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    digest = sha(output)
    markdown = (
        "# v19r4 frozen evidence\n\n"
        "- Fixed development gates: **3**\n"
        "- Complete candidate compositions per gate: **250,000**\n"
        "- Preserved no-program gates: **2** (7 and 13)\n"
        "- Gate 10 demonstration survivors: **5 → 1** after **7** preregistered CEGIS cases\n"
        "- Selected-descriptor removal survivors: **0**\n"
        "- Disjoint fresh validation: **10,000 / 10,000 exact in both runtimes**\n"
        "- Verifier correct outputs: **1,000 / 1,000 accepted in both implementations**\n"
        "- Mutants: **8,000 / 8,000 rejected by both learned screens and both soundness anchors**\n"
        "- Learned exact-digest predicate: **no**\n\n"
        "## Claim boundary\n\n"
        "v19r4 is a credible complete-composition invention candidate within a frozen, human-supplied finite meta-grammar. It does not establish unrestricted semantic-substrate invention, transfer to another task, sealed external success, external discovery, or a world-level breakthrough.\n\n"
        f"Evidence JSON SHA-256: `{digest}`\n"
    )
    (HERE / "EVIDENCE.md").write_bytes(markdown.encode("utf-8"))
    print(json.dumps({"evidence_sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
