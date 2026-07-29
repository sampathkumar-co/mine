from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_V15 = "4a8d1bf303951b1c4de704eada76bb95ffc89b77"
ARCGEN = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
V13_EVIDENCE = "13cd271a2d4813001563842a51a1e72dd100aa1a"
CODE_FILES = (
    "mutations_v16.py",
    "verifier_grammar_v16.py",
    "portable_verifier_v16.py",
    "cosynthesize_verifier_v16.py",
    "validate_v16.py",
    "test_v16.py",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    report_path = HERE / "V16_REPORT.json"
    report = load(report_path)
    require(report["families"] == 9, "unexpected v16 family count")
    require(report["correct_outputs_checked"] == 9000, "correct-output denominator changed")
    require(report["mutant_outputs_checked"] == 69_841, "mutation denominator changed")
    require(
        report["screening_rejections"] == report["mutant_outputs_checked"],
        "primary screening admitted mutations",
    )
    require(
        report["portable_screening_rejections"] == report["mutant_outputs_checked"],
        "portable screening admitted mutations",
    )
    require(
        report["soundness_rejections"] == report["mutant_outputs_checked"],
        "primary soundness anchor admitted mutations",
    )
    require(
        report["portable_soundness_rejections"] == report["mutant_outputs_checked"],
        "portable soundness anchor admitted mutations",
    )
    require(report["contracts_using_exact_digest"] == 0, "screening fell back to exact digest")
    require(report["contracts_requiring_revision"] >= 1, "CEGIS did not exercise refinement")
    require(not report["target_used_by_verifier"], "target leakage flag changed")
    require(not report["world_level_breakthrough"], "claim boundary was weakened")

    contract_files = sorted((HERE / "contracts").glob("v16-contract-*.json"))
    require(len(contract_files) == 9, "unexpected contract count")
    contracts = [load(path) for path in contract_files]
    require(
        all(contract["soundness_anchor"] == {"name": "exact_digest", "mandatory": True} for contract in contracts),
        "soundness anchor missing",
    )
    require(
        all(not contract["exact_digest_used"] for contract in contracts),
        "screening contract used exact digest",
    )

    evidence = {
        "schema": "lexigen-v16-frozen-evidence-v1",
        "version": 16,
        "base_v15_commit": BASE_V15,
        "arcgen_commit": ARCGEN,
        "v13_visible_evidence_commit": V13_EVIDENCE,
        "families": report["families"],
        "correct_outputs_checked": report["correct_outputs_checked"],
        "mutant_outputs_checked": report["mutant_outputs_checked"],
        "contracts_requiring_revision": report["contracts_requiring_revision"],
        "screening_exact_digest_contracts": report["contracts_using_exact_digest"],
        "primary_screening_failures": 0,
        "portable_screening_failures": 0,
        "primary_soundness_failures": 0,
        "portable_soundness_failures": 0,
        "target_used_by_verifier": False,
        "contracts": {
            path.name: {
                "sha256": sha256(path),
                "contract_sha256": contract["contract_sha256"],
                "predicates": contract["predicates"],
                "revision": contract["revision"],
            }
            for path, contract in zip(contract_files, contracts)
        },
        "files": {
            "V16_REPORT.json": sha256(report_path),
            **{name: sha256(HERE / name) for name in CODE_FILES},
        },
        "claim_boundary": {
            "world_level_breakthrough": False,
            "autonomous_primitive_invention": False,
            "reason": (
                "Verifier contracts were co-synthesized automatically, but both the scene IR "
                "and verifier predicate grammar remain human supplied."
            ),
        },
    }
    evidence_path = HERE / "V16_EVIDENCE.json"
    evidence_path.write_bytes(
        (json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    evidence_hash = sha256(evidence_path)
    markdown = f"""# v16 frozen evidence

- Programs/contracts: **{evidence['families']}**
- Correct fresh outputs checked: **{evidence['correct_outputs_checked']:,}**
- Mutant outputs checked: **{evidence['mutant_outputs_checked']:,}**
- Screening and soundness failures: **0**
- Contracts revised by counterexamples: **{evidence['contracts_requiring_revision']}**
- Screening contracts using exact digest: **0**
- Hidden target used by verifier: **no**

## Claim boundary

This is verifier co-synthesis over human-authored scene and verifier atoms. It is not autonomous primitive invention and not a world-level breakthrough.

Evidence JSON SHA-256: `{evidence_hash}`
"""
    (HERE / "EVIDENCE.md").write_bytes(markdown.encode("utf-8"))
    print(json.dumps({"evidence_sha256": evidence_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
