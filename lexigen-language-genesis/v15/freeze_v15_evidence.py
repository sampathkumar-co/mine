from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_V14 = "9b4ce5aa7f2600b04290fab272fc1922135007c8"
V13_EVIDENCE = "13cd271a2d4813001563842a51a1e72dd100aa1a"
ARCGEN = "a15cbdb44c776610aeeb9f487a06af875d3d0878"

REPORT_FILES = (
    "V15_INDUCTION_REPORT.json",
    "V15_PORTFOLIO_BENCHMARK.json",
    "V15_SEARCH_BENCHMARK.json",
    "V15_TRANSFER_REPORT.json",
    "V15_INDEPENDENT_REPORT.json",
    "V15_HELDOUT_MACRO_SCAN.json",
)
CODE_FILES = (
    "ir_runtime_v15.py",
    "portable_ir_runtime_v15.py",
    "compiler_v15.py",
    "macro_miner_v15.py",
    "induce_language_v15.py",
    "benchmark_portfolio_v15.py",
    "benchmark_search_v15.py",
    "validate_v15_transfer.py",
    "validate_v15_independent.py",
    "scan_macro_transfer_v15.py",
    "test_v15.py",
)


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    induction = load("V15_INDUCTION_REPORT.json")
    portfolio = load("V15_PORTFOLIO_BENCHMARK.json")
    transfer = load("V15_TRANSFER_REPORT.json")
    independent = load("V15_INDEPENDENT_REPORT.json")
    heldout = load("V15_HELDOUT_MACRO_SCAN.json")

    require(induction["source_programs"] == 9, "unexpected source-program count")
    require(induction["exact_macro_expansion"], "macro expansion is not exact")
    require(induction["semantic_replay_exact"], "demonstration replay failed")
    require(len(induction["macros"]) == 3, "unexpected induced macro count")
    require(not induction["human_supplied_macro_definitions"], "macro definitions were supplied")
    require(induction["human_supplied_v14_scene_atoms"], "claim boundary was weakened")

    require(transfer["total_cases"] == 9000, "small transfer total changed")
    require(transfer["ir_failures"] == 0, "primary IR transfer failed")
    require(transfer["portable_failures"] == 0, "portable v14 transfer failed")
    require(transfer["runtime_disagreements"] == 0, "small transfer runtimes disagree")

    require(independent["total_cases"] == 90000, "independent freeze total changed")
    require(independent["primary_failures"] == 0, "primary IR freeze failed")
    require(independent["portable_ir_failures"] == 0, "portable IR freeze failed")
    require(independent["portable_v14_failures"] == 0, "portable v14 freeze failed")
    require(independent["three_runtime_disagreements"] == 0, "three runtimes disagree")

    require(portfolio["baseline_total"] == 328, "baseline portfolio total changed")
    require(portfolio["portfolio_total"] == 216, "induced portfolio total changed")
    require(portfolio["aggregate_speedup"] > 1.5, "portfolio speedup threshold missed")
    require(portfolio["worst_slowdown"] <= 1.065, "portfolio slowdown boundary exceeded")

    require(heldout["checked_families"] == 898, "held-out denominator changed")
    require(heldout["import_or_validation_errors"] == 0, "held-out scan incomplete")
    require(heldout["match_count"] == 0, "unexpected held-out macro transfer")

    all_files = REPORT_FILES + CODE_FILES
    evidence = {
        "schema": "lexigen-v15-frozen-evidence-v1",
        "version": 15,
        "base_v14_commit": BASE_V14,
        "v13_visible_evidence_commit": V13_EVIDENCE,
        "arcgen_commit": ARCGEN,
        "induced_macros": len(induction["macros"]),
        "source_programs": induction["source_programs"],
        "saved_nodes_per_replay": induction["saved_nodes_per_full_replay"],
        "break_even_replays": induction["break_even_replays"],
        "dynamic_compression_ratio_at_10000_replays": induction["dynamic_compression_ratio_at_10000_replays"],
        "portfolio_aggregate_speedup": portfolio["aggregate_speedup"],
        "portfolio_worst_slowdown": portfolio["worst_slowdown"],
        "three_runtime_cases": independent["total_cases"],
        "three_runtime_failures": 0,
        "heldout_public_families": heldout["checked_families"],
        "heldout_macro_matches": heldout["match_count"],
        "claim_boundary": {
            "world_level_breakthrough": False,
            "autonomous_primitive_invention": False,
            "reason": "Macros were induced automatically, but every underlying v14 scene atom was human-authored after visible failures.",
        },
        "files": {name: sha256(HERE / name) for name in all_files},
    }
    evidence_path = HERE / "V15_EVIDENCE.json"
    evidence_path.write_bytes((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    evidence_hash = sha256(evidence_path)
    markdown = f"""# v15 frozen evidence

- Source programs: **{evidence['source_programs']}**
- Automatically induced executable macros: **{evidence['induced_macros']}**
- Three-runtime accepted cases: **{evidence['three_runtime_cases']:,}**
- Three-runtime failures/disagreements: **0**
- Portfolio aggregate candidate speedup: **{evidence['portfolio_aggregate_speedup']:.3f}x**
- Worst unrelated portfolio slowdown: **{evidence['portfolio_worst_slowdown']:.3f}x**
- Held-out public families checked: **{evidence['heldout_public_families']}**
- Held-out exact macro matches: **{evidence['heldout_macro_matches']}**

## Claim boundary

This is automatic macro induction over human-authored v14 scene atoms. It is not autonomous primitive invention and not a world-level breakthrough.

Evidence JSON SHA-256: `{evidence_hash}`
"""
    (HERE / "EVIDENCE.md").write_bytes(markdown.encode("utf-8"))
    print(json.dumps({"evidence_sha256": evidence_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
