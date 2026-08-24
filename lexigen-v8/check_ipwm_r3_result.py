from __future__ import annotations

import argparse
import json
from pathlib import Path

import ipwm_eval_r3 as r3

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "IPWM_COMPOSITIONAL_PROTOCOL_R3.json").read_text())
T = PROTOCOL["frozen_synthetic_r3_gates"]


def require(name: str, cond: bool, detail) -> None:
    if not cond:
        raise AssertionError(f"{name} failed: {detail}")


def check_feature_contract() -> None:
    rec = {
        "program_features": {"alloc": 0.8, "dispatch": 0.2, "memory": 0.6, "loop": 0.4},
        "intervention_features": {"alloc": 0.7, "dispatch": 0.1, "memory": 0.3, "loop": 0.5},
        "language": "python",
        "environment_id": "cpu-a",
        "intervention_family": "SHOULD_NOT_APPEAR",
        "repository_id": "SHOULD_NOT_APPEAR",
        "repository_family": "SHOULD_NOT_APPEAR",
    }
    feats = r3.aligned_features(rec)
    keys = set(feats)
    require("no intervention-family identity", not any("SHOULD_NOT_APPEAR" in k for k in keys), sorted(keys))
    require("no arbitrary cross keys", not any(k.startswith("x:") for k in keys), sorted(keys))
    for primitive in ("alloc", "dispatch", "memory", "loop"):
        require(f"aligned {primitive}", f"a:{primitive}" in keys, sorted(keys))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", type=Path, required=True)
    args = ap.parse_args()
    ev = json.loads(args.evaluation.read_text())
    check_feature_contract()
    require("claim boundary", ev["scientific_transfer_evidence"] is False, ev.get("scientific_transfer_evidence"))
    require("family identity flag", ev["intervention_family_identity_used_as_feature"] is False, ev.get("intervention_family_identity_used_as_feature"))

    repo = ev["repository_holdout"]
    fam = ev["repository_family_holdout"]
    lang = ev["language_holdout"]
    intfam = ev["intervention_family_holdout"]

    rf = repo["macro"]["full"]
    require("repo AUROC", rf["positive_speedup_auroc"] >= T["repository_holdout_full_positive_auroc_min"], rf)
    require("repo Spearman", rf["spearman_log_speedup"] >= T["repository_holdout_full_spearman_min"], rf)
    require("repo alignment AUROC delta", repo["deltas"]["full_minus_no_alignment_positive_auroc"] >= T["repository_holdout_full_minus_no_alignment_auroc_min"], repo["deltas"])
    require("repo alignment Spearman delta", repo["deltas"]["full_minus_no_alignment_spearman"] >= T["repository_holdout_full_minus_no_alignment_spearman_min"], repo["deltas"])
    require("repo null AUROC", repo["null_summary"]["global_max_positive_auroc"] <= T["repository_holdout_global_null_max_positive_auroc"], repo["null_summary"])
    require("repo null Spearman", repo["null_summary"]["global_max_abs_spearman"] <= T["repository_holdout_global_null_max_abs_spearman"], repo["null_summary"])
    require("repo validity null", repo["null_summary"]["global_max_validity_auroc"] <= T["repository_holdout_global_null_max_validity_auroc"], repo["null_summary"])

    ff = fam["macro"]["full"]
    require("family AUROC", ff["positive_speedup_auroc"] >= T["repository_family_holdout_full_positive_auroc_min"], ff)
    require("family Spearman", ff["spearman_log_speedup"] >= T["repository_family_holdout_full_spearman_min"], ff)

    lf = lang["macro"]["full"]
    require("language AUROC", lf["positive_speedup_auroc"] >= T["language_holdout_full_positive_auroc_min"], lf)
    require("language Spearman", lf["spearman_log_speedup"] >= T["language_holdout_full_spearman_min"], lf)

    inf = intfam["macro"]["full"]
    require("intervention-family AUROC", inf["positive_speedup_auroc"] >= T["intervention_family_holdout_full_positive_auroc_min"], inf)
    require("intervention-family Spearman", inf["spearman_log_speedup"] >= T["intervention_family_holdout_full_spearman_min"], inf)
    require("intervention alignment AUROC delta", intfam["deltas"]["full_minus_no_alignment_positive_auroc"] >= T["intervention_family_holdout_full_minus_no_alignment_auroc_min"], intfam["deltas"])
    require("intervention alignment Spearman delta", intfam["deltas"]["full_minus_no_alignment_spearman"] >= T["intervention_family_holdout_full_minus_no_alignment_spearman_min"], intfam["deltas"])
    require("intervention null AUROC delta", intfam["deltas"]["full_minus_stratified_null_positive_auroc"] >= T["intervention_family_holdout_full_minus_stratified_null_auroc_min"], intfam["deltas"])
    require("intervention null Spearman delta", intfam["deltas"]["full_minus_stratified_null_spearman"] >= T["intervention_family_holdout_full_minus_stratified_null_spearman_min"], intfam["deltas"])

    print(json.dumps({
        "synthetic_r3_passed": True,
        "scientific_transfer_evidence": False,
        "repository_full": rf,
        "repository_deltas": repo["deltas"],
        "repository_null": repo["null_summary"],
        "intervention_family_full": inf,
        "intervention_family_deltas": intfam["deltas"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
