from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = json.loads((ROOT / "IPWM_CONTROL_HARDENING_R2.json").read_text())
T = PROTOCOL["synthetic_r2_assertions_frozen_before_execution"]


def check(name: str, cond: bool, detail: str) -> None:
    if not cond:
        raise AssertionError(f"{name} failed: {detail}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        corpus = td / "synthetic.jsonl"
        result = td / "evaluation-r2.json"
        subprocess.run([
            sys.executable, str(ROOT / "generate_synthetic_ipwm_r1.py"),
            "--output", str(corpus), "--repos", "12", "--samples-per-repo", "18",
        ], check=True)
        subprocess.run([
            sys.executable, str(ROOT / "ipwm_eval_r2.py"),
            "--input", str(corpus), "--output", str(result),
        ], check=True)
        ev = json.loads(result.read_text())

    repo = ev["repository_holdout"]
    fam = ev["repository_family_holdout"]
    intfam = ev["intervention_family_holdout"]
    lang = ev["language_holdout"]

    rf = repo["macro"]["full"]
    check("repo full AUROC", rf["positive_speedup_auroc"] >= T["repository_holdout_full_positive_auroc_min"], str(rf))
    check("repo full Spearman", rf["spearman_log_speedup"] >= T["repository_holdout_full_spearman_min"], str(rf))
    check("repo full-no-cross AUROC", repo["deltas"]["full_minus_no_cross_positive_auroc"] >= T["repository_holdout_full_minus_no_cross_auroc_min"], str(repo["deltas"]))
    check("repo full-no-cross Spearman", repo["deltas"]["full_minus_no_cross_spearman"] >= T["repository_holdout_full_minus_no_cross_spearman_min"], str(repo["deltas"]))
    check("repo global-null AUROC", repo["null_summary"]["global_max_positive_auroc"] <= T["repository_holdout_global_null_max_positive_auroc"], str(repo["null_summary"]))
    check("repo global-null Spearman", repo["null_summary"]["global_max_abs_spearman"] <= T["repository_holdout_global_null_max_abs_spearman"], str(repo["null_summary"]))
    check("repo full-stratified AUROC", repo["deltas"]["full_minus_stratified_null_positive_auroc"] >= T["repository_holdout_full_minus_stratified_null_auroc_min"], str(repo["deltas"]))
    check("repo full-stratified Spearman", repo["deltas"]["full_minus_stratified_null_spearman"] >= T["repository_holdout_full_minus_stratified_null_spearman_min"], str(repo["deltas"]))

    ff = fam["macro"]["full"]
    check("family full AUROC", ff["positive_speedup_auroc"] >= T["family_holdout_full_positive_auroc_min"], str(ff))
    check("family full Spearman", ff["spearman_log_speedup"] >= T["family_holdout_full_spearman_min"], str(ff))

    inf = intfam["macro"]["full"]
    check("intervention-family full AUROC", inf["positive_speedup_auroc"] >= T["intervention_family_holdout_full_positive_auroc_min"], str(inf))
    check("intervention-family full Spearman", inf["spearman_log_speedup"] >= T["intervention_family_holdout_full_spearman_min"], str(inf))

    lf = lang["macro"]["full"]
    check("language full AUROC", lf["positive_speedup_auroc"] >= T["language_holdout_full_positive_auroc_min"], str(lf))

    check("claim boundary", ev["scientific_transfer_evidence"] is False, "synthetic R2 must not count as transfer evidence")
    print(json.dumps({
        "synthetic_r2_passed": True,
        "scientific_transfer_evidence": False,
        "repository_macro_full": rf,
        "repository_deltas": repo["deltas"],
        "repository_null_summary": repo["null_summary"],
        "family_macro_full": ff,
        "intervention_family_macro_full": inf,
        "language_macro_full": lf,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
