from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "IPWM_CONTROL_HARDENING_R2.json").read_text())
    g = protocol["synthetic_r2_assertions_frozen_before_execution"]
    with tempfile.TemporaryDirectory() as td0:
        td = Path(td0)
        corpus = td / "synthetic-r2.jsonl"
        result = td / "evaluation-r2.json"
        subprocess.run([
            sys.executable, str(root / "generate_synthetic_ipwm_r1.py"),
            "--output", str(corpus), "--repos", "8", "--samples-per-repo", "10",
        ], check=True)
        subprocess.run([
            sys.executable, str(root / "ipwm_eval_r2.py"),
            "--input", str(corpus), "--output", str(result),
        ], check=True)
        ev = json.loads(result.read_text())
        repo = ev["repository_holdout"]
        fam = ev["repository_family_holdout"]
        lang = ev["language_holdout"]
        intv = ev["intervention_family_holdout"]

        assert repo["macro"]["full"]["positive_speedup_auroc"] >= g["repository_holdout_full_positive_auroc_min"], repo
        assert repo["macro"]["full"]["spearman_log_speedup"] >= g["repository_holdout_full_spearman_min"], repo
        assert repo["deltas"]["full_minus_no_cross_positive_auroc"] >= g["repository_holdout_full_minus_no_cross_auroc_min"], repo
        assert repo["deltas"]["full_minus_no_cross_spearman"] >= g["repository_holdout_full_minus_no_cross_spearman_min"], repo
        assert repo["null_summary"]["global_max_positive_auroc"] <= g["repository_holdout_global_null_max_positive_auroc"], repo
        assert repo["null_summary"]["global_max_abs_spearman"] <= g["repository_holdout_global_null_max_abs_spearman"], repo
        assert repo["deltas"]["full_minus_stratified_null_positive_auroc"] >= g["repository_holdout_full_minus_stratified_null_auroc_min"], repo
        assert repo["deltas"]["full_minus_stratified_null_spearman"] >= g["repository_holdout_full_minus_stratified_null_spearman_min"], repo

        assert fam["macro"]["full"]["positive_speedup_auroc"] >= g["family_holdout_full_positive_auroc_min"], fam
        assert fam["macro"]["full"]["spearman_log_speedup"] >= g["family_holdout_full_spearman_min"], fam
        assert intv is not None
        assert intv["macro"]["full"]["positive_speedup_auroc"] >= g["intervention_family_holdout_full_positive_auroc_min"], intv
        assert intv["macro"]["full"]["spearman_log_speedup"] >= g["intervention_family_holdout_full_spearman_min"], intv
        assert lang is not None
        assert lang["macro"]["full"]["positive_speedup_auroc"] >= g["language_holdout_full_positive_auroc_min"], lang

        print(json.dumps({
            "synthetic_control_hardening_r2_passed": True,
            "scientific_transfer_evidence": False,
            "repository_holdout": {
                "full": repo["macro"]["full"],
                "no_cross": repo["macro"]["no_cross"],
                "static_only": repo["macro"]["static_only"],
                "null_summary": repo["null_summary"],
                "deltas": repo["deltas"],
            },
            "family_holdout_full": fam["macro"]["full"],
            "language_holdout_full": lang["macro"]["full"],
            "intervention_family_holdout_full": intv["macro"]["full"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
