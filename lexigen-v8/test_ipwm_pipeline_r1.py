from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        corpus = td / "synthetic.jsonl"
        result = td / "result.json"
        subprocess.run([
            sys.executable, str(root / "generate_synthetic_ipwm_r1.py"),
            "--output", str(corpus), "--repos", "12", "--samples-per-repo", "18",
        ], check=True)
        subprocess.run([
            sys.executable, str(root / "ipwm_eval_r1.py"),
            "--input", str(corpus), "--output", str(result),
        ], check=True)
        r = json.loads(result.read_text())
        repo = r["repository_holdout"]
        fam = r["repository_family_holdout"]
        lang = r["language_holdout"]

        assert repo["group_count"] == 12
        assert repo["full"]["positive_speedup_auroc"] >= 0.70, repo
        assert repo["full"]["spearman_predicted_vs_observed_log_speedup"] >= 0.45, repo
        assert repo["full"]["positive_speedup_auroc"] >= repo["frequency"]["positive_speedup_auroc"] + 0.10, repo
        assert repo["full"]["spearman_predicted_vs_observed_log_speedup"] >= repo["frequency"]["spearman_predicted_vs_observed_log_speedup"] + 0.20, repo
        assert repo["full"]["positive_speedup_auroc"] >= repo["shuffled"]["positive_speedup_auroc"] + 0.12, repo
        assert repo["full"]["spearman_predicted_vs_observed_log_speedup"] >= repo["shuffled"]["spearman_predicted_vs_observed_log_speedup"] + 0.25, repo
        assert repo["relative_top_k_gain_over_frequency_baseline"] > 0.05, repo

        assert fam["full"]["positive_speedup_auroc"] >= 0.65, fam
        assert fam["full"]["spearman_predicted_vs_observed_log_speedup"] >= 0.35, fam
        assert lang is not None and lang["group_count"] == 2
        assert lang["full"]["positive_speedup_auroc"] >= 0.60, lang

        print(json.dumps({
            "synthetic_pipeline_passed": True,
            "scientific_transfer_evidence": False,
            "repository_holdout": repo["full"],
            "repository_family_holdout": fam["full"],
            "language_holdout": lang["full"],
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
