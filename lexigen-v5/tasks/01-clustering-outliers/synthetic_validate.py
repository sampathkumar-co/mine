from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import hdbscan
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score

from candidates import CANDIDATES_BY_ARM, PROVENANCE, Problem, Solution

CASES = (
    ("small_n80", 80, 11),
    ("small_n150", 150, 23),
    ("medium_n320", 320, 37),
    ("medium_n640", 640, 41),
    ("large_n1000", 1000, 53),
    ("large_n1600", 1600, 67),
)


def generate_problem(n: int, random_seed: int) -> Problem:
    np.random.seed(random_seed)
    random.seed(random_seed)
    n = max(n, 2)
    dim = max(2, min(10, n // 50))
    n_clusters = max(2, min(8, n // 30))
    outlier_ratio = 0.1
    if (n - int(n * outlier_ratio)) < n_clusters:
        n_clusters = max(1, n - int(n * outlier_ratio))
    points_per_cluster = (n - int(n * outlier_ratio)) // n_clusters
    dataset: list[list[float]] = []
    cluster_centers = np.random.randn(n_clusters, dim) * 10
    for i in range(n_clusters):
        current = max(1, points_per_cluster)
        cluster_points = np.random.randn(current, dim) * 1.5 + cluster_centers[i]
        dataset.extend(cluster_points.tolist())
    n_outliers = int(n * outlier_ratio)
    current_cluster_points = len(dataset)
    intended_cluster_points = points_per_cluster * n_clusters
    if current_cluster_points < intended_cluster_points:
        n_outliers = max(0, n - current_cluster_points)
    if n_outliers > 0:
        dataset.extend((np.random.randn(n_outliers, dim) * (15 + dim)).tolist())
    if len(dataset) < n:
        needed = n - len(dataset)
        dataset.extend((np.random.randn(needed, dim) * (15 + dim) * 1.1).tolist())
    elif len(dataset) > n:
        np.random.shuffle(dataset)
        dataset = dataset[:n]
    np.random.shuffle(dataset)
    min_cluster_size = max(5, n // 20)
    min_samples = max(3, min_cluster_size // 2)
    return {"dataset": dataset, "min_cluster_size": min_cluster_size, "min_samples": min_samples}


def reference(problem: Problem) -> Solution:
    dataset = np.array(problem["dataset"])
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(problem.get("min_cluster_size", 5)),
        min_samples=int(problem.get("min_samples", 3)),
    )
    clusterer.fit(dataset)
    labels = clusterer.labels_
    return {
        "labels": labels.tolist(),
        "probabilities": clusterer.probabilities_.tolist(),
        "cluster_persistence": clusterer.cluster_persistence_.tolist(),
        "num_clusters": len(set(labels[labels != -1])),
        "num_noise_points": int(np.sum(labels == -1)),
    }


def verify(problem: Problem, solution: Solution, expected: Solution) -> tuple[bool, str | None, dict[str, float]]:
    required = ["labels", "probabilities", "cluster_persistence"]
    if not isinstance(solution, dict) or not all(key in solution for key in required):
        return False, "missing_required_key", {}
    try:
        labels = np.array(solution["labels"])
        probabilities = np.array(solution["probabilities"], dtype=np.float64)
    except Exception:
        return False, "decode_failure", {}
    dataset = np.array(problem["dataset"])
    if len(labels) != len(dataset):
        return False, "labels_length", {}
    if not np.all(np.logical_or(labels == -1, labels >= 0)):
        return False, "invalid_labels", {}
    if np.any(np.isnan(probabilities)) or np.any(np.isinf(probabilities)):
        return False, "nonfinite_probabilities", {}
    if np.any(probabilities < 0) or np.any(probabilities > 1):
        return False, "probability_range", {}
    ref_labels = np.array(expected["labels"])
    num_clusters = len(set(labels[labels != -1]))
    ref_num_clusters = len(set(ref_labels[ref_labels != -1]))
    cluster_deviation = abs(num_clusters - ref_num_clusters) / max(1, ref_num_clusters)
    if cluster_deviation > 0.3:
        return False, "cluster_count_deviation", {"cluster_deviation": float(cluster_deviation)}
    noise_ratio = float(np.sum(labels == -1) / len(labels))
    ref_noise_ratio = float(np.sum(ref_labels == -1) / len(ref_labels))
    noise_deviation = abs(noise_ratio - ref_noise_ratio)
    if noise_deviation > 0.2:
        return False, "noise_deviation", {"noise_deviation": float(noise_deviation)}
    ari = 1.0
    if num_clusters > 0 and ref_num_clusters > 0:
        ari = float(adjusted_rand_score(ref_labels, labels))
        if ari < 0.5:
            return False, "ari", {"ari": ari, "cluster_deviation": float(cluster_deviation), "noise_deviation": float(noise_deviation)}
    return True, None, {"ari": ari, "cluster_deviation": float(cluster_deviation), "noise_deviation": float(noise_deviation)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flat = [(arm, name, fn) for arm, rows in CANDIDATES_BY_ARM.items() for name, fn in rows]
    if len(flat) != 30:
        raise RuntimeError(f"expected 30 frozen candidates, got {len(flat)}")
    if set(CANDIDATES_BY_ARM) != {"v5_full","v5_no_transfer","random_search","static_template","v4_compatible"}:
        raise RuntimeError("unexpected comparison arms")

    results: list[dict[str, object]] = []
    for case_name, n, seed in CASES:
        problem = generate_problem(n, seed)
        expected = reference(problem)
        for arm, name, fn in flat:
            try:
                proposed = fn(problem)
                valid, reason, metrics = verify(problem, proposed, expected)
                error = None
            except Exception as exc:
                valid, reason, metrics = False, "exception", {}
                error = f"{type(exc).__name__}: {exc}"
            provenance = next(row for row in PROVENANCE[arm] if row["candidate"] == name)
            row = {
                "case": case_name,
                "n": n,
                "seed": seed,
                "arm": arm,
                "candidate": name,
                "valid": valid,
                "failure_reason": reason,
                "exception": error,
                "metrics": metrics,
                "transfer_ids": provenance["transfer_ids"],
                "learned_template": provenance["learned_template"],
                "semantic_signature": provenance["semantic_signature"],
            }
            results.append(row)
            print(f"{case_name} {arm}/{name} valid={valid} reason={reason} metrics={metrics}", flush=True)

    eligibility: dict[str, list[str]] = {}
    candidate_summaries: list[dict[str, object]] = []
    for arm, rows in CANDIDATES_BY_ARM.items():
        eligibility[arm] = []
        for name, _ in rows:
            selected = [r for r in results if r["arm"] == arm and r["candidate"] == name]
            passed = sum(1 for r in selected if r["valid"])
            eligible = passed == len(CASES)
            if eligible:
                eligibility[arm].append(name)
            provenance = next(row for row in PROVENANCE[arm] if row["candidate"] == name)
            candidate_summaries.append({
                "arm": arm,
                "candidate": name,
                "checks_passed": passed,
                "checks_total": len(CASES),
                "eligible_for_official_training": eligible,
                "transfer_ids": provenance["transfer_ids"],
                "learned_template": provenance["learned_template"],
            })

    v5_has_candidate = bool(eligibility["v5_full"])
    every_arm_has_candidate = all(bool(eligibility[arm]) for arm in eligibility)
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 1,
        "task": "clustering_outliers",
        "stage": "synthetic_correctness_r1",
        "synthetic_cases": len(CASES),
        "candidate_count": len(flat),
        "checks": len(results),
        "passed_checks": sum(1 for row in results if row["valid"]),
        "failed_checks": sum(1 for row in results if not row["valid"]),
        "candidate_summaries": candidate_summaries,
        "eligible_candidates_by_arm": eligibility,
        "v5_full_has_training_eligible_candidate": v5_has_candidate,
        "every_arm_has_training_eligible_candidate": every_arm_has_candidate,
        "task_may_proceed_to_official_training": v5_has_candidate,
        "candidate_repair_permitted_after_synthetic": False,
        "official_training_manifest_opened": False,
        "official_training_payloads_opened": 0,
        "official_test_manifest_opened": False,
        "official_test_payloads_opened": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "synthetic-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "synthetic-results.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in results) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_candidates_by_arm": eligibility, "failed_checks": report["failed_checks"], "task_may_proceed_to_official_training": v5_has_candidate}, indent=2), flush=True)
    if not v5_has_candidate:
        raise SystemExit("Task 1 has no v5_full candidate surviving the frozen synthetic gate")


if __name__ == "__main__":
    main()
