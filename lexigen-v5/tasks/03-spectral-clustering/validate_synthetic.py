from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from candidates import CANDIDATES_BY_ARM, PROVENANCE, _reference_exact


def block_case(n: int, k: int, *, within: float, between: float, noise: float, seed: int, sparse: bool) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    latent = np.arange(n, dtype=int) % k
    rng.shuffle(latent)
    base = np.where(latent[:, None] == latent[None, :], within, between).astype(np.float64)
    perturb = rng.normal(0.0, noise, size=(n, n))
    perturb = (perturb + perturb.T) * 0.5
    similarity = np.clip(base + perturb, 0.0, 1.0)
    if sparse:
        similarity[similarity < 0.25] = 0.0
    np.fill_diagonal(similarity, 1.0)
    return {"similarity_matrix": similarity, "n_clusters": k}


def cases() -> list[tuple[str, dict[str, object]]]:
    return [
        ("dense_clear_60_k3", block_case(60, 3, within=0.90, between=0.08, noise=0.03, seed=11, sparse=False)),
        ("sparse_clear_80_k4", block_case(80, 4, within=0.85, between=0.03, noise=0.04, seed=23, sparse=True)),
        ("dense_weak_48_k3", block_case(48, 3, within=0.62, between=0.45, noise=0.05, seed=37, sparse=False)),
        ("disconnected_72_k4", block_case(72, 4, within=0.95, between=0.00, noise=0.02, seed=41, sparse=True)),
        ("single_cluster_40_k1", {**block_case(40, 2, within=0.80, between=0.10, noise=0.05, seed=53, sparse=False), "n_clusters": 1}),
        ("n_equals_k_4", {"similarity_matrix": np.eye(4, dtype=np.float64), "n_clusters": 4}),
    ]


def valid_shape(problem: dict[str, object], labels: np.ndarray) -> bool:
    n = int(np.asarray(problem["similarity_matrix"]).shape[0])
    k = int(problem["n_clusters"])
    if labels.shape != (n,) or not np.all(np.isfinite(labels)) or np.any(labels < 0):
        return False
    uniq = np.unique(labels.astype(int, copy=False))
    if n == 0:
        return True
    if k == 1:
        return uniq.size == 1
    if k >= n:
        return uniq.size == n
    return uniq.size == k


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frozen_cases = cases()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for case_index, (case_name, problem) in enumerate(frozen_cases, 1):
            start = time.perf_counter()
            reference = _reference_exact(problem)
            reference_s = time.perf_counter() - start
            reference_labels = np.asarray(reference["labels"], dtype=int)
            if not valid_shape(problem, reference_labels):
                raise RuntimeError(f"reference invalid on {case_name}")

            for arm, candidates in CANDIDATES_BY_ARM.items():
                for candidate_name, fn in candidates:
                    start = time.perf_counter()
                    error = None
                    try:
                        solution = fn(problem)
                        candidate_s = time.perf_counter() - start
                        labels = np.asarray(solution["labels"], dtype=int)
                        repeat = np.asarray(fn(problem)["labels"], dtype=int)
                        deterministic = bool(np.array_equal(labels, repeat))
                        shape_ok = valid_shape(problem, labels)
                        ari = float(adjusted_rand_score(reference_labels, labels)) if labels.size else 1.0
                        nmi = float(normalized_mutual_info_score(reference_labels, labels)) if labels.size else 1.0
                        quality_ok = bool(max(ari, nmi) >= 0.50)
                        valid = bool(shape_ok and quality_ok and deterministic)
                    except Exception as exc:
                        candidate_s = time.perf_counter() - start
                        deterministic = False
                        shape_ok = False
                        ari = -1.0
                        nmi = -1.0
                        quality_ok = False
                        valid = False
                        error = f"{type(exc).__name__}: {exc}"

                    row = {
                        "case_index": case_index,
                        "case": case_name,
                        "arm": arm,
                        "candidate": candidate_name,
                        "valid": valid,
                        "shape_ok": shape_ok,
                        "quality_ok": quality_ok,
                        "deterministic": deterministic,
                        "ari_vs_reference": ari,
                        "nmi_vs_reference": nmi,
                        "candidate_s": candidate_s,
                        "reference_s": reference_s,
                        "speedup_observation": reference_s / candidate_s if candidate_s > 0 else 0.0,
                        "error": error,
                        "official_training_access": False,
                        "official_test_access": False
                    }
                    rows.append(row)
                    if not valid:
                        failures.append(row)
                    print(f"{case_name} {arm}/{candidate_name} valid={valid} ari={ari:.3f} nmi={nmi:.3f}", flush=True)

    expected = 6 * 30
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} synthetic checks, got {len(rows)}")
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 3,
        "task": "spectral_clustering",
        "stage": "synthetic_r1",
        "synthetic_cases": 6,
        "candidate_count": 30,
        "checks": len(rows),
        "passed_checks": len(rows) - len(failures),
        "failed_checks": len(failures),
        "all_candidates_passed": not failures,
        "determinism_required": True,
        "quality_gate": "shape/exact-cluster-count + max(ARI,NMI)>=0.50 versus frozen source reference",
        "performance_is_not_a_synthetic_gate": True,
        "official_training_manifest_opened": False,
        "official_training_payloads_opened": 0,
        "official_test_manifest_opened": False,
        "official_test_payloads_opened": 0,
        "thresholds_changed": False,
        "human_task_specific_solver_design": False
    }
    (args.output / "synthetic-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "synthetic-results.jsonl").write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8"
    )
    (args.output / "candidate-provenance.json").write_text(json.dumps(PROVENANCE, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
