from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from engine_v2 import ENGINE_VERSION, fingerprint, generate_proposals
from selector import MIN_FAMILIES, TASK_COUNT, classify, select_tasks, task_score

SYNTHETIC_TASKS = {
    "byte_exact": '''
import hmac
from cryptography.hazmat.primitives import hashes

def solve(problem):
    digest = hashes.Hash(hashes.SHA256())
    digest.update(problem["payload"])
    return {"digest": digest.finalize()}

def is_solution(problem, solution):
    return hmac.compare_digest(solve(problem)["digest"], solution["digest"])
''',
    "matrix_tolerance": '''
import numpy as np

def solve(problem):
    u, s, vt = np.linalg.svd(problem["matrix"], full_matrices=False)
    return {"result": u @ vt}

def is_solution(problem, solution):
    return np.allclose(solve(problem)["result"], solution["result"], atol=1e-5)
''',
    "graph_boolean": '''
def solve(problem):
    graph = problem["graph"]
    seen = set([0])
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return {"value": len(seen)}

def is_solution(problem, solution):
    return solution["value"] == solve(problem)["value"]
''',
    "grouped_numeric": '''
import numpy as np

def generate(n):
    blocks = np.stack([np.ones((20, 10)) * index for index in range(n)])
    return {"X": blocks.reshape(n * 20, 10), "k": n}

def solve(problem):
    centers = problem["X"].reshape(problem["k"], 20, 10).mean(axis=1)
    return {"labels": np.repeat(np.arange(problem["k"]), 20).tolist(), "centers": centers.tolist()}
''',
    "convex_projection": '''
import numpy as np

def project_constraints(problem):
    x = np.asarray(problem["x"], dtype=float)
    dual = float(problem["threshold"])
    for _ in range(100):
        x = np.maximum(x - dual, 0.0)
        if abs(float(x.sum()) - 1.0) < 1e-7:
            break
    return {"x": x.tolist()}

def is_solution(problem, solution):
    return abs(sum(solution["x"]) - 1.0) < 1e-5
''',
    "statistical_quantile": '''
import numpy as np

def solve(problem):
    values = np.asarray(problem["values"])
    return {"value": float(np.quantile(values, 0.95))}
''',
}

EXPECTED_PRESENT = {
    "byte_exact": {"bytes", "crypto", "hash", "bit_exact"},
    "matrix_tolerance": {"matrix", "decomposition", "tolerance", "approximate_verifier"},
    "graph_boolean": {"graph", "set", "iterative", "verifier"},
    "grouped_numeric": {"array", "cluster", "grouped_generator", "block_structure"},
    "convex_projection": {"array", "projection", "constraints", "convex", "iterative"},
    "statistical_quantile": {"array", "statistics", "order_statistic"},
}

EXPECTED_ABSENT = {
    "byte_exact": {"matrix", "grouped_generator"},
    "matrix_tolerance": {"hash", "crypto", "graph"},
    "graph_boolean": {"matrix", "hash", "crypto", "decomposition"},
    "grouped_numeric": {"hash", "crypto", "sequence"},
    "convex_projection": {"hash", "crypto", "graph"},
    "statistical_quantile": {"hash", "crypto", "graph"},
}

REQUIRED_FULL_OPERATORS = {
    "byte_exact": {"native_one_shot_backend", "zero_copy_representation"},
    "matrix_tolerance": {"mixed_precision_with_local_refinement", "risk_aware_staging"},
    "graph_boolean": {"bit_parallel_representation", "sparse_frontier_search"},
    "grouped_numeric": {"structure_aware_initialization", "bounded_exact_refinement"},
    "convex_projection": {"active_set_decomposition", "risk_aware_staging"},
    "statistical_quantile": {"sort_partition_reduction", "vectorized_batch_kernel"},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    report: dict[str, object] = {"engine_version": ENGINE_VERSION, "tasks": {}}
    all_ids: set[str] = set()

    for label, source in SYNTHETIC_TASKS.items():
        fp = fingerprint(source, source)
        require(fp == fingerprint(source, source), f"fingerprint is not deterministic for {label}")
        features = set(fp.features)
        require(EXPECTED_PRESENT[label].issubset(features), f"missing features for {label}: {EXPECTED_PRESENT[label] - features}")
        require(not features.intersection(EXPECTED_ABSENT[label]), f"false-positive features for {label}: {features & EXPECTED_ABSENT[label]}")

        arms: dict[str, list[dict[str, object]]] = {}
        for arm in ("v4_full", "v4_no_transfer", "random_search", "template_synthesis", "v3_compatible"):
            proposals = generate_proposals(fp, arm=arm, limit=6, random_seed="validation-r2")
            repeated = generate_proposals(fp, arm=arm, limit=6, random_seed="validation-r2")
            require(proposals == repeated, f"{arm} is not deterministic for {label}")
            if arm != "v3_compatible":
                require(bool(proposals), f"{arm} generated no proposal for {label}")
            ids = [proposal.proposal_id for proposal in proposals]
            require(len(ids) == len(set(ids)), f"duplicate proposal IDs for {label}/{arm}")
            all_ids.update(ids)
            arms[arm] = [asdict(proposal) for proposal in proposals]

        full_operators = {operator for proposal in arms["v4_full"] for operator in proposal["operators"]}
        require(REQUIRED_FULL_OPERATORS[label].issubset(full_operators), f"v4 omitted expected mechanisms for {label}: {REQUIRED_FULL_OPERATORS[label] - full_operators}")
        require(all(len(proposal["operators"]) == 1 for proposal in arms["template_synthesis"]), "template baseline composed operators")
        report["tasks"][label] = {"fingerprint": asdict(fp), "arms": arms}

    selector_rows = [
        {"task": name, "family": classify(name), "score": task_score(name)}
        for name in (
            "graph_shortest_path",
            "matrix_inverse",
            "fft_filter",
            "quantile_statistics",
            "string_alignment",
            "convex_hull_geometry",
            "portfolio_optimization",
            "subset_knapsack",
            "cipher_stream",
            "graph_tree",
            "matrix_qr",
        )
    ]
    selected = select_tasks(selector_rows)
    require(len(selected) == TASK_COUNT, "selector returned the wrong task count")
    require(len({row["family"] for row in selected}) >= MIN_FAMILIES, "selector failed diversity gate")

    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["validation_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    report["unique_proposal_ids"] = len(all_ids)
    report["selected_synthetic_inventory"] = selected
    report["holdout_inventory_accessed"] = False
    report["task_contents_accessed"] = False
    output = Path("validation-evidence-r2")
    output.mkdir(parents=True, exist_ok=True)
    (output / "engine-validation-r2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"engine_version": ENGINE_VERSION, "synthetic_tasks": len(SYNTHETIC_TASKS), "unique_proposal_ids": len(all_ids), "validation_sha256": report["validation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
