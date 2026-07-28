from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from engine import ENGINE_VERSION, fingerprint, generate_proposals
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
    return {"labels": np.repeat(np.arange(problem["k"]), 20).tolist()}
''',
    "convex_projection": '''
import numpy as np

def solve(problem):
    x = np.asarray(problem["x"], dtype=float)
    threshold = float(problem["threshold"])
    for _ in range(100):
        x = np.maximum(x - threshold, 0.0)
        if abs(float(x.sum()) - 1.0) < 1e-7:
            break
    return {"x": x.tolist()}
''',
    "statistical_quantile": '''
import numpy as np

def solve(problem):
    values = np.asarray(problem["values"])
    return {"value": float(np.quantile(values, 0.95))}
''',
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    report: dict[str, object] = {"engine_version": ENGINE_VERSION, "tasks": {}}
    all_ids: set[str] = set()

    for label, source in SYNTHETIC_TASKS.items():
        fp = fingerprint(source, source)
        repeat = fingerprint(source, source)
        assert_true(fp == repeat, f"fingerprint is not deterministic for {label}")
        assert_true(label not in json.dumps(asdict(fp)), f"synthetic label leaked into fingerprint for {label}")

        arms = {}
        for arm in ("v4_full", "v4_no_transfer", "random_search", "template_synthesis", "v3_compatible"):
            proposals = generate_proposals(fp, arm=arm, limit=6, random_seed="validation")
            repeated = generate_proposals(fp, arm=arm, limit=6, random_seed="validation")
            assert_true(proposals == repeated, f"{arm} is not deterministic for {label}")
            assert_true(bool(proposals), f"{arm} generated no proposal for {label}")
            ids = [proposal.proposal_id for proposal in proposals]
            assert_true(len(ids) == len(set(ids)), f"duplicate proposal IDs for {label}/{arm}")
            all_ids.update(ids)
            arms[arm] = [asdict(proposal) for proposal in proposals]

        full_operators = {operator for proposal in arms["v4_full"] for operator in proposal["operators"]}
        v3_operators = {operator for proposal in arms["v3_compatible"] for operator in proposal["operators"]}
        assert_true(len(full_operators - v3_operators) >= 1, f"v4 has no mechanism advantage on {label}")
        assert_true(all(len(proposal["operators"]) == 1 for proposal in arms["template_synthesis"]), "template baseline composed operators")
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
    assert_true(len(selected) == TASK_COUNT, "selector returned the wrong task count")
    assert_true(len({row["family"] for row in selected}) >= MIN_FAMILIES, "selector failed diversity gate")

    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"))
    report["validation_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    report["unique_proposal_ids"] = len(all_ids)
    report["selected_synthetic_inventory"] = selected
    output = Path("validation-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "engine-validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"engine_version": ENGINE_VERSION, "synthetic_tasks": len(SYNTHETIC_TASKS), "unique_proposal_ids": len(all_ids), "validation_sha256": report["validation_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
