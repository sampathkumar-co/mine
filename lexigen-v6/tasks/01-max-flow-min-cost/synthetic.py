from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lexigen-v5"))
sys.path.insert(0, str(HERE))

from candidates import build_candidates, independent_semantic_certificate, official_verifier_accepts


def source_like(n_layers: int, seed: int, width: int = 3) -> dict:
    rng = random.Random(seed)
    n = 2 + n_layers * width
    s, t = 0, n - 1
    cap = [[0 for _ in range(n)] for _ in range(n)]
    cost = [[0 for _ in range(n)] for _ in range(n)]

    def node(layer: int, w: int) -> int:
        return 1 + layer * width + w

    def add(i: int, j: int) -> None:
        cap[i][j] = rng.randint(1, 20)
        cost[i][j] = rng.randint(1, 10)

    for w in range(width):
        add(s, node(0, w))
    for layer in range(n_layers - 1):
        for u in range(width):
            for v in range(width):
                if rng.random() < 0.7:
                    add(node(layer, u), node(layer + 1, v))
    for layer in range(n_layers):
        for u in range(width):
            for v in range(u + 1, width):
                if rng.random() < 0.2:
                    add(node(layer, u), node(layer, v))
    for i in range(n_layers):
        for j in range(i + 2, n_layers):
            for u in range(width):
                for v in range(width):
                    if rng.random() < 0.9:
                        add(node(i, u), node(j, v))
    for w in range(width):
        add(node(n_layers - 1, w), t)
    return {"capacity": cap, "cost": cost, "s": s, "t": t}


def handmade(kind: int) -> dict:
    n = 7
    s, t = 0, 6
    cap = [[0 for _ in range(n)] for _ in range(n)]
    cost = [[0 for _ in range(n)] for _ in range(n)]

    def add(i: int, j: int, c: int, w: int) -> None:
        cap[i][j] = c
        cost[i][j] = w

    if kind == 0:
        add(0, 1, 7, 2); add(1, 6, 7, 3)
    elif kind == 1:
        add(0, 1, 4, 5); add(1, 6, 4, 1); add(0, 2, 6, 1); add(2, 6, 6, 7)
    elif kind == 2:
        add(0, 1, 5, 2); add(1, 3, 5, 1); add(3, 6, 5, 2); add(0, 2, 5, 3); add(2, 3, 5, 1)
    elif kind == 3:
        add(0, 1, 3, 0); add(1, 6, 3, 0); add(0, 2, 8, 4); add(2, 6, 8, 4)
    elif kind == 4:
        add(0, 1, 10, 9); add(1, 4, 4, 1); add(4, 6, 4, 1); add(1, 5, 6, 2); add(5, 6, 6, 2)
    elif kind == 5:
        add(0, 1, 8, 2); add(1, 2, 8, 2); add(2, 6, 8, 2); add(3, 4, 9, 0)
    elif kind == 6:
        add(0, 1, 5, 2); add(1, 6, 5, 2); add(0, 2, 5, 2); add(2, 6, 5, 2); add(1, 2, 2, 1)
    elif kind == 7:
        add(0, 1, 9, 8); add(0, 2, 4, 1); add(2, 3, 4, 1); add(3, 6, 4, 1); add(1, 6, 9, 1)
    else:
        raise ValueError(kind)
    return {"capacity": cap, "cost": cost, "s": s, "t": t}


def problems() -> list[dict]:
    rows = []
    for k in range(16):
        rows.append(source_like(1 + (k % 6), 91000 + k))
    rows.extend(handmade(k) for k in range(8))
    return rows


def main() -> None:
    task_lock = json.loads((HERE / "TASK_LOCK.json").read_text())
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("task-source.py")
    raw = source_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != task_lock["source_sha256"]:
        raise SystemExit("task source sha256 mismatch")
    source_text = raw.decode("utf-8")
    arms = build_candidates(source_text)
    cases = problems()
    rows = []
    candidate_meta = []
    for arm in ["v6_full", "v6_no_transfer", "random_search", "static_template", "v5_compatible", "strong_baseline"]:
        for candidate in arms[arm]:
            candidate_meta.append({
                "name": candidate.name,
                "arm": candidate.arm,
                "implementation_class": candidate.implementation_class,
                "operators": list(candidate.operators),
                "transfer_ids": list(candidate.transfer_ids),
                "learned_template": candidate.learned_template,
                "baseline_id": candidate.baseline_id,
            })
            for index, problem in enumerate(cases):
                started = time.perf_counter_ns()
                error = None
                try:
                    solution = candidate.solve(problem)
                    semantic_valid = independent_semantic_certificate(problem, solution)
                    official_valid = official_verifier_accepts(problem, solution)
                except Exception as exc:
                    semantic_valid = False
                    official_valid = False
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter_ns() - started
                rows.append({
                    "candidate": candidate.name,
                    "arm": candidate.arm,
                    "implementation_class": candidate.implementation_class,
                    "case": index,
                    "semantic_valid": semantic_valid,
                    "official_valid": official_valid,
                    "elapsed_ns": elapsed,
                    "error": error,
                })
    by_candidate = {}
    for meta in candidate_meta:
        subset = [r for r in rows if r["candidate"] == meta["name"]]
        by_candidate[meta["name"]] = {
            **meta,
            "cases": len(subset),
            "semantic_valid": sum(bool(r["semantic_valid"]) for r in subset),
            "official_valid": sum(bool(r["official_valid"]) for r in subset),
            "errors": sum(r["error"] is not None for r in subset),
            "median_elapsed_ns": statistics.median(r["elapsed_ns"] for r in subset),
            "eligible": all(r["semantic_valid"] and r["official_valid"] and r["error"] is None for r in subset),
        }
    payload = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n"
    Path("synthetic-results.jsonl").write_text(payload)
    proposal_payload = json.dumps(candidate_meta, sort_keys=True, separators=(",", ":"))
    Path("synthetic-candidate-plan.json").write_text(json.dumps(candidate_meta, indent=2) + "\n")
    summary = {
        "campaign": "LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication",
        "task_index": 1,
        "task": "max_flow_min_cost",
        "stage": "synthetic_r1",
        "case_count": len(cases),
        "candidate_count": len(candidate_meta),
        "row_count": len(rows),
        "eligible_count": sum(bool(v["eligible"]) for v in by_candidate.values()),
        "by_candidate": by_candidate,
        "candidate_plan_sha256": hashlib.sha256(proposal_payload.encode()).hexdigest(),
        "results_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": importlib.metadata.version("numpy"),
            "networkx": importlib.metadata.version("networkx"),
            "ortools": importlib.metadata.version("ortools"),
        },
        "official_train_manifest_opened": False,
        "official_test_manifest_opened": False,
        "public_task_specific_solvers_opened": False,
        "verifier_capacity_loophole_exploited": False,
    }
    Path("synthetic-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["case_count", "candidate_count", "row_count", "eligible_count", "candidate_plan_sha256", "results_sha256", "versions"]}, indent=2))
    if summary["eligible_count"] != summary["candidate_count"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
