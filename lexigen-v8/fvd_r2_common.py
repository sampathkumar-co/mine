from __future__ import annotations

import hashlib
import json
import math
from typing import Any

SHUFFLE_SEED = "LEXIGEN-V8-FVD-R2-SHUFFLED-OUTCOME-LEDGER-R1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classes_from_candidate(candidate: str) -> list[str]:
    """Task-independent mapping from historical candidate-name mechanism tokens to FVD classes."""
    name = candidate.lower()
    out: set[str] = set()
    if "frontier" in name or ("bit" in name and "sparse" in name):
        out.add("PC-FRONTIER-CERT")
    if "active" in name:
        out.add("PC-ACTIVE-CORE")
    if "structure" in name and "refine" in name:
        out.add("PC-REDUCED-REFINE")
    if any(token in name for token in ("dtype", "mixed", "float32", "precision")):
        out.add("PC-PRECISION-BACKEND")
    if "vector" in name:
        out.add("PC-EXEC-CERT")
    if any(token in name for token in ("sort", "partition", "closed")):
        out.add("PC-REDUCE-EXEC")
    if any(token in name for token in ("zero", "contiguous", "bit")):
        out.add("PC-REP-SPECIALIZE")
    if "early" in name:
        out.add("PC-RESTRICT-RECOVER")
    return sorted(out)


def classes_from_root_cause(text: str) -> list[str]:
    """Task-independent mapping from historical root-cause language to mechanism classes."""
    lower = text.lower()
    out: set[str] = set()
    if any(token in lower for token in ("float32", "dtype", "precision", "tolerance", "phase error", "numerical error", "error accumulation", "accumulate")):
        out.add("PC-PRECISION-BACKEND")
    if any(token in lower for token in ("frontier", "branch-and-bound", "search explosion", "search space")):
        out.add("PC-FRONTIER-CERT")
    if any(token in lower for token in ("active set", "active-set", "constraint reduction")):
        out.add("PC-ACTIVE-CORE")
    if any(token in lower for token in ("reduction invalid", "domain reduction", "closed form")):
        out.add("PC-REDUCE-EXEC")
    if any(token in lower for token in ("layout", "representation", "contiguous")):
        out.add("PC-REP-SPECIALIZE")
    if any(token in lower for token in ("kernel", "execution path", "vectorized")):
        out.add("PC-EXEC-CERT")
    if any(token in lower for token in ("early certificate", "predicate", "restriction invalid")):
        out.add("PC-RESTRICT-RECOVER")
    return sorted(out)


def nested_get(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def context_coverage(context_tags: list[str], traits: set[str]) -> float:
    tags = set(map(str, context_tags))
    if not tags:
        return 0.0
    return len(tags & traits) / len(tags)


def reachability_factor(class_id: str, traits: set[str], controller: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    cfg = controller["generic_reachability"]
    rule = cfg["rules"][class_id]
    boost_hits = sorted(set(map(str, rule.get("boost_tags", []))) & traits)
    blocker_hits = sorted(set(map(str, rule.get("blocker_tags", []))) & traits)
    factor = 1.0
    if boost_hits:
        factor *= float(cfg["boost_factor_per_any_hit"])
    if blocker_hits:
        factor *= float(cfg["blocker_factor_per_any_hit"])
    factor = max(float(cfg["floor_factor"]), factor)
    return factor, {"boost_hits": boost_hits, "blocker_hits": blocker_hits}


def predicted_success_probability(class_id: str, traits: set[str], ledger: dict[str, Any], controller: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    cfg = controller["outcome_model"]
    alpha = float(cfg["beta_prior_success"])
    beta = float(cfg["beta_prior_failure"])
    neighbors: list[dict[str, Any]] = []
    for record in ledger["records"]:
        if record["proposal_class_id"] != class_id:
            continue
        other = set(map(str, record["traits"]))
        sim = jaccard(traits, other)
        weight = max(0.10, sim) ** 2
        success = 1.0 if record["gate_success"] else 0.0
        alpha += weight * success
        beta += weight * (1.0 - success)
        neighbors.append({
            "task": record["task"],
            "similarity": sim,
            "weight": weight,
            "gate_success": bool(record["gate_success"]),
        })
    if not neighbors:
        return float(cfg["no_same_class_records_probability"]), []
    return alpha / (alpha + beta), sorted(neighbors, key=lambda x: (-x["similarity"], x["task"]))


def utility_multiplier(probability: float, controller: dict[str, Any]) -> float:
    # Frozen R2 spec: 0.65 + 0.70 * predicted_success_probability.
    return 0.65 + 0.70 * probability


def shuffled_ledger(ledger: dict[str, Any], proposal_class_ids: list[str]) -> dict[str, Any]:
    ids = sorted(proposal_class_ids)
    order = sorted(ids, key=lambda pid: hashlib.sha256(f"{SHUFFLE_SEED}\0{pid}".encode()).hexdigest())
    mapping = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    out = dict(ledger)
    out["records"] = [dict(record, proposal_class_id=mapping[record["proposal_class_id"]]) for record in ledger["records"]]
    out["shuffled_class_mapping"] = mapping
    out["artifact_sha256"] = canonical_sha256({k: v for k, v in out.items() if k != "artifact_sha256"})
    return out


def allocate(scores: list[tuple[str, float]], budget: int) -> dict[str, int]:
    if budget <= 0 or not scores:
        raise ValueError("invalid budget or empty score list")
    total = sum(max(0.0, score) for _, score in scores)
    if total <= 0:
        raise ValueError("all scores are nonpositive")
    exact = {pid: budget * max(0.0, score) / total for pid, score in scores}
    counts = {pid: int(math.floor(value)) for pid, value in exact.items()}
    remainder = budget - sum(counts.values())
    ranked = sorted(scores, key=lambda row: (-(exact[row[0]] - counts[row[0]]), -row[1], row[0]))
    for pid, _ in ranked[:remainder]:
        counts[pid] += 1
    if sum(counts.values()) != budget:
        raise AssertionError("budget accounting error")
    return counts
