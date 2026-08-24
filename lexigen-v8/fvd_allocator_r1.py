from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

GRADE_WEIGHT = {
    "confirmed_causal": 1.00,
    "source_learned_unconfirmed": 0.35,
    "generic_baseline": 0.15,
}
SHUFFLE_SEED = "LEXIGEN-V8-FVD-SHUFFLED-EXPERIENCE-R1"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def overlap_score(context_tags: list[str], traits: set[str]) -> tuple[float, list[str]]:
    tags = set(map(str, context_tags))
    matched = sorted(tags & traits)
    if not tags:
        return 0.0, matched
    return len(matched) / len(tags), matched


def shuffled_profiles(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = sorted(
        range(len(profiles)),
        key=lambda i: hashlib.sha256(f"{SHUFFLE_SEED}\0{profiles[i]['proposal_class_id']}".encode()).hexdigest(),
    )
    payloads = []
    for p in profiles:
        payloads.append({
            "context_tags": list(p.get("context_tags", [])),
            "evidence_grade": p.get("evidence_grade", "generic_baseline"),
            "source_families": list(p.get("source_families", [])),
            "source_causal_ids": list(p.get("source_causal_ids", [])),
            "confirmed_causal_successes": int(p.get("confirmed_causal_successes", 0)),
            "risk_tags": list(p.get("risk_tags", [])),
        })
    rotated = [None] * len(profiles)
    for dest_pos, dest_idx in enumerate(order):
        src_idx = order[(dest_pos + 1) % len(order)]
        rotated[dest_idx] = payloads[src_idx]
    out = []
    for i, p in enumerate(profiles):
        q = dict(p)
        q.update(rotated[i])
        out.append(q)
    return out


def score_profile(profile: dict[str, Any], traits: set[str], arm: str) -> tuple[float, dict[str, Any]]:
    coverage, matched = overlap_score(list(profile.get("context_tags", [])), traits)
    grade = str(profile.get("evidence_grade", "generic_baseline"))
    risk_hits = sorted(set(map(str, profile.get("risk_tags", []))) & traits)

    if arm in {"fvd_no_experience", "evolution_only"}:
        score = 1.0
    elif arm == "retrieval_only":
        # Retrieval sees context similarity but intentionally ignores causal evidence grade,
        # global guards, and negative-lesson controller penalties.
        score = 0.20 + 2.00 * coverage
    else:
        grade_weight = GRADE_WEIGHT.get(grade)
        if grade_weight is None:
            raise RuntimeError(f"unknown evidence grade: {grade}")
        # Full/shuffled use the same frozen controller. Experience changes only the
        # profile payload presented to it. The nonzero base keeps exploration alive.
        score = 0.20 + 1.50 * coverage + grade_weight * coverage
        if risk_hits:
            score -= 1.25
        if profile.get("confirmed_causal_successes", 0) and coverage > 0:
            score += 0.10

    score = max(0.05, float(score))
    return score, {
        "proposal_class_id": profile["proposal_class_id"],
        "score": score,
        "context_coverage": coverage,
        "matched_tags": matched,
        "risk_hits": risk_hits,
        "evidence_grade": grade,
        "confirmed_causal_successes": int(profile.get("confirmed_causal_successes", 0)),
        "source_causal_ids": list(profile.get("source_causal_ids", [])),
    }


def allocate(scores: list[tuple[str, float]], budget: int) -> dict[str, int]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not scores:
        raise ValueError("at least one proposal class is required")
    total = sum(max(0.0, score) for _, score in scores)
    if total <= 0:
        raise ValueError("all proposal scores are nonpositive")
    exact = {pid: budget * score / total for pid, score in scores}
    counts = {pid: int(math.floor(value)) for pid, value in exact.items()}
    remainder = budget - sum(counts.values())
    ranked_fraction = sorted(
        scores,
        key=lambda row: (-(exact[row[0]] - counts[row[0]]), -row[1], row[0]),
    )
    for pid, _ in ranked_fraction[:remainder]:
        counts[pid] += 1
    if sum(counts.values()) != budget:
        raise AssertionError("budget accounting error")
    return counts


def run_allocator(artifact: dict[str, Any], task: dict[str, Any], arm: str, budget: int) -> dict[str, Any]:
    valid_arms = {
        "fvd_full",
        "fvd_no_experience",
        "fvd_shuffled_experience",
        "retrieval_only",
        "evolution_only",
    }
    if arm not in valid_arms:
        raise ValueError(f"unsupported allocator arm: {arm}")
    if artifact.get("official_holdout_data_accessed") is not False:
        raise RuntimeError("development artifact crossed official holdout boundary")
    traits = set(map(str, task.get("traits", [])))
    base_profiles = [dict(p) for p in artifact["proposal_profiles"]]
    profiles = shuffled_profiles(base_profiles) if arm == "fvd_shuffled_experience" else base_profiles

    details = []
    scores = []
    for profile in profiles:
        score_arm = "fvd_full" if arm == "fvd_shuffled_experience" else arm
        score, detail = score_profile(profile, traits, score_arm)
        details.append(detail)
        scores.append((str(profile["proposal_class_id"]), score))

    counts = allocate(scores, budget)
    ranking = sorted(details, key=lambda d: (-float(d["score"]), str(d["proposal_class_id"])))
    experience_view = {
        "arm": arm,
        "profiles": [
            {
                "proposal_class_id": p["proposal_class_id"],
                "context_tags": p.get("context_tags", []),
                "evidence_grade": p.get("evidence_grade"),
                "source_causal_ids": p.get("source_causal_ids", []),
                "risk_tags": p.get("risk_tags", []),
            }
            for p in profiles
        ] if arm not in {"fvd_no_experience", "evolution_only"} else [],
    }
    return {
        "schema": "lexigen-v8-fvd-allocation-r1",
        "arm": arm,
        "task_descriptor_id": task.get("task_descriptor_id"),
        "task_traits": sorted(traits),
        "proposal_budget": budget,
        "budget_used": sum(counts.values()),
        "allocation": dict(sorted(counts.items())),
        "ranking": ranking,
        "experience_artifact_sha256": artifact.get("artifact_sha256"),
        "experience_view_sha256": canonical_sha256(experience_view),
        "official_holdout_data_accessed": False,
        "scientific_transfer_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_allocator(load_json(args.artifact), load_json(args.task), args.arm, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
