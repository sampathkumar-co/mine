from __future__ import annotations

import hashlib
import itertools
import json
import random
from typing import Iterable

import engine_v2 as ranking
import engine_v3 as semantic
from engine import OPERATORS, Proposal, V3_ALLOWED, _compose, _operator_score

ENGINE_VERSION = "lexigen-v4.0.3-prelock"
Fingerprint = semantic.Fingerprint
fingerprint = semantic.fingerprint
failure_update = semantic.failure_update


def _reachable_pool(features: set[str]) -> list[object]:
    known = set(features)
    selected: list[object] = []
    remaining = list(OPERATORS)
    changed = True
    while changed:
        changed = False
        next_remaining: list[object] = []
        for operator in remaining:
            if ranking._eligible(operator.name, known):
                selected.append(operator)
                known.update(operator.provides)
                changed = True
            else:
                next_remaining.append(operator)
        remaining = next_remaining
    return selected


def generate_proposals(
    task_fingerprint: Fingerprint,
    arm: str = "v4_full",
    limit: int = 6,
    random_seed: str = "LEXIGEN-V4",
) -> list[Proposal]:
    if arm not in {"v4_full", "v4_no_transfer", "random_search", "template_synthesis", "v3_compatible"}:
        raise ValueError(f"unknown arm: {arm}")
    if limit <= 0:
        return []

    features = set(task_fingerprint.features)
    if arm in {"v4_full", "v4_no_transfer", "random_search"}:
        pool = _reachable_pool(features)
        lengths = (1, 2, 3)
    else:
        pool = [operator for operator in OPERATORS if ranking._eligible(operator.name, features)]
        lengths = (1,)
    if arm == "v3_compatible":
        pool = [operator for operator in pool if operator.name in V3_ALLOWED]
    use_transfer = arm == "v4_full"

    candidates: list[tuple[tuple[object, ...], float, float, tuple[str, ...]]] = []
    for length in lengths:
        for composition in itertools.combinations(pool, length):
            known = set(features)
            compatible = True
            score = 0.0
            risk = 0.0
            reasons: list[str] = []
            for operator in composition:
                if not ranking._eligible(operator.name, known):
                    compatible = False
                    break
                operator_score, operator_risk, operator_reasons = _operator_score(operator, known, use_transfer)
                score += operator_score
                risk += operator_risk
                reasons.extend(operator_reasons)
                known.update(operator.provides)
            if not compatible:
                continue
            synergy_score, composed_risk, _, synergy_reasons = _compose(composition, features)
            score += 0.34 * synergy_score
            risk = 0.55 * risk + 0.45 * composed_risk
            score -= 0.65 * risk
            score -= 0.10 * max(0, length - 1)
            reasons.extend(synergy_reasons)
            candidates.append((composition, score, risk, tuple(reasons)))

    candidates.sort(key=lambda row: (-row[1], row[2], tuple(operator.name for operator in row[0])))
    if arm == "random_search":
        seed_material = f"{random_seed}\0{task_fingerprint.source_sha256}\0{task_fingerprint.verifier_sha256}"
        rng = random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest(), 16))
        rng.shuffle(candidates)

    proposals: list[Proposal] = []
    seen: set[tuple[str, ...]] = set()
    for composition, score, risk, reasons in candidates:
        names = tuple(operator.name for operator in composition)
        if names in seen:
            continue
        seen.add(names)
        payload = {
            "engine": ENGINE_VERSION,
            "arm": arm,
            "operators": names,
            "fingerprint": task_fingerprint.source_sha256,
        }
        proposals.append(
            Proposal(
                arm=arm,
                rank=len(proposals) + 1,
                operators=names,
                score=round(score, 12),
                predicted_benefit=round(sum(operator.base_benefit for operator in composition), 12),
                correctness_risk=round(risk, 12),
                rationale=reasons,
                proposal_id=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20],
            )
        )
        if len(proposals) >= limit:
            break
    return proposals


def serialise_fingerprint(value: Fingerprint) -> str:
    return semantic.serialise_fingerprint(value)


def serialise_proposals(values: Iterable[Proposal]) -> str:
    return json.dumps([value.__dict__ for value in values], sort_keys=True, indent=2)


__all__ = [
    "ENGINE_VERSION",
    "Fingerprint",
    "Proposal",
    "failure_update",
    "fingerprint",
    "generate_proposals",
    "serialise_fingerprint",
    "serialise_proposals",
]
