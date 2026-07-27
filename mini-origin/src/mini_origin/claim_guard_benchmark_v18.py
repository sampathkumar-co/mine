from __future__ import annotations

from dataclasses import replace
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from .claim_contract_v18 import (
    Bundle,
    Contract,
    Manifest,
    Run,
    _hash,
    contract,
    manifest,
    valid_bundle,
    verify,
)


MUTATIONS = (
    "boundary",
    "shortcut",
    "holdout_selection",
    "holdout_policy",
    "threshold",
    "candidate",
    "pre_freeze",
    "cherry_pick",
    "duplicate_seed",
    "oracle",
    "budget",
    "control",
    "ablation",
    "contract",
)


def _check(run: Run, name: str, value: bool) -> Run:
    checks = dict(run.checks)
    checks[name] = value
    return replace(run, checks=tuple(checks.items()))


def mutate(
    kind: str,
    spec: Contract,
    sealed: Manifest,
    bundle: Bundle,
) -> tuple[Contract, Manifest, Bundle]:
    runs = list(bundle.runs)
    new_spec, new_manifest = spec, sealed
    if kind == "boundary":
        runs[0] = _check(runs[0], "fixed_boundary", False)
    elif kind == "shortcut":
        runs[1] = _check(runs[1], "shortcut_resistance", False)
    elif kind == "holdout_selection":
        runs[0] = replace(runs[0], holdout_candidates=48, selected_after_holdout=True)
    elif kind == "holdout_policy":
        runs[0] = replace(runs[0], holdout_policy_violations=1)
    elif kind == "threshold":
        runs = [replace(run, threshold_used=0.70) for run in runs]
    elif kind == "candidate":
        runs[0] = replace(runs[0], candidate_hash=_hash("changed-candidate"))
    elif kind == "pre_freeze":
        new_manifest = replace(sealed, issued_after_freeze=False)
    elif kind == "cherry_pick":
        runs.pop(min(range(len(runs)), key=lambda index: runs[index].score))
    elif kind == "duplicate_seed":
        runs[-1] = replace(runs[-1], seed=runs[0].seed)
    elif kind == "oracle":
        runs[0] = replace(runs[0], score=1.04)
    elif kind == "budget":
        runs[0] = replace(runs[0], candidate_budget=0.30)
    elif kind == "control":
        runs[0] = replace(runs[0], control=runs[0].score + 0.02)
    elif kind == "ablation":
        runs[0] = replace(runs[0], ablation=runs[0].score + 0.01)
    elif kind == "contract":
        new_spec = replace(spec, score_threshold=0.70)
    else:
        raise ValueError(kind)
    return new_spec, new_manifest, Bundle(bundle.claim_id, True, tuple(runs))


def threshold_gate(spec: Contract, bundle: Bundle) -> bool:
    return statistics.median(run.score for run in bundle.runs) >= spec.score_threshold


def replication_gate(spec: Contract, bundle: Bundle) -> bool:
    return (
        len(bundle.runs) >= spec.required_runs
        and sum(run.score >= spec.score_threshold for run in bundle.runs)
        >= spec.min_successes
    )


def provenance_gate(spec: Contract, sealed: Manifest, bundle: Bundle) -> bool:
    return (
        sealed.contract_digest == spec.digest
        and sealed.candidate_hash == spec.candidate_hash
        and all(run.contract_digest == spec.digest for run in bundle.runs)
        and all(run.candidate_hash == spec.candidate_hash for run in bundle.runs)
    )


def red_team(seed: int, repeats: int = 16, valid_trials: int = 64) -> dict[str, object]:
    rng = random.Random(seed)
    spec = contract()
    sealed = manifest(spec, f"evaluator-{seed}")
    detectors = {
        "threshold": lambda c, m, b: threshold_gate(c, b),
        "replication": lambda c, m, b: replication_gate(c, b),
        "provenance": provenance_gate,
        "claim_guard": lambda c, m, b: bool(verify(c, m, b)["accepted"]),
    }
    false_accepts = {name: 0 for name in detectors}
    for kind in MUTATIONS:
        for _ in range(repeats):
            changed = mutate(kind, spec, sealed, valid_bundle(spec, sealed, rng))
            for name, detector in detectors.items():
                false_accepts[name] += int(detector(*changed))
    total = len(MUTATIONS) * repeats
    false_rejects = {name: 0 for name in detectors}
    for _ in range(valid_trials):
        bundle = valid_bundle(spec, sealed, rng)
        for name, detector in detectors.items():
            false_rejects[name] += int(not detector(spec, sealed, bundle))
    detection = {
        name: 1.0 - count / total for name, count in false_accepts.items()
    }
    false_reject_rate = {
        name: count / valid_trials for name, count in false_rejects.items()
    }
    best_simple = max(value for name, value in detection.items() if name != "claim_guard")
    return {
        "seed": seed,
        "mutation_types": list(MUTATIONS),
        "invalid_trials": total,
        "valid_trials": valid_trials,
        "false_accepts": false_accepts,
        "detection_rate": detection,
        "false_rejects": false_rejects,
        "false_reject_rate": false_reject_rate,
        "claim_guard_pass": (
            detection["claim_guard"] >= 0.99
            and false_reject_rate["claim_guard"] == 0.0
            and detection["claim_guard"] - best_simple >= 0.30
        ),
    }


def historical_replay() -> dict[str, object]:
    spec = contract()
    sealed = manifest(spec, "historical")
    base = valid_bundle(spec, sealed, random.Random(18))
    cases = [("valid_v05_style", True, spec, sealed, base)]
    names = (
        "v01_toroidal_wrap",
        "v05_covariance_shortcut",
        "hidden_selection",
        "hidden_policy_violation",
        "posthoc_threshold",
        "candidate_changed",
        "manifest_pre_freeze",
        "seed_cherry_pick",
        "duplicate_seed",
        "impossible_score",
        "unfair_budget",
        "stronger_control",
        "stronger_ablation",
        "contract_modified",
    )
    for name, kind in zip(names, MUTATIONS):
        cases.append((name, False, *mutate(kind, spec, sealed, base)))
    rows = []
    for name, expected, case_spec, case_manifest, case_bundle in cases:
        verdict = verify(case_spec, case_manifest, case_bundle)
        rows.append(
            {
                "case": name,
                "expected": expected,
                "observed": verdict["accepted"],
                "correct": verdict["accepted"] == expected,
                "problems": verdict["problems"],
            }
        )
    return {
        "case_count": len(rows),
        "correct_count": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "all_correct": all(row["correct"] for row in rows),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "status": "prospective_claim_contract_candidate",
        "claim_scope": (
            "prospective enforcement of candidate freeze, sealed holdout seeds, "
            "single-touch evaluation, immutable thresholds, feasible oracle ceilings, "
            "equal control budgets, replication and mandatory ablations"
        ),
        "red_team": red_team(args.seed),
        "historical_replay": historical_replay(),
    }
    report["candidate_gate"] = bool(
        report["red_team"]["claim_guard_pass"]
        and report["historical_replay"]["all_correct"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_gate": report["candidate_gate"],
                "historical_accuracy": report["historical_replay"]["accuracy"],
                "detection": report["red_team"]["detection_rate"],
                "false_reject": report["red_team"]["false_reject_rate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
