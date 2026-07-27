from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from typing import Callable, Iterable

from .claim_contract_v18 import (
    Bundle,
    Contract,
    Manifest,
    Run,
    SHA256_PATTERN,
    _hash,
    contract,
    manifest,
    valid_bundle,
)
from .claim_guard_benchmark_v18 import MUTATIONS, mutate


@dataclass(frozen=True)
class ProtocolCase:
    name: str
    valid: bool
    contract: Contract
    manifest: Manifest
    bundle: Bundle


Predicate = Callable[[Contract, Manifest, Bundle], bool]


@dataclass(frozen=True)
class ProtocolRule:
    name: str
    cost: float
    predicate: Predicate


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _checks(run: Run) -> dict[str, bool]:
    return dict(run.checks)


def _manifest_commitment(spec: Contract, seeds: Iterable[int]) -> str:
    return _hash(f"{spec.digest}:sealed-v1:" + ",".join(map(str, seeds)))


def _success_metrics(spec: Contract, bundle: Bundle) -> tuple[int, list[float], list[float], list[float]]:
    successful = 0
    scores: list[float] = []
    control_gaps: list[float] = []
    ablation_gaps: list[float] = []
    for run in bundle.runs:
        if not _finite(run.score, run.control, run.ablation):
            continue
        control_gap = run.score - run.control
        ablation_gap = run.score - run.ablation
        scores.append(run.score)
        control_gaps.append(control_gap)
        ablation_gaps.append(ablation_gap)
        if (
            run.score >= spec.score_threshold
            and control_gap >= spec.min_control_gap
            and ablation_gap >= spec.min_ablation_gap
        ):
            successful += 1
    return successful, scores, control_gaps, ablation_gaps


def rule_library() -> list[ProtocolRule]:
    def identity(spec: Contract, _: Manifest, __: Bundle) -> bool:
        return bool(spec.claim_id.strip()) and bool(SHA256_PATTERN.fullmatch(spec.candidate_hash))

    def counts(spec: Contract, _: Manifest, __: Bundle) -> bool:
        return spec.required_runs > 0 and 1 <= spec.min_successes <= spec.required_runs

    def numeric_contract(spec: Contract, _: Manifest, __: Bundle) -> bool:
        values = (
            spec.score_threshold,
            spec.median_threshold,
            spec.min_control_gap,
            spec.median_control_gap,
            spec.min_ablation_gap,
            spec.oracle_ceiling,
            spec.operation_budget,
        )
        return (
            _finite(*values)
            and spec.oracle_ceiling > 0.0
            and 0.0 <= spec.score_threshold <= spec.oracle_ceiling
            and 0.0 <= spec.median_threshold <= spec.oracle_ceiling
            and spec.min_control_gap >= 0.0
            and spec.median_control_gap >= 0.0
            and spec.min_ablation_gap >= 0.0
            and spec.operation_budget >= 0.0
        )

    def check_schema(spec: Contract, _: Manifest, __: Bundle) -> bool:
        return (
            bool(spec.required_checks)
            and all(name.strip() for name in spec.required_checks)
            and len(set(spec.required_checks)) == len(spec.required_checks)
        )

    def manifest_binding(spec: Contract, sealed: Manifest, _: Bundle) -> bool:
        return (
            sealed.contract_digest == spec.digest
            and sealed.candidate_hash == spec.candidate_hash
        )

    def freeze_order(_: Contract, sealed: Manifest, __: Bundle) -> bool:
        return sealed.issued_after_freeze

    def manifest_seeds(spec: Contract, sealed: Manifest, _: Bundle) -> bool:
        return (
            len(sealed.seeds) == spec.required_runs
            and len(set(sealed.seeds)) == len(sealed.seeds)
            and all(seed > 0 for seed in sealed.seeds)
        )

    def commitment(spec: Contract, sealed: Manifest, _: Bundle) -> bool:
        return sealed.commitment == _manifest_commitment(spec, sealed.seeds)

    def bundle_binding(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        return bundle.claim_id == spec.claim_id and len(bundle.runs) == spec.required_runs

    def run_seeds(_: Contract, sealed: Manifest, bundle: Bundle) -> bool:
        seeds = [run.seed for run in bundle.runs]
        return (
            len(set(seeds)) == len(seeds)
            and all(seed > 0 for seed in seeds)
            and set(seeds) == set(sealed.seeds)
        )

    def provenance(spec: Contract, sealed: Manifest, bundle: Bundle) -> bool:
        return all(
            run.contract_digest == spec.digest
            and run.candidate_hash == spec.candidate_hash
            and run.manifest_digest == sealed.digest
            for run in bundle.runs
        )

    def threshold_lock(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        return all(
            math.isfinite(run.threshold_used)
            and math.isclose(
                run.threshold_used,
                spec.score_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for run in bundle.runs
        )

    def holdout_isolation(_: Contract, __: Manifest, bundle: Bundle) -> bool:
        return all(
            run.holdout_candidates == 1
            and not run.selected_after_holdout
            and run.holdout_policy_violations == 0
            for run in bundle.runs
        )

    def required_checks(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        for run in bundle.runs:
            names = [name for name, _ in run.checks]
            if len(set(names)) != len(names):
                return False
            checks = _checks(run)
            if any(checks.get(name) is not True for name in spec.required_checks):
                return False
        return True

    def metric_domain(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        return all(
            _finite(run.score, run.control, run.ablation)
            and 0.0 <= run.score <= spec.oracle_ceiling
            and 0.0 <= run.control <= spec.oracle_ceiling
            and 0.0 <= run.ablation <= spec.oracle_ceiling
            for run in bundle.runs
        )

    def budget_integrity(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        return all(
            _finite(run.candidate_budget, run.control_budget)
            and 0.0 <= run.candidate_budget <= spec.operation_budget + 1e-12
            and 0.0 <= run.control_budget <= spec.operation_budget + 1e-12
            and abs(run.candidate_budget - run.control_budget) <= 1e-12
            for run in bundle.runs
        )

    def per_run_evidence(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        return all(
            _finite(run.score, run.control, run.ablation)
            and run.score - run.control >= spec.min_control_gap
            and run.score - run.ablation >= spec.min_ablation_gap
            for run in bundle.runs
        )

    def aggregate_evidence(spec: Contract, _: Manifest, bundle: Bundle) -> bool:
        successful, scores, control_gaps, ablation_gaps = _success_metrics(spec, bundle)
        return (
            len(scores) == len(bundle.runs)
            and successful >= spec.min_successes
            and statistics.median(scores) >= spec.median_threshold
            and statistics.median(control_gaps) >= spec.median_control_gap
            and min(ablation_gaps, default=-math.inf) >= spec.min_ablation_gap
        )

    def breakthrough_label(_: Contract, __: Manifest, bundle: Bundle) -> bool:
        return bundle.claimed_breakthrough

    return [
        ProtocolRule("contract_identity", 1.0, identity),
        ProtocolRule("contract_counts", 1.0, counts),
        ProtocolRule("contract_numeric_domain", 1.5, numeric_contract),
        ProtocolRule("required_check_schema", 1.0, check_schema),
        ProtocolRule("manifest_binding", 1.0, manifest_binding),
        ProtocolRule("candidate_freeze_order", 0.8, freeze_order),
        ProtocolRule("manifest_seed_integrity", 1.2, manifest_seeds),
        ProtocolRule("seed_commitment", 1.0, commitment),
        ProtocolRule("bundle_binding", 0.8, bundle_binding),
        ProtocolRule("run_seed_integrity", 1.0, run_seeds),
        ProtocolRule("run_provenance", 1.2, provenance),
        ProtocolRule("threshold_immutability", 0.8, threshold_lock),
        ProtocolRule("holdout_isolation", 1.0, holdout_isolation),
        ProtocolRule("mandatory_checks", 1.0, required_checks),
        ProtocolRule("metric_domain", 1.2, metric_domain),
        ProtocolRule("budget_integrity", 1.0, budget_integrity),
        ProtocolRule("per_run_evidence", 1.0, per_run_evidence),
        ProtocolRule("aggregate_evidence", 1.3, aggregate_evidence),
        ProtocolRule("breakthrough_label", 0.5, breakthrough_label),
    ]


def accepts(rules: Iterable[ProtocolRule], case: ProtocolCase) -> bool:
    return all(
        rule.predicate(case.contract, case.manifest, case.bundle)
        for rule in rules
    )


def _case(name: str, valid: bool, value: tuple[Contract, Manifest, Bundle]) -> ProtocolCase:
    return ProtocolCase(name, valid, *value)


def _self_consistent_case(
    name: str,
    spec: Contract,
    seed: int,
    mutate_bundle: Callable[[Bundle], Bundle] | None = None,
) -> ProtocolCase:
    sealed = manifest(spec, f"protocol-synth-{name}-{seed}")
    bundle = valid_bundle(spec, sealed, random.Random(seed))
    if mutate_bundle is not None:
        bundle = mutate_bundle(bundle)
    return ProtocolCase(name, False, spec, sealed, bundle)


def _replace_first(bundle: Bundle, **changes: object) -> Bundle:
    return replace(
        bundle,
        runs=(replace(bundle.runs[0], **changes),) + bundle.runs[1:],
    )


def build_training_cases(seed: int) -> list[ProtocolCase]:
    spec = contract()
    sealed = manifest(spec, f"training-{seed}")
    cases: list[ProtocolCase] = [
        ProtocolCase(
            f"valid-{index}",
            True,
            spec,
            sealed,
            valid_bundle(spec, sealed, random.Random(seed + index)),
        )
        for index in range(32)
    ]
    base = valid_bundle(spec, sealed, random.Random(seed + 100))
    for kind in MUTATIONS:
        cases.append(_case(f"first-party-{kind}", False, mutate(kind, spec, sealed, base)))

    cases.extend(
        [
            _self_consistent_case("identity-empty-claim", replace(spec, claim_id=""), seed),
            _self_consistent_case("count-zero-success", replace(spec, min_successes=0), seed),
            _self_consistent_case("numeric-negative-score", replace(spec, score_threshold=-0.2), seed),
            _self_consistent_case("checks-empty", replace(spec, required_checks=()), seed),
            _self_consistent_case(
                "metric-negative-control",
                spec,
                seed,
                lambda bundle: _replace_first(bundle, control=-0.1),
            ),
            _self_consistent_case(
                "holdout-zero",
                spec,
                seed,
                lambda bundle: _replace_first(bundle, holdout_candidates=0),
            ),
            _self_consistent_case(
                "budget-negative",
                spec,
                seed,
                lambda bundle: _replace_first(
                    bundle,
                    candidate_budget=-0.2,
                    control_budget=-0.2,
                ),
            ),
        ]
    )
    return cases


def _rebind_seeds(
    spec: Contract,
    sealed: Manifest,
    bundle: Bundle,
    seeds: tuple[int, ...],
) -> tuple[Manifest, Bundle]:
    changed_manifest = replace(
        sealed,
        seeds=seeds,
        commitment=_manifest_commitment(spec, seeds),
    )
    changed_bundle = replace(
        bundle,
        runs=tuple(
            replace(
                run,
                seed=seed,
                manifest_digest=changed_manifest.digest,
            )
            for run, seed in zip(bundle.runs, seeds)
        ),
    )
    return changed_manifest, changed_bundle


def build_hidden_cases(seed: int) -> list[ProtocolCase]:
    # This function is called only after the protocol has been frozen.
    spec = contract()
    sealed = manifest(spec, f"withheld-{seed}")
    valid_cases = [
        ProtocolCase(
            f"hidden-valid-{index}",
            True,
            spec,
            sealed,
            valid_bundle(spec, sealed, random.Random(seed + 1000 + index)),
        )
        for index in range(64)
    ]
    base = valid_bundle(spec, sealed, random.Random(seed + 2000))

    hidden: list[ProtocolCase] = [
        _self_consistent_case("hidden-empty-hash", replace(spec, candidate_hash=""), seed),
        _self_consistent_case("hidden-run-count-zero", replace(spec, required_runs=0, min_successes=0), seed),
        _self_consistent_case("hidden-successes-too-high", replace(spec, min_successes=6), seed),
        _self_consistent_case("hidden-negative-median", replace(spec, median_threshold=-0.2), seed),
        _self_consistent_case("hidden-negative-ablation-gap", replace(spec, min_ablation_gap=-0.1), seed),
        _self_consistent_case(
            "hidden-negative-operation-budget",
            replace(spec, operation_budget=-0.2),
            seed,
            lambda bundle: replace(
                bundle,
                runs=tuple(
                    replace(run, candidate_budget=-0.2, control_budget=-0.2)
                    for run in bundle.runs
                ),
            ),
        ),
        _self_consistent_case("hidden-duplicate-check-schema", replace(spec, required_checks=("fixed_boundary", "fixed_boundary")), seed),
        _self_consistent_case(
            "hidden-negative-ablation",
            spec,
            seed,
            lambda bundle: _replace_first(bundle, ablation=-0.1),
        ),
        _self_consistent_case(
            "hidden-negative-holdout-count",
            spec,
            seed,
            lambda bundle: _replace_first(bundle, holdout_candidates=-2),
        ),
        _self_consistent_case(
            "hidden-control-budget-exceeded",
            spec,
            seed,
            lambda bundle: _replace_first(bundle, control_budget=0.3),
        ),
        ProtocolCase(
            "hidden-no-breakthrough-label",
            False,
            spec,
            sealed,
            replace(base, claimed_breakthrough=False),
        ),
        ProtocolCase(
            "hidden-missing-required-check",
            False,
            spec,
            sealed,
            _replace_first(
                base,
                checks=tuple(
                    item for item in base.runs[0].checks
                    if item[0] != "control_present"
                ),
            ),
        ),
        ProtocolCase(
            "hidden-duplicate-run-check",
            False,
            spec,
            sealed,
            _replace_first(
                base,
                checks=base.runs[0].checks + (("fixed_boundary", True),),
            ),
        ),
    ]

    bad_seeds = (0,) + sealed.seeds[1:]
    bad_manifest, bad_bundle = _rebind_seeds(spec, sealed, base, bad_seeds)
    hidden.append(ProtocolCase("hidden-nonpositive-seed", False, spec, bad_manifest, bad_bundle))

    short_seeds = sealed.seeds[:-1]
    short_manifest = replace(
        sealed,
        seeds=short_seeds,
        commitment=_manifest_commitment(spec, short_seeds),
    )
    hidden.append(ProtocolCase("hidden-manifest-seed-count", False, spec, short_manifest, base))

    return valid_cases + hidden


@dataclass(frozen=True)
class SynthesisedProtocol:
    selected: tuple[ProtocolRule, ...]
    trace: tuple[dict[str, object], ...]

    def names(self) -> tuple[str, ...]:
        return tuple(rule.name for rule in self.selected)


def synthesise_protocol(
    cases: list[ProtocolCase],
    library: list[ProtocolRule],
) -> SynthesisedProtocol:
    valid = [case for case in cases if case.valid]
    invalid = [case for case in cases if not case.valid]
    safe = [rule for rule in library if all(accepts((rule,), case) for case in valid)]
    selected: list[ProtocolRule] = []
    remaining = list(invalid)
    trace: list[dict[str, object]] = []

    while remaining:
        candidates = []
        for rule in safe:
            if rule in selected:
                continue
            rejected = [case for case in remaining if not accepts((rule,), case)]
            if rejected:
                utility = len(rejected) / rule.cost
                candidates.append((utility, len(rejected), -rule.cost, rule, rejected))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3].name), reverse=True)
        _, _, _, chosen, rejected = candidates[0]
        selected.append(chosen)
        rejected_names = {case.name for case in rejected}
        remaining = [case for case in remaining if case.name not in rejected_names]
        trace.append(
            {
                "iteration": len(trace),
                "rule": chosen.name,
                "rejected": sorted(rejected_names),
                "remaining_invalid": len(remaining),
            }
        )

    # Minimum-description post-pass: remove any selected rule whose removal
    # preserves perfect classification on the frozen training set.
    changed = True
    while changed:
        changed = False
        for rule in sorted(selected, key=lambda item: item.cost, reverse=True):
            trial = [item for item in selected if item != rule]
            if all(accepts(trial, case) == case.valid for case in cases):
                selected = trial
                trace.append({"minimised_rule": rule.name})
                changed = True
                break

    return SynthesisedProtocol(tuple(selected), tuple(trace))


def evaluate_protocol(
    protocol: Iterable[ProtocolRule],
    cases: list[ProtocolCase],
) -> dict[str, object]:
    rows = [
        {
            "name": case.name,
            "expected": case.valid,
            "observed": accepts(protocol, case),
        }
        for case in cases
    ]
    invalid = [row for row in rows if not row["expected"]]
    valid = [row for row in rows if row["expected"]]
    detection = sum(not row["observed"] for row in invalid) / max(1, len(invalid))
    false_reject = sum(not row["observed"] for row in valid) / max(1, len(valid))
    return {
        "case_count": len(rows),
        "invalid_count": len(invalid),
        "valid_count": len(valid),
        "detection_rate": detection,
        "false_reject_rate": false_reject,
        "accuracy": sum(row["expected"] == row["observed"] for row in rows) / max(1, len(rows)),
        "misses": [row["name"] for row in rows if row["expected"] != row["observed"]],
    }


def random_protocol_baseline(
    seed: int,
    library: list[ProtocolRule],
    size: int,
    cases: list[ProtocolCase],
    trials: int = 256,
) -> dict[str, float]:
    rng = random.Random(seed)
    detections: list[float] = []
    false_rejects: list[float] = []
    size = min(size, len(library))
    for _ in range(trials):
        rules = rng.sample(library, size)
        report = evaluate_protocol(rules, cases)
        detections.append(float(report["detection_rate"]))
        false_rejects.append(float(report["false_reject_rate"]))
    return {
        "trials": trials,
        "detection_median": statistics.median(detections),
        "detection_max": max(detections),
        "false_reject_median": statistics.median(false_rejects),
        "false_reject_min": min(false_rejects),
    }


def run(seed: int = 401) -> dict[str, object]:
    library = rule_library()
    training = build_training_cases(seed * 10_000)
    protocol = synthesise_protocol(training, library)
    frozen_digest = _hash("|".join(protocol.names()))

    # Withheld families are constructed only after the protocol and digest are frozen.
    hidden = build_hidden_cases(seed * 10_000)
    training_report = evaluate_protocol(protocol.selected, training)
    hidden_report = evaluate_protocol(protocol.selected, hidden)
    random_baseline = random_protocol_baseline(
        seed + 99,
        library,
        len(protocol.selected),
        hidden,
    )
    full_report = evaluate_protocol(library, hidden)

    candidate_gate = (
        training_report["accuracy"] == 1.0
        and hidden_report["detection_rate"] == 1.0
        and hidden_report["false_reject_rate"] == 0.0
        and len(protocol.selected) < len(library)
        and hidden_report["detection_rate"] - random_baseline["detection_median"] >= 0.20
        and full_report["accuracy"] == 1.0
    )
    return {
        "status": "protocol_synthesis_candidate" if candidate_gate else "not_yet",
        "claim_scope": (
            "counterexample-guided synthesis selects and minimises executable scientific-validity "
            "invariants from labelled evidence bundles, freezes the protocol, and then evaluates "
            "withheld mutation families; external novelty still requires an independently authored "
            "predicate grammar, outside mutation generator, reproduction and peer review"
        ),
        "seed": seed,
        "library_size": len(library),
        "selected_rule_count": len(protocol.selected),
        "selected_rules": list(protocol.names()),
        "frozen_protocol_digest": frozen_digest,
        "training": training_report,
        "hidden": hidden_report,
        "random_equal_size": random_baseline,
        "full_hand_protocol": full_report,
        "synthesis_trace": list(protocol.trace),
        "candidate_gate": candidate_gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_rule_count": report["selected_rule_count"],
                "library_size": report["library_size"],
                "hidden_detection": report["hidden"]["detection_rate"],
                "hidden_false_reject": report["hidden"]["false_reject_rate"],
                "random_detection_median": report["random_equal_size"]["detection_median"],
                "selected_rules": report["selected_rules"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
