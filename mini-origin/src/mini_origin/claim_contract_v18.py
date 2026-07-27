from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import random
import re
import statistics


CONTRACT_FLOAT_FIELDS = {
    "score_threshold",
    "median_threshold",
    "min_control_gap",
    "median_control_gap",
    "min_ablation_gap",
    "oracle_ceiling",
    "operation_budget",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _canon(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_float(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("contract numbers must be finite")
    if value == 0.0:
        return "0"
    return format(value, ".17g")


def _contract_digest_payload(value: dict[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["required_checks"] = list(payload["required_checks"])
    for name in CONTRACT_FLOAT_FIELDS:
        payload[name] = f"float:{_canonical_float(float(payload[name]))}"
    return payload


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


@dataclass(frozen=True)
class Contract:
    claim_id: str
    candidate_hash: str
    required_runs: int = 5
    min_successes: int = 4
    score_threshold: float = 0.78
    median_threshold: float = 0.82
    min_control_gap: float = 0.08
    median_control_gap: float = 0.10
    min_ablation_gap: float = 0.12
    oracle_ceiling: float = 1.0
    operation_budget: float = 0.20
    required_checks: tuple[str, ...] = (
        "fixed_boundary",
        "bipolar",
        "sealed_holdout",
        "shortcut_resistance",
        "control_present",
    )

    @property
    def digest(self) -> str:
        return _hash(_canon(_contract_digest_payload(asdict(self))))


@dataclass(frozen=True)
class Manifest:
    contract_digest: str
    candidate_hash: str
    seeds: tuple[int, ...]
    commitment: str
    issued_after_freeze: bool

    @property
    def digest(self) -> str:
        value = asdict(self)
        value["seeds"] = list(self.seeds)
        return _hash(_canon(value))


@dataclass(frozen=True)
class Run:
    seed: int
    score: float
    control: float
    ablation: float
    candidate_budget: float
    control_budget: float
    threshold_used: float
    contract_digest: str
    candidate_hash: str
    manifest_digest: str
    holdout_candidates: int
    selected_after_holdout: bool
    holdout_policy_violations: int
    checks: tuple[tuple[str, bool], ...]


@dataclass(frozen=True)
class Bundle:
    claim_id: str
    claimed_breakthrough: bool
    runs: tuple[Run, ...]


def contract() -> Contract:
    return Contract("mini-origin-claim-guard-v18", _hash("frozen-candidate-v18"))


def manifest(spec: Contract, secret: str) -> Manifest:
    seeds = []
    for index in range(spec.required_runs):
        message = f"{spec.digest}:sealed-v1:{index}".encode()
        digest = hmac.new(secret.encode(), message, hashlib.sha256).digest()
        seeds.append(int.from_bytes(digest[:8], "big") % 2_000_000_000 + 1)
    commitment = _hash(
        f"{spec.digest}:sealed-v1:" + ",".join(map(str, seeds))
    )
    return Manifest(spec.digest, spec.candidate_hash, tuple(seeds), commitment, True)


def valid_checks() -> tuple[tuple[str, bool], ...]:
    return tuple((name, True) for name in contract().required_checks)


def valid_bundle(spec: Contract, sealed: Manifest, rng: random.Random) -> Bundle:
    runs = []
    for index, seed in enumerate(sealed.seeds):
        score = 0.84 + 0.015 * math.sin(index + 1) + rng.uniform(-0.004, 0.004)
        runs.append(
            Run(
                seed=seed,
                score=score,
                control=score - 0.12 - rng.uniform(0.0, 0.012),
                ablation=score - 0.16 - rng.uniform(0.0, 0.015),
                candidate_budget=0.20,
                control_budget=0.20,
                threshold_used=spec.score_threshold,
                contract_digest=spec.digest,
                candidate_hash=spec.candidate_hash,
                manifest_digest=sealed.digest,
                holdout_candidates=1,
                selected_after_holdout=False,
                holdout_policy_violations=0,
                checks=valid_checks(),
            )
        )
    return Bundle(spec.claim_id, True, tuple(runs))


def verify(spec: Contract, sealed: Manifest, bundle: Bundle) -> dict[str, object]:
    problems: list[str] = []

    # Intrinsic contract-schema validity. A self-consistent hash cannot make an
    # ill-formed scientific claim contract valid.
    if not spec.claim_id.strip():
        problems.append("contract_empty_claim_id")
    if not SHA256_PATTERN.fullmatch(spec.candidate_hash):
        problems.append("contract_invalid_candidate_hash")
    if spec.required_runs <= 0:
        problems.append("contract_invalid_run_count")
    if not 1 <= spec.min_successes <= max(spec.required_runs, 0):
        problems.append("contract_invalid_success_requirement")
    if not spec.required_checks:
        problems.append("contract_missing_required_checks")
    if any(not name.strip() for name in spec.required_checks):
        problems.append("contract_empty_required_check")
    if len(set(spec.required_checks)) != len(spec.required_checks):
        problems.append("contract_duplicate_required_check")
    if not _finite(
        spec.score_threshold,
        spec.median_threshold,
        spec.min_control_gap,
        spec.median_control_gap,
        spec.min_ablation_gap,
        spec.oracle_ceiling,
        spec.operation_budget,
    ):
        problems.append("contract_non_finite_number")
    if spec.oracle_ceiling <= 0.0:
        problems.append("contract_invalid_oracle_ceiling")
    if not 0.0 <= spec.score_threshold <= spec.oracle_ceiling:
        problems.append("contract_invalid_score_threshold")
    if not 0.0 <= spec.median_threshold <= spec.oracle_ceiling:
        problems.append("contract_invalid_median_threshold")
    if spec.min_control_gap < 0.0 or spec.median_control_gap < 0.0:
        problems.append("contract_negative_control_gap")
    if spec.min_ablation_gap < 0.0:
        problems.append("contract_negative_ablation_gap")
    if spec.operation_budget < 0.0:
        problems.append("contract_negative_operation_budget")

    if sealed.contract_digest != spec.digest:
        problems.append("manifest_contract_mismatch")
    if sealed.candidate_hash != spec.candidate_hash:
        problems.append("manifest_candidate_mismatch")
    if not sealed.issued_after_freeze:
        problems.append("manifest_precedes_freeze")
    if len(sealed.seeds) != spec.required_runs:
        problems.append("manifest_wrong_seed_count")
    if len(set(sealed.seeds)) != len(sealed.seeds):
        problems.append("manifest_duplicate_seed")
    if any(seed <= 0 for seed in sealed.seeds):
        problems.append("manifest_nonpositive_seed")
    expected = _hash(f"{spec.digest}:sealed-v1:" + ",".join(map(str, sealed.seeds)))
    if sealed.commitment != expected:
        problems.append("seed_commitment_mismatch")

    if bundle.claim_id != spec.claim_id:
        problems.append("claim_id_mismatch")
    if len(bundle.runs) != spec.required_runs:
        problems.append("wrong_run_count")
    seeds = [run.seed for run in bundle.runs]
    if len(set(seeds)) != len(seeds):
        problems.append("duplicate_seed")
    if any(seed <= 0 for seed in seeds):
        problems.append("nonpositive_run_seed")
    if set(seeds) != set(sealed.seeds):
        problems.append("seed_set_mismatch")

    scores, control_gaps, ablation_gaps = [], [], []
    successes = 0
    for index, run in enumerate(bundle.runs):
        prefix = f"run_{index}"
        check_names = [name for name, _ in run.checks]
        checks = dict(run.checks)
        if len(set(check_names)) != len(check_names):
            problems.append(f"{prefix}_duplicate_check")
        for name in spec.required_checks:
            if not checks.get(name, False):
                problems.append(f"{prefix}_failed_{name}")
        if run.contract_digest != spec.digest:
            problems.append(f"{prefix}_contract_mismatch")
        if run.candidate_hash != spec.candidate_hash:
            problems.append(f"{prefix}_candidate_mismatch")
        if run.manifest_digest != sealed.digest:
            problems.append(f"{prefix}_manifest_mismatch")
        if not math.isclose(run.threshold_used, spec.score_threshold, abs_tol=1e-12):
            problems.append(f"{prefix}_threshold_changed")
        if run.holdout_candidates != 1 or run.selected_after_holdout:
            problems.append(f"{prefix}_holdout_selection")
        if run.holdout_policy_violations != 0:
            problems.append(f"{prefix}_holdout_policy")
        values = (
            run.score,
            run.control,
            run.ablation,
            run.candidate_budget,
            run.control_budget,
        )
        if not all(math.isfinite(value) for value in values):
            problems.append(f"{prefix}_non_finite")
            continue
        if not 0.0 <= run.score <= spec.oracle_ceiling:
            problems.append(f"{prefix}_outside_oracle")
        if not 0.0 <= run.control <= spec.oracle_ceiling:
            problems.append(f"{prefix}_control_outside_oracle")
        if not 0.0 <= run.ablation <= spec.oracle_ceiling:
            problems.append(f"{prefix}_ablation_outside_oracle")
        if run.candidate_budget < 0.0 or run.control_budget < 0.0:
            problems.append(f"{prefix}_negative_budget")
        if run.candidate_budget > spec.operation_budget + 1e-12:
            problems.append(f"{prefix}_budget_exceeded")
        if run.control_budget > spec.operation_budget + 1e-12:
            problems.append(f"{prefix}_control_budget_exceeded")
        if abs(run.candidate_budget - run.control_budget) > 1e-12:
            problems.append(f"{prefix}_budget_mismatch")
        control_gap = run.score - run.control
        ablation_gap = run.score - run.ablation
        if control_gap < spec.min_control_gap:
            problems.append(f"{prefix}_control_gap")
        if ablation_gap < spec.min_ablation_gap:
            problems.append(f"{prefix}_ablation_gap")
        scores.append(run.score)
        control_gaps.append(control_gap)
        ablation_gaps.append(ablation_gap)
        if (
            run.score >= spec.score_threshold
            and control_gap >= spec.min_control_gap
            and ablation_gap >= spec.min_ablation_gap
        ):
            successes += 1

    median_score = statistics.median(scores) if scores else -math.inf
    median_control = statistics.median(control_gaps) if control_gaps else -math.inf
    minimum_ablation = min(ablation_gaps) if ablation_gaps else -math.inf
    if successes < spec.min_successes:
        problems.append("insufficient_successes")
    if median_score < spec.median_threshold:
        problems.append("median_score")
    if median_control < spec.median_control_gap:
        problems.append("median_control_gap")
    if minimum_ablation < spec.min_ablation_gap:
        problems.append("minimum_ablation_gap")

    accepted = not problems and bundle.claimed_breakthrough
    if bundle.claimed_breakthrough and problems:
        problems.append("false_breakthrough_label")
    return {
        "accepted": accepted,
        "problems": list(dict.fromkeys(problems)),
        "successful_runs": successes,
        "median_score": median_score,
        "median_control_gap": median_control,
        "minimum_ablation_gap": minimum_ablation,
    }
