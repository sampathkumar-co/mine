import random

from mini_origin.claim_contract_v18 import (
    contract,
    manifest,
    valid_bundle,
    verify,
)
from mini_origin.claim_guard_benchmark_v18 import (
    MUTATIONS,
    historical_replay,
    mutate,
    red_team,
)


def test_valid_claim_passes() -> None:
    spec = contract()
    sealed = manifest(spec, "test-secret")
    bundle = valid_bundle(spec, sealed, random.Random(3))
    verdict = verify(spec, sealed, bundle)
    assert verdict["accepted"]
    assert verdict["problems"] == []


def test_every_integrity_mutation_is_rejected() -> None:
    spec = contract()
    sealed = manifest(spec, "mutation-secret")
    base = valid_bundle(spec, sealed, random.Random(4))
    for kind in MUTATIONS:
        changed = mutate(kind, spec, sealed, base)
        verdict = verify(*changed)
        assert not verdict["accepted"], kind
        assert verdict["problems"], kind


def test_sealed_seeds_are_deterministic_and_secret_scoped() -> None:
    spec = contract()
    first = manifest(spec, "secret-a")
    second = manifest(spec, "secret-a")
    third = manifest(spec, "secret-b")
    assert first.seeds == second.seeds
    assert first.commitment == second.commitment
    assert first.seeds != third.seeds


def test_historical_replay_is_exact() -> None:
    report = historical_replay()
    assert report["all_correct"]
    assert report["accuracy"] == 1.0
    assert report["case_count"] == 15


def test_claim_guard_beats_simple_gates() -> None:
    report = red_team(211, repeats=2, valid_trials=8)
    assert report["claim_guard_pass"]
    assert report["detection_rate"]["claim_guard"] == 1.0
    assert report["false_reject_rate"]["claim_guard"] == 0.0
    best_simple = max(
        value
        for name, value in report["detection_rate"].items()
        if name != "claim_guard"
    )
    assert report["detection_rate"]["claim_guard"] - best_simple >= 0.30
