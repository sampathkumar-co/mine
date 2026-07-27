from dataclasses import replace

from mini_origin import protocol_synth_v19 as v19
from mini_origin import protocol_synth_v19_1 as v191
from mini_origin.claim_contract_v18 import contract, manifest, valid_bundle


def test_total_predicates_reject_malformed_zero_run_contract() -> None:
    spec = replace(contract(), required_runs=0, min_successes=0)
    sealed = manifest(spec, "zero-run")
    case = v19.ProtocolCase(
        "zero-run",
        False,
        spec,
        sealed,
        valid_bundle(spec, sealed, __import__("random").Random(1)),
    )
    assert not v191.safe_accepts(v19.rule_library(), case)


def test_synthesis_fits_frozen_training_cases() -> None:
    training = v19.build_training_cases(19001)
    protocol = v19.synthesise_protocol(training, v19.rule_library())
    report = v19.evaluate_protocol(protocol.selected, training)
    assert report["accuracy"] == 1.0
    assert report["false_reject_rate"] == 0.0
    assert protocol.selected


def test_protocol_is_minimal_on_training_set() -> None:
    training = v19.build_training_cases(19002)
    protocol = v19.synthesise_protocol(training, v19.rule_library())
    for rule in protocol.selected:
        reduced = tuple(item for item in protocol.selected if item != rule)
        report = v19.evaluate_protocol(reduced, training)
        assert report["accuracy"] < 1.0


def test_hidden_cases_are_constructed_after_freeze_in_runner() -> None:
    source = open(
        "src/mini_origin/protocol_synth_v19.py",
        encoding="utf-8",
    ).read()
    freeze = source.index("frozen_digest = _hash")
    hidden = source.index("hidden = build_hidden_cases")
    assert freeze < hidden


def test_seeded_protocol_synthesis_is_deterministic() -> None:
    first = v191.run(409)
    second = v191.run(409)
    assert first["selected_rules"] == second["selected_rules"]
    assert first["frozen_protocol_digest"] == second["frozen_protocol_digest"]
    assert first["hidden"] == second["hidden"]
