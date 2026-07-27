import numpy as np

from mini_origin import state_invention_v8 as v8
from mini_origin import mechanism_quotient_v17 as v17
from mini_origin import mechanism_quotient_v17_1 as v171


def test_signature_is_deterministic() -> None:
    program = v8.hand_dynamic_program()
    first = v17.behaviour_signature(program)
    second = v17.behaviour_signature(program)
    assert np.allclose(first.values, second.values)
    assert first.phase_slots == second.phase_slots


def test_known_hand_mechanism_is_rejected() -> None:
    candidates, metadata = v17.quotient_candidates(
        [v8.hand_dynamic_program()],
        minimum_known_distance=0.16,
    )
    assert candidates == []
    assert metadata["rejected_as_known"] == 1


def test_duplicate_behaviour_collapses_to_one_class() -> None:
    rng = np.random.default_rng(4)
    program = v8.dynamic_programs(rng, count=1)[0]
    candidates, _ = v17.quotient_candidates(
        [program, program],
        minimum_known_distance=-1.0,
    )
    assert len(candidates) == 1


def test_corrected_runner_freezes_before_hidden() -> None:
    source = open(
        "src/mini_origin/mechanism_quotient_v17_1.py",
        encoding="utf-8",
    ).read()
    freeze = source.index("development_score, best, development_scores = ranked[0]")
    hidden = source.index("hidden = v17._hidden_scenarios")
    assert freeze < hidden
    assert 'metadata["hidden_candidates_evaluated"] = 1' in source
    assert "hidden_ranked" not in source
