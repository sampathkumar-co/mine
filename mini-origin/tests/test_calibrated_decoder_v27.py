from __future__ import annotations

import inspect

from mini_origin import calibrated_decoder_v27 as v27


def test_threshold_candidates_cover_every_empirical_interval() -> None:
    examples = [
        v27.DecisionExample(-0.1, True),
        v27.DecisionExample(0.1, True),
        v27.DecisionExample(0.5, False),
    ]
    candidates = v27.threshold_candidates(examples)
    assert len(candidates) == 4
    assert candidates[0] < -0.1
    assert 0.0 in candidates
    assert 0.3 in candidates
    assert candidates[-1] > 0.5


def test_fit_threshold_prefers_balanced_class_accuracy() -> None:
    training = [
        v27.DecisionExample(-0.2, True),
        v27.DecisionExample(0.0, True),
        v27.DecisionExample(0.5, False),
        v27.DecisionExample(0.7, False),
    ]
    development = [
        v27.DecisionExample(-0.1, True),
        v27.DecisionExample(0.1, True),
        v27.DecisionExample(0.45, False),
        v27.DecisionExample(0.8, False),
    ]
    fit = v27.fit_threshold(training, development)
    assert 0.1 < fit.threshold < 0.5
    assert fit.development_move_accuracy == 1.0
    assert fit.development_accept_accuracy == 1.0


def test_hidden_weak_signal_tasks_follow_freeze() -> None:
    source = inspect.getsource(v27.run)
    freeze = source.index("frozen_digest = digest")
    hidden = source.index("hidden_tasks = v26.make_tasks")
    evaluation = source.index("candidate_result = evaluate")
    assert freeze < hidden < evaluation
    assert "rho_low=0.15" in source


def test_gate_requires_advantage_over_fixed_threshold() -> None:
    source = inspect.getsource(v27.run)
    assert "fixed_gap >= 0.01" in source
    assert "low_signal_gap >= 0.03" in source
    assert "candidate_result.low_signal_accuracy >= 0.95" in source
    assert "fit.candidates_checked >= 100" in source
