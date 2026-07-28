from __future__ import annotations

import inspect

from mini_origin import support_robust_v27 as v27


def test_exact_worlds_cover_low_support_before_hidden_evaluation() -> None:
    worlds = v27.robust_exact_worlds()
    assert min(world.rho for world in worlds) == 0.24
    assert max(world.dimension for world in worlds) == 9
    assert all(
        sum(world.root == root and world.dimension == dimension for world in worlds)
        == 4
        for dimension in range(2, 10)
        for root in range(dimension)
    )


def test_development_uses_multiple_locked_support_strata() -> None:
    strata = v27.development_strata()
    assert [stratum.name for stratum in strata] == [
        "low-coupling",
        "transition",
        "medium",
        "high",
    ]
    assert min(stratum.rho_low for stratum in strata) == 0.24
    assert {stratum.replicates for stratum in strata} == {256, 384, 512}


def test_search_optimises_worst_stratum_without_step_labels() -> None:
    source = inspect.getsource(v27.search_support_robust_programs)
    assert ".action" not in source
    assert "best_minimum" in source
    assert "maximise_worst_support_stratum_then_mean" in source
    assert "terminal_root_success_only" in source


def test_hidden_strata_are_larger_and_created_after_freeze() -> None:
    assert v27.hidden_strata()[0].dimensions == (17, 31, 63, 127, 255)
    source = inspect.getsource(v27.run)
    freeze = source.index("frozen_digest = digest(selected)")
    hidden = source.index("strata = hidden_strata()")
    evaluation = source.index("candidate = evaluate_strata")
    assert freeze < hidden < evaluation


def test_gate_preserves_v26_thresholds_and_adds_robustness() -> None:
    source = inspect.getsource(v27.run)
    assert "development.minimum_accuracy >= 0.985" in source
    assert "candidate.minimum_accuracy >= 0.985" in source
    assert "specialist_gap >= -0.01" in source
    assert "single_gap >= 0.20" in source
    assert "query_gap >= 0.20" in source
    assert "random_gap >= 0.45" in source


def test_specialist_is_exact_on_robust_training_worlds() -> None:
    from mini_origin.terminal_reward_v26 import run_noiseless_trial, specialist_program

    program = specialist_program()
    assert all(
        run_noiseless_trial(program, world)
        for world in v27.robust_exact_worlds()
    )
