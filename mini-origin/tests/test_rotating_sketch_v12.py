import numpy as np

from mini_origin.rotating_sketch_v12 import (
    WRITE_LIMIT,
    SketchProgram,
    SketchScenario,
    collect_statistics,
    dense_program,
    evaluate_program,
    hand_rotating_program,
    selected_cells,
    static_program,
)


def _scenario() -> SketchScenario:
    return SketchScenario(
        seed=401,
        contexts=4,
        dimension=5,
        redundancy=3.0,
        examples_per_context=24,
        condition=12.0,
        noise=0.02,
        damage_fraction=0.50,
    )


def test_every_sparse_schedule_respects_per_example_write_budget() -> None:
    scenario = _scenario()
    for schedule in ("iid", "cyclic", "balanced", "affine", "antithetic"):
        program = SketchProgram(
            density=WRITE_LIMIT,
            schedule=schedule,
            stride=5,
            context_stride=3,
            phase_stride=7,
            ridge=1e-5,
            seed_salt=17,
        )
        expected_max = int(np.floor(WRITE_LIMIT * scenario.cells))
        for context in range(scenario.contexts):
            for occurrence in range(12):
                selected = selected_cells(program, scenario, context, occurrence)
                assert 1 <= len(selected) <= expected_max
                assert len(np.unique(selected)) == len(selected)


def test_balanced_rotation_eventually_covers_every_cell() -> None:
    scenario = _scenario()
    program = hand_rotating_program()
    statistics, _ = collect_statistics(program, scenario)
    per_context_coverage = np.count_nonzero(statistics.writes, axis=0)
    assert np.all(per_context_coverage == scenario.cells)


def test_dense_and_sparse_write_accounting_is_explicit() -> None:
    scenario = _scenario()
    dense = evaluate_program(dense_program(), scenario)
    rotating = evaluate_program(hand_rotating_program(), scenario)
    assert dense.write_fraction == 1.0
    assert rotating.write_fraction <= WRITE_LIMIT


def test_static_schedule_reuses_the_same_cells() -> None:
    scenario = _scenario()
    program = static_program()
    first = selected_cells(program, scenario, context=0, occurrence=0)
    later = selected_cells(program, scenario, context=0, occurrence=17)
    assert np.array_equal(first, later)


def test_evaluation_is_deterministic_and_bounded() -> None:
    scenario = _scenario()
    program = hand_rotating_program()
    first = evaluate_program(program, scenario)
    second = evaluate_program(program, scenario)
    assert first == second
    assert 0.0 <= first.post_damage <= 1.0
    assert 0.0 <= first.retention <= 1.1
