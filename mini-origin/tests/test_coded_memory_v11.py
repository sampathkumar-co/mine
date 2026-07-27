import numpy as np

from mini_origin import coded_memory_v10 as v10
from mini_origin.coded_memory_v11 import (
    WRITE_LIMIT,
    _evaluate_expander,
    constrained_programs,
    rank_feasible,
    specialist_expander_program,
)


def _scenarios():
    return [
        v10.CodeScenario(301, 5, 7, 2.6, 50, 0.025, 0.48),
        v10.CodeScenario(303, 6, 8, 2.7, 55, 0.030, 0.52),
    ]


def test_feasible_ranker_enforces_write_and_rank_constraints() -> None:
    rng = np.random.default_rng(11)
    ranked = rank_feasible(constrained_programs(rng, 220), _scenarios(), 12)
    assert ranked
    for item in ranked:
        for evaluation, scenario in zip(item.evaluations.values(), _scenarios()):
            assert evaluation.write_fraction <= WRITE_LIMIT + 1e-12
            assert evaluation.surviving_rank == scenario.contexts


def test_specialist_expander_is_sparse_and_well_formed() -> None:
    program = specialist_expander_program()
    for scenario in _scenarios():
        matrix = v10.make_code_matrix(program, scenario)
        evaluation = _evaluate_expander(scenario)
        assert np.linalg.matrix_rank(matrix) == scenario.contexts
        assert evaluation.write_fraction <= 0.55
        assert evaluation.pre_damage > 0.75
        assert 0.0 <= evaluation.post_damage <= 1.0


def test_hard_constraint_excludes_dense_program() -> None:
    dense = v10.CodeProgram(1.0, False, "gaussian", True, 1e-6, 0.22, 99)
    assert not rank_feasible([dense], _scenarios(), 1)
