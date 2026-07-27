import numpy as np

from mini_origin import coded_memory_v10 as v10
from mini_origin import coded_memory_v11 as v11


def _scenarios():
    return [
        v10.CodeScenario(301, 5, 7, 2.6, 50, 0.025, 0.48),
        v10.CodeScenario(303, 6, 8, 2.7, 55, 0.030, 0.52),
    ]


def _evaluation(
    scenario: v10.CodeScenario,
    *,
    write_fraction: float,
    surviving_rank: int,
    post_damage: float = 0.90,
) -> v10.CodeEvaluation:
    return v10.CodeEvaluation(
        score=0.90,
        pre_damage=0.95,
        post_damage=post_damage,
        retention=post_damage / 0.95,
        write_fraction=write_fraction,
        surviving_rank=surviving_rank,
    )


def test_feasible_ranker_enforces_write_and_rank_constraints(monkeypatch) -> None:
    valid = v10.CodeProgram(0.25, False, "rademacher", False, 1e-6, 0.22, 1)
    over_budget = v10.CodeProgram(0.25, False, "rademacher", False, 1e-6, 0.22, 2)
    rank_deficient = v10.CodeProgram(0.25, False, "rademacher", False, 1e-6, 0.22, 3)

    def fake_evaluate(program: v10.CodeProgram, scenario: v10.CodeScenario):
        if program.seed_salt == 2:
            return _evaluation(
                scenario,
                write_fraction=v11.WRITE_LIMIT + 0.01,
                surviving_rank=scenario.contexts,
            )
        if program.seed_salt == 3:
            return _evaluation(
                scenario,
                write_fraction=v11.WRITE_LIMIT - 0.01,
                surviving_rank=scenario.contexts - 1,
            )
        return _evaluation(
            scenario,
            write_fraction=v11.WRITE_LIMIT - 0.01,
            surviving_rank=scenario.contexts,
        )

    monkeypatch.setattr(v10, "evaluate_program", fake_evaluate)
    monkeypatch.setattr(
        v11,
        "_evaluate_expander",
        lambda scenario: _evaluation(
            scenario,
            write_fraction=0.50,
            surviving_rank=scenario.contexts,
            post_damage=0.88,
        ),
    )

    ranked = v11.rank_feasible(
        [valid, over_budget, rank_deficient],
        _scenarios(),
        limit=12,
    )
    assert [item.program.seed_salt for item in ranked] == [1]


def test_specialist_expander_is_sparse_and_well_formed() -> None:
    program = v11.specialist_expander_program()
    for scenario in _scenarios():
        matrix = v10.make_code_matrix(program, scenario)
        evaluation = v11._evaluate_expander(scenario)
        assert matrix.shape[1] == scenario.contexts
        assert np.all(np.isfinite(matrix))
        assert 0.0 < evaluation.write_fraction <= 0.55
        assert 0.0 <= evaluation.pre_damage <= 1.0
        assert 0.0 <= evaluation.post_damage <= 1.0


def test_hard_constraint_excludes_dense_program() -> None:
    dense = v10.CodeProgram(1.0, False, "gaussian", True, 1e-6, 0.22, 99)
    assert not v11.rank_feasible([dense], _scenarios(), 1)
