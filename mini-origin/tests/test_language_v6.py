import numpy as np

from mini_origin.language_v6 import (
    Expr,
    Scenario,
    _hand_delta,
    _hand_hebb,
    base_atoms,
    execute,
    regression_score,
    shallow_programs,
    task_specific_programs,
)


def test_expression_execution_constructs_error_credit() -> None:
    program = _hand_delta()
    context = {
        "teacher": np.array([[1.0], [0.5]]),
        "pred": np.array([[0.25], [0.75]]),
        "peer": 0.0,
        "elig": np.array([[2.0, -1.0], [1.0, 3.0]]),
        "weight": np.zeros((2, 2)),
    }
    result = execute(program, context)
    expected = np.array([[1.5, -0.75], [-0.25, -0.75]])
    assert np.allclose(result, expected)


def test_shallow_base_language_cannot_spell_full_delta_program() -> None:
    texts = {program.text() for program in shallow_programs(base_atoms())}
    assert _hand_delta().text() not in texts


def test_deep_task_specific_language_contains_delta_program() -> None:
    texts = {program.text() for program in task_specific_programs()}
    assert _hand_delta().text() in texts


def test_delta_beats_hebb_on_ill_conditioned_hidden_queries() -> None:
    scenario = Scenario(
        "regression",
        9123,
        6,
        32.0,
        32,
        150,
        0.04,
        0.30,
        0.50,
    )
    delta = regression_score(_hand_delta(), scenario)
    hebb = regression_score(_hand_hebb(), scenario)
    assert delta > hebb + 0.12


def test_expr_text_is_stable() -> None:
    value = Expr("mul", (Expr("elig"), Expr("teacher")))
    assert value.text() == "mul(elig,teacher)"
    assert value.size() == 3
