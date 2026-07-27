import numpy as np

from mini_origin.language_v7 import (
    RankedProgram,
    Scenario,
    binary,
    deployment_programs,
    execute,
    mine_macros,
    primitive_signal_atoms,
    regression_score,
    replace_subtree,
    terminal,
    unary,
)


def test_safe_division_is_finite() -> None:
    expr = binary("div", terminal("teacher"), terminal("c0"))
    context = {
        "weight": np.zeros((3, 4)),
        "teacher": np.ones((3, 1)),
        "pred": np.zeros((3, 1)),
        "peer": 0.0,
        "elig": np.ones((3, 4)),
        "err_ema": np.zeros((3, 1)),
        "abs_ema": np.zeros((3, 1)),
        "momentum": np.zeros((3, 4)),
    }
    result = execute(expr, context)
    assert np.all(np.isfinite(result))
    assert np.max(np.abs(result)) <= 8.0


def test_macro_mining_requires_cross_family_reuse() -> None:
    residual = binary("sub", terminal("teacher"), terminal("pred"))
    scale = binary("add", terminal("c01"), unary("abs", terminal("pred")))
    macro = binary("div", residual, scale)
    program = binary("mul", terminal("elig"), macro)
    ranked = [RankedProgram(program, 0.8, {"x": 0.8})]
    macros = mine_macros(
        {
            "regression": ranked,
            "nonlinear": ranked,
            "bandit": ranked,
        },
        limit=5,
    )
    assert macro.text() in {value.text() for value in macros}


def test_shallow_control_cannot_express_deep_normalizer() -> None:
    programs = deployment_programs(primitive_signal_atoms())
    forbidden = "div(sub(teacher,pred),add(c01,abs(pred)))"
    assert all(forbidden not in program.text() for program in programs)


def test_subtree_ablation_removes_macro() -> None:
    macro = binary("sub", terminal("teacher"), terminal("pred"))
    program = binary("mul", terminal("elig"), macro)
    ablated = replace_subtree(program, macro, terminal("c0"))
    assert ablated.text() == "mul(elig,c0)"


def test_stateful_regression_is_deterministic() -> None:
    program = binary(
        "mul",
        terminal("elig"),
        binary("sub", terminal("teacher"), terminal("pred")),
    )
    scenario = Scenario(
        "regression",
        17,
        4,
        10.0,
        8,
        12,
        0.01,
        0.10,
        0.10,
        0.0,
        0,
    )
    assert regression_score(program, scenario) == regression_score(program, scenario)
