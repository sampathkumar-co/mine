from mini_origin.language_v6 import binary, terminal, unary
from mini_origin.operator_transfer_v7 import (
    Template,
    TemporalScenario,
    _skeleton,
    hand_temporal_delta,
    instantiate,
    temporal_score,
)


def _source_delta():
    residual = binary(
        "add",
        terminal("teacher"),
        unary("neg", terminal("pred")),
    )
    return binary("mul", terminal("elig"), residual)


def test_skeleton_abstracts_signal_names() -> None:
    skeleton, variables = _skeleton(_source_delta())
    assert skeleton.text() == "mul(v0,add(v1,neg(v2)))"
    assert variables == 3


def test_template_instantiates_new_temporal_roles() -> None:
    skeleton, variables = _skeleton(_source_delta())
    template = Template(
        expression=skeleton,
        variables=variables,
        source_families=("regression", "bandit"),
        score=1.0,
    )
    instantiated = instantiate(
        template,
        (
            terminal("trace"),
            terminal("future"),
            terminal("prediction"),
        ),
    )
    assert instantiated.text() == "mul(trace,add(future,neg(prediction)))"


def test_closed_source_rule_cannot_access_temporal_signals() -> None:
    scenario = TemporalScenario(
        seed=7_101,
        dimension=6,
        cells=28,
        train_steps=180,
        switches=2,
        noise=0.02,
        dropout=0.20,
        damage=0.35,
    )
    transferred = temporal_score(hand_temporal_delta(), scenario)
    closed = temporal_score(_source_delta(), scenario)
    assert transferred > closed + 0.20


def test_temporal_delta_survives_harder_transfer_case() -> None:
    scenario = TemporalScenario(
        seed=7_103,
        dimension=9,
        cells=36,
        train_steps=300,
        switches=3,
        noise=0.045,
        dropout=0.30,
        damage=0.50,
    )
    score = temporal_score(hand_temporal_delta(), scenario)
    assert score > 0.65
