from pathlib import Path

from mini_origin import compiled_export_v49 as v49
from mini_origin import cost_prior_quotient_v48 as v48
from mini_origin import state_policy_v34 as state


ROOT = Path(__file__).resolve().parents[1]


def toy_task():
    return state.base.make_task(
        "compiled-export-toy",
        ("a", "a-copy", "b"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "y", "y", "x"),
    )


def test_compact_export_preserves_dimensions_and_positive_weights() -> None:
    task = toy_task()
    row = v49.compact_state(
        task,
        task.full_mask,
        (1 << task.query_count) - 1,
        4801,
    )
    assert len(row["labels"]) == task.candidate_count
    assert len(row["masses"]) == task.candidate_count
    assert len(row["query_ids"]) == task.query_count
    assert len(row["costs"]) == task.query_count
    assert len(row["matrix"]) == task.candidate_count
    assert all(len(values) == task.query_count for values in row["matrix"])
    assert min(row["masses"]) > 0
    assert min(row["costs"]) > 0
    assert len(row["digest"]) == 64


def test_export_profile_matches_frozen_weighted_generator() -> None:
    task = toy_task()
    row = v49.compact_state(
        task,
        task.full_mask,
        (1 << task.query_count) - 1,
        4802,
    )
    profile = v48.profile_for_task(task, 4802)
    assert row["masses"] == list(profile.hypothesis_mass)
    assert row["costs"] == list(profile.query_cost)


def test_compiled_solver_is_standalone_source() -> None:
    source = (
        ROOT / "compiled" / "weighted_quotient_v49.cpp"
    ).read_text(encoding="utf-8")
    forbidden = (
        "Python.h",
        "pybind",
        "average_odt_frontier_v44",
        "cost_prior_quotient_v48",
        "LocalWeightedPlanner",
        "PlainWeightedPlanner",
    )
    assert not any(token in source for token in forbidden)
    assert "class ExactSolver" in source
    assert "--self-test" in source
