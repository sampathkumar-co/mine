from mini_origin import attainable_envelope_v38 as v38
from mini_origin import exact_tail_v36 as v36
from mini_origin import safe_portfolio_v37 as v37
from mini_origin import state_policy_v34 as v34


def toy_task():
    return v34.base.make_task(
        "toy-envelope",
        ("a", "b", "c"),
        (
            ("0", "0", "0"),
            ("0", "0", "1"),
            ("0", "1", "0"),
            ("1", "0", "0"),
            ("1", "1", "0"),
            ("1", "1", "1"),
        ),
        ("x", "x", "y", "y", "z", "z"),
    )


def test_baseline_is_one_real_specialist() -> None:
    task = toy_task()
    rows = v38.constant_rows(task)
    selected = v38.choose_attainable(rows)
    assert selected in rows
    assert selected.metric() == max(row.metric() for row in rows)


def test_repaired_envelope_never_loses_to_real_baseline() -> None:
    tasks = [
        toy_task(),
        v34.base.load_monk(1, "train"),
        v34.base.load_zoo(),
    ]
    for task in tasks:
        baseline = v38.choose_attainable(v38.constant_rows(task))
        exact = v36.ExactPlanner(task)
        fallbacks = {
            objective: v37.FallbackPlanner(task, objective)
            for objective in v34.OBJECTIVE_NAMES
        }
        row = v38.compare_task(
            task,
            12,
            12,
            exact,
            fallbacks,
            baseline,
        )
        assert row["lex_no_harm"]
        assert row["coordinate_certificate"]


def test_selection_score_prefers_attainable_metrics() -> None:
    better = v38.AttainableRow(
        "entropy", 1.0, 2.0, 3, 10, 4
    )
    worse = v38.AttainableRow(
        "gini", 1.0, 2.1, 3, 10, 0
    )
    assert v38.choose_attainable((worse, better)) == better
