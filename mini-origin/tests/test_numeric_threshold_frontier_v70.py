from decimal import Decimal

from mini_origin import numeric_threshold_frontier_v70 as numeric


def test_quantile_thresholds_follow_frozen_rank_rule():
    values = [Decimal(index) for index in range(1, 17)]
    assert numeric.quantile_thresholds(values) == (
        Decimal(2),
        Decimal(4),
        Decimal(6),
        Decimal(8),
        Decimal(10),
        Decimal(12),
        Decimal(14),
    )


def test_high_cardinality_numeric_column_becomes_thresholds():
    records = [
        ((str(index), "A" if index % 2 else "B"), str(index % 2))
        for index in range(1, 17)
    ]
    task, summary = numeric.compile_task("numeric", records)
    assert summary["original_features"] == 2
    assert summary["threshold_columns"] == 1
    assert summary["threshold_queries"] == 7
    assert summary["exact_columns"] == 1
    assert summary["compiled_queries"] == 8
    assert task.query_count == 8
    assert all(value in {"le", "gt"} for row in task.rows for value in row[:7])


def test_low_cardinality_numeric_column_remains_exact():
    records = [
        ((str(index % 3),), str(index % 2))
        for index in range(20)
    ]
    task, summary = numeric.compile_task("low-cardinality", records)
    assert summary["threshold_columns"] == 0
    assert summary["exact_columns"] == 1
    assert task.query_count == 1
    assert {row[0] for row in task.rows} == {"0", "1", "2"}


def test_missing_numeric_values_remain_explicit():
    records = [
        (("?" if index == 0 else str(index),), str(index % 2))
        for index in range(17)
    ]
    task, summary = numeric.compile_task("missing", records)
    assert summary["threshold_queries"] == 7
    assert any("missing" in row for row in task.rows)


def test_labels_do_not_change_compiled_responses():
    features = [(str(index), str(index % 4)) for index in range(16)]
    first = [(row, "left") for row in features]
    second = [
        (row, "right" if index % 3 else "other")
        for index, row in enumerate(features)
    ]
    first_task, first_summary = numeric.compile_task("label-free", first)
    second_task, second_summary = numeric.compile_task("label-free", second)
    assert first_task.query_count == second_task.query_count
    assert first_task.rows == second_task.rows
    assert first_summary["compiled_queries"] == second_summary["compiled_queries"]
