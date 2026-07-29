import json

from mini_origin import small_query_coverage_v79 as coverage
from mini_origin import numeric_threshold_frontier_v70 as core


def small_records(label_shift=0):
    rows = []
    for index in range(384):
        features = (
            str(index % 9),
            str((index // 9) % 7),
            "M" if index % 2 else "F",
            str(index % 3),
        )
        rows.append((features, str((index + label_shift) % 6)))
    return rows


def normal_records():
    rows = []
    for index in range(384):
        features = tuple(str((index * (column + 3)) % 31) for column in range(5))
        rows.append((features, str(index % 3)))
    return rows

def test_effective_limits_make_conditioned_small_tasks_feasible():
    coverage.configure_module()
    task, _ = core.compile_task("small-limit-test", small_records())
    assert task.query_count == 10
    assert coverage.effective_limits(task) == (9, 3)


def test_small_query_selector_is_deterministic():
    coverage.configure_module()
    task, _ = core.compile_task("small-coverage-test", small_records())
    first, summary = coverage.adaptive_select_states(task)
    second, _ = coverage.adaptive_select_states(task)
    assert summary["adaptive_small_query_mode"] is True
    assert first == second
    assert summary["selected_state_set_digest"] == coverage.parent_state_set_digest(
        task, first
    )


def test_small_query_selection_is_label_independent():
    coverage.configure_module()
    first, _ = core.compile_task("small-label-free", small_records(0))
    second, _ = core.compile_task("small-label-free", small_records(3))
    assert coverage.adaptive_select_states(first)[0] == coverage.adaptive_select_states(second)[0]

def test_normal_query_selector_is_exactly_unchanged():
    coverage.configure_module()
    task, _ = core.compile_task("normal-selector-test", normal_records())
    assert task.query_count > coverage.SMALL_QUERY_LIMIT
    expected, _ = coverage._PARENT_SELECT_STATES(task)
    actual, summary = coverage.adaptive_select_states(task)
    assert actual == expected
    assert summary["adaptive_small_query_mode"] is False
    assert summary["selected_state_set_digest"] == coverage.parent_state_set_digest(
        task, expected
    )


def test_preregistration_declares_opened_data_only():
    row = json.loads(coverage.PREREGISTRATION.read_text(encoding="utf-8"))
    assert row["fresh_blind_claim"] is False
    assert row["exact_solver_revisions"] == 0
    assert row["compiler_revisions"] == 0
    assert row["selector_revisions"] == 1
    assert row["adaptive_selector"]["dataset_specific_exceptions"] is False


def test_parent_reconfiguration_keeps_adaptive_selector_installed():
    coverage.configure_module()
    coverage.parent.frontier.configure_module()
    assert coverage.conditioned.select_states is coverage.adaptive_select_states
    assert coverage.frontier.protocol is coverage.protocol
    assert coverage.frontier.compact_state is coverage.compact_state
