import json

from mini_origin import numeric_threshold_frontier_v70 as core
from mini_origin import partition_signature_coverage_v84 as coverage


def records(label_shift=0):
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


def test_partition_signature_groups_are_deterministic_and_label_independent():
    first, _ = core.compile_task("signature-synthetic", records(0))
    second, _ = core.compile_task("signature-synthetic", records(3))
    remaining = (1 << first.query_count) - 1
    first_groups = coverage.partition_signature_groups(first, first.full_mask, remaining)
    assert first_groups == coverage.partition_signature_groups(
        first, first.full_mask, remaining
    )
    assert first_groups == coverage.partition_signature_groups(
        second, second.full_mask, remaining
    )


def test_duplicate_signatures_are_grouped_and_masks_keep_complete_classes():
    task, _ = core.compile_task("signature-duplicates", records())
    remaining = (1 << task.query_count) - 1
    groups = coverage.partition_signature_groups(task, task.full_mask, remaining)
    assert any(len(queries) > 1 for _, queries in groups)

    masks = coverage.complete_class_masks(task, task.full_mask, remaining, 1, 0)
    group_sets = [set(queries) for _, queries in groups]
    for mask, representatives in masks:
        selected = {query for query in range(task.query_count) if mask & (1 << query)}
        assert representatives == sum(group <= selected for group in group_sets)
        for group in group_sets:
            assert group <= selected or group.isdisjoint(selected)


def test_contributing_parent_state_set_is_exactly_preserved(monkeypatch):
    task, _ = core.compile_task("preserve-contributor", records())
    expected = [(task.full_mask, (1 << task.query_count) - 1, 6)]
    digest = coverage.parent.state_set_digest(task, expected)

    monkeypatch.setattr(
        coverage.parent,
        "adaptive_select_states",
        lambda _task: (
            expected,
            {
                "selected_states": 1,
                "selected_state_set_digest": digest,
                "selector_revision": "near-small-query-coverage-v83",
            },
        ),
    )
    actual, summary = coverage.select_states(task)
    assert actual == expected
    assert summary["selected_state_set_digest"] == digest
    assert summary["partition_signature_fallback"] is False


def test_preregistration_keeps_thresholds_and_claim_boundary_frozen():
    row = json.loads(coverage.PREREGISTRATION.read_text(encoding="utf-8"))
    assert row["status"] == "preregistered_before_implementation_or_evaluation"
    assert row["development_gate_before_opened_data"]["solver_or_threshold_changes"] == 0
    assert row["negative_result_policy"].startswith("Every failed")
    assert "world-class" in row["claim_boundary"]


def test_configuration_installs_only_the_v84_selector():
    coverage.configure_module()
    assert coverage.conditioned.select_states is coverage.select_states
    assert coverage.frontier.protocol is coverage.protocol
