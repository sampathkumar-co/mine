import json
from pathlib import Path

from mini_origin import conditioned_cell_frontier_v60 as conditioned
from mini_origin import label_free_frontier_v72 as frontier
from mini_origin import label_free_selector_certificate_v71 as certificate
from mini_origin import numeric_threshold_repaired_v70 as numeric


def test_v72_combines_both_label_free_sampling_repairs():
    frontier.configure_module()
    first = certificate.synthetic_records(72001, False)
    second = certificate.synthetic_records(72001, True)

    first_sample = numeric.label_free_sample("v72-pair", first)
    second_sample = numeric.label_free_sample("v72-pair", second)
    assert [features for features, _ in first_sample] == [
        features for features, _ in second_sample
    ]

    first_task, _ = numeric.compile_task("v72-pair", first)
    second_task, _ = numeric.compile_task("v72-pair", second)
    assert first_task.rows == second_task.rows
    assert first_task.query_count == second_task.query_count

    first_cells = tuple(sorted(conditioned.conditioned_cells(first_task)))
    second_cells = tuple(sorted(conditioned.conditioned_cells(second_task)))
    assert first_cells == second_cells


def test_v72_selected_states_are_label_independent():
    frontier.configure_module()
    first = certificate.synthetic_records(72002, False)
    second = certificate.synthetic_records(72002, True)
    first_task, _ = numeric.compile_task("v72-states", first)
    second_task, _ = numeric.compile_task("v72-states", second)

    first_selected, first_summary = conditioned.select_states(first_task)
    second_selected, second_summary = conditioned.select_states(second_task)
    assert tuple(first_selected) == tuple(second_selected)
    assert first_summary == second_summary


def test_v72_keeps_v70_scientific_gate_unchanged():
    root = Path(__file__).resolve().parents[1]
    v70 = json.loads(
        (root / "campaigns" / "v70-numeric-threshold-frontier-preregistration.json")
        .read_text(encoding="utf-8")
    )
    v72 = json.loads(
        (root / "campaigns" / "v72-label-free-frontier-revalidation.json")
        .read_text(encoding="utf-8")
    )
    assert v72["locked_gate"] == v70["locked_gate"]
    protocol = frontier.protocol()
    assert "labels excluded" in protocol["record_sampler"]
    assert "labels excluded" in protocol["state_selector"]
