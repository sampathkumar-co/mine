from __future__ import annotations

import json

from mini_origin import conditioned_cell_frontier_v60 as conditioned
from mini_origin import label_free_numeric_frontier_v72 as v72
from mini_origin import label_free_selector_certificate_v71 as selector
from mini_origin import numeric_threshold_frontier_v70 as original
from mini_origin import numeric_threshold_repaired_v70 as repaired


def test_v72_uses_both_label_free_sampling_stages() -> None:
    v72.configure()
    assert conditioned.sample_allowed is selector.label_free_sample_allowed
    assert original.external.deterministic_sample is repaired.label_free_sample
    protocol = v72.compiler_protocol()
    assert protocol["labels_or_costs_used"] is False
    assert "labels excluded" in protocol["sampling"]
    assert "labels excluded" in protocol["oversized_cell_sampling"]


def test_v72_gate_is_identical_to_v70_gate() -> None:
    preregistration, parent, evidence = v72.verify_frozen_inputs()
    v70 = json.loads(original.PREREGISTRATION.read_text(encoding="utf-8"))
    assert preregistration["locked_gate"] == v70["locked_gate"]
    assert preregistration["exact_budget"] == v70["exact_budget"]
    assert parent["development_gate"] is False
    assert evidence["development_gate"] is True


def test_relabeling_cannot_change_compilation_or_selection() -> None:
    name = "v72-focused-label-independence"
    first = []
    second = []
    for index in range(420):
        features = (
            f"{(index * 17) % 1009 / 37:.6f}",
            str(index % 5),
            "?" if index % 61 == 0 else f"{(index * index) % 2003 / 43:.6f}",
            chr(ord("A") + index % 4),
        )
        first.append((features, f"L{index % 3}"))
        second.append((features, f"R{(index * 7) % 5}"))

    v72.configure()
    first_sample = repaired.label_free_sample(name, first)
    second_sample = repaired.label_free_sample(name, second)
    assert tuple(row[0] for row in first_sample) == tuple(row[0] for row in second_sample)

    first_task, _ = repaired.compile_task(name, first)
    second_task, _ = repaired.compile_task(name, second)
    assert first_task.rows == second_task.rows
    assert first_task.query_count == second_task.query_count

    first_selected, first_summary = conditioned.select_states(first_task)
    second_selected, second_summary = conditioned.select_states(second_task)
    assert first_selected == second_selected
    assert first_summary == second_summary
