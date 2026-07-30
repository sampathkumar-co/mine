from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace

from mini_origin import medium_small_query_coverage_v83 as coverage


def test_preregistration_freezes_opened_development_boundary():
    row = json.loads(coverage.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert row["status"] == "opened_data_development_preregistration"
    assert row["parent_v82_commit"] == coverage.FROZEN_V82_COMMIT
    assert row["fresh_blind_claim"] is False
    assert row["exact_solver_revisions"] == 0
    assert row["compiler_revisions"] == 0
    assert row["selector_revisions"] == 1
    assert row["fallback_selector"]["dataset_specific_exceptions"] is False
    assert row["fallback_selector"]["label_or_cost_use"] is False


def test_parent_evidence_is_exact_rejection():
    payload = coverage.V82_EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == coverage.V82_EVIDENCE_SHA256
    row = json.loads(payload.decode("utf-8"))
    assert row["status"] == "pmlb_cross_source_blind_rejected_v82"
    assert row["evidence_digest"] == coverage.V82_EVIDENCE_DIGEST
    assert row["rust_mismatch_count"] == 0
    assert row["adapter_verification_passed"] is True


def test_fallback_limits_cover_only_medium_small_queries():
    assert coverage.fallback_limits(SimpleNamespace(query_count=11)) == (7, 2)
    assert coverage.fallback_limits(SimpleNamespace(query_count=12)) == (8, 2)
    assert coverage.fallback_limits(SimpleNamespace(query_count=16)) == (10, 4)


def test_fallback_sampler_does_not_read_labels():
    first = SimpleNamespace(
        name="synthetic",
        rows=(("a",), ("b",), ("c",), ("d",)),
        labels=("0", "0", "1", "1"),
    )
    second = SimpleNamespace(
        name="synthetic",
        rows=first.rows,
        labels=("x", "y", "z", "w"),
    )
    cell = (1 << 4) - 1
    assert coverage.label_free_sample_allowed(first, cell, 3, "salt") == (
        coverage.label_free_sample_allowed(second, cell, 3, "salt")
    )


def test_nonzero_parent_selection_is_preserved(monkeypatch):
    expected = [(3, 5, 2)]
    summary = {"selected_states": 1, "selected_state_set_digest": "frozen"}
    monkeypatch.setattr(
        coverage.v79,
        "adaptive_select_states",
        lambda task: (expected, summary),
    )
    rows, actual = coverage.select_states(SimpleNamespace(query_count=12))
    assert rows is expected
    assert actual["selected_state_set_digest"] == "frozen"
    assert actual["medium_small_zero_candidate_fallback"] is False


def test_zero_parent_outside_range_is_not_repaired(monkeypatch):
    summary = {"selected_states": 0, "selected_state_set_digest": "empty"}
    monkeypatch.setattr(
        coverage.v79,
        "adaptive_select_states",
        lambda task: ([], summary),
    )
    rows, actual = coverage.select_states(SimpleNamespace(query_count=10))
    assert rows == []
    assert actual["medium_small_zero_candidate_fallback"] is False


def test_zero_parent_inside_range_uses_generic_fallback(monkeypatch):
    parent_summary = {"selected_states": 0}
    fallback_rows = [(7, 11, 2)]
    fallback_summary = {
        "selected_states": 1,
        "medium_small_zero_candidate_fallback": True,
    }
    monkeypatch.setattr(
        coverage.v79,
        "adaptive_select_states",
        lambda task: ([], parent_summary),
    )
    monkeypatch.setattr(
        coverage,
        "fallback_select_states",
        lambda task: (fallback_rows, fallback_summary),
    )
    rows, actual = coverage.select_states(SimpleNamespace(query_count=12))
    assert rows is fallback_rows
    assert actual is fallback_summary


def test_fallback_implementation_has_no_dataset_branch():
    source = inspect.getsource(coverage.fallback_select_states)
    assert coverage.SOLAR_TASK not in source
    assert "task.name ==" not in source
    assert "task.name in" not in source


def test_protocol_declares_narrow_label_free_fallback():
    row = coverage.protocol()
    assert row["medium_small_fallback_range"] == [11, 16]
    assert row["medium_small_label_free_sampler"] is True
    assert row["preserve_nonzero_parent_selection_exactly"] is True


def test_compact_state_and_solver_remain_frozen():
    assert coverage.compact_state is coverage.v79.compact_state
    assert coverage.parent.core is coverage.core


def test_load_inputs_accepts_preserved_parent():
    preregistration, v82_preregistration, manifest, evidence = coverage.load_inputs()
    assert preregistration["parent_v82_evidence_digest"] == coverage.V82_EVIDENCE_DIGEST
    assert v82_preregistration["status"] == "preregistered_before_record_access"
    assert manifest["dataset_count"] == 7
    assert evidence["contributing_dataset_count"] == 6
