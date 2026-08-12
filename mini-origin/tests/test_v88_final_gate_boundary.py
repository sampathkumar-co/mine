from __future__ import annotations

import json
from pathlib import Path

from mini_origin import openml_final_blind_v88 as blind
from mini_origin import openml_final_lock_v88 as lock

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "mini-origin" / "campaigns" / "v88-final-fresh-openml-gate.json"


def test_v88_preregistration_is_fail_closed() -> None:
    row = json.loads(CAMPAIGN.read_text(encoding="utf-8-sig"))
    assert row["status"] == "preregistered_before_v88_openml_suite_access"
    assert row["parent_v85_commit"] == blind.V85_COMMIT == lock.V85_COMMIT
    assert row["parent_v85_evidence_digest"] == blind.V85_EVIDENCE_DIGEST == lock.V85_EVIDENCE_DIGEST
    assert row["parent_v86_registry_digest"] == blind.V86_REGISTRY_DIGEST == lock.V86_REGISTRY_DIGEST
    assert row["selection"]["dataset_count"] == 7
    assert row["selection"]["reject_any_frozen_openml_id"] is True
    assert row["selection"]["reject_any_frozen_dataset_name"] is True
    assert row["selection"]["reject_uci_origin_sources"] is True
    assert all(value is False for value in row["preaccess_boundary"].values())
    assert all(int(value) == 0 for value in row["revision_budget_after_suite_access"].values())


def test_v88_final_gate_preserves_v82_thresholds() -> None:
    row = json.loads(CAMPAIGN.read_text(encoding="utf-8-sig"))
    gate = row["locked_gate"]
    assert row["exact_budget"] == 500000
    assert row["budget_ladder"] == [10000, 50000, 250000, 500000]
    assert gate["contributing_datasets"] == 7
    assert gate["minimum_base_states"] == 60
    assert gate["minimum_profiled_states"] == 180
    assert gate["minimum_both_plain_bounded"] == 40
    assert gate["minimum_bounded_only"] == 25
    assert gate["minimum_states_with_lower_bound_pruning"] == 30
    assert gate["minimum_aggregate_bounded_reduction_fraction"] == 0.1
    assert gate["minimum_dominated_queries_removed"] == 1000
    assert gate["minimum_median_plain_bounded_ratio"] == 10.0
    assert gate["minimum_p90_plain_bounded_ratio"] == 30.0
    assert gate["minimum_50k_solve_advantage"] == 20
    assert gate["rust_mismatches"] == 0
    assert row["blind_gate"]["minimum_states_from_each_dataset"] == 6
