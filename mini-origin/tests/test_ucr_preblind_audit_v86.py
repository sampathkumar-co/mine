from __future__ import annotations

import json
from pathlib import Path

from mini_origin import ucr_preblind_audit_v86 as audit


def test_extracts_explicit_ucr_dataset_names_from_supported_forms():
    text = """
    https://www.timeseriesclassification.com/description.php?Dataset=__fixture_alpha__
    UCRArchive_2018/__fixture_beta__/__fixture_beta___TRAIN.tsv
    load_UCR_UEA_dataset(name="__fixture_gamma__")
    UCR_UEA_datasets().load_dataset('__fixture_delta__')
    aeon.datasets.load_classification("__fixture_epsilon__")
    """
    assert audit.extract_ucr_names(text) == {
        "fixture-alpha",
        "fixture-beta",
        "fixture-gamma",
        "fixture-delta",
        "fixture-epsilon",
    }


def test_unrelated_time_series_text_does_not_create_ucr_names():
    text = "time series classification can be useful without an archive URL"
    assert audit.extract_ucr_names(text) == set()


def test_extracts_conservative_named_dataset_values():
    text = json.dumps({
        "dataset_name": "__fixture_json__",
        "nested": {
            "name": "__fixture_context__",
            "raw_sha256": "abc",
        },
    })
    assert audit.extract_named_dataset_values(text, ".json") == {
        "fixture-json",
        "fixture-context",
    }


def test_plain_name_without_dataset_context_is_not_collected():
    text = json.dumps({"name": "ordinary workflow name"})
    assert audit.extract_named_dataset_values(text, ".json") == set()


def test_preregistration_freezes_ucr_preblind_boundary_and_selection_rule():
    data = json.loads(audit.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert data["status"] == "preregistered_before_ucr_catalogue_access"
    assert data["parent_v85_commit"] == audit.FROZEN_V85_COMMIT
    assert (
        data["parent_v85_authoritative_evidence_digest"]
        == audit.V85_EVIDENCE_DIGEST
    )
    assert data["future_metadata_only_selection"]["dataset_count"] == 7
    assert (
        data["future_metadata_only_selection"]["selection_seed"]
        == "mini-origin-v87-ucr-untouched-lock-2026-07-31"
    )


def test_parent_evidence_and_baseline_registry_match_frozen_records():
    preregistration, evidence, reproducibility, registry = audit.load_inputs()
    assert preregistration["future_source"]["catalogue_access_before_audit"] is False
    assert preregistration["solver_execution_before_audit"] is False
    assert evidence["verdict"] == "development_pass"
    assert evidence["gate_results"]["development_gate"] is True
    assert evidence["rust_crosscheck"]["mismatch_count"] == 0
    assert evidence["gate_results"]["fresh_external_evidence"] is False
    assert reproducibility["verdict"] == "reproduced"
    assert reproducibility["comparison"]["states_byte_identical"] is True
    assert registry["status"] == "pmlb_preblind_registry_v80_complete"
    assert registry["registry_digest"] == audit.V80_REGISTRY_DIGEST


def test_source_has_no_network_or_ucr_client_dependency():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "requests" not in source
    assert "urlopen" not in source
    assert "urllib" not in source
    assert "import aeon" not in source
    assert "import sktime" not in source
    assert "import tslearn" not in source


def test_occurrence_rows_remain_deterministic():
    mapping = {
        "9": [{"ref": "b", "path": "two"}],
        "2": [{"ref": "a", "path": "one"}],
    }
    rows = audit.occurrence_rows(mapping, "dataset_id", numeric=True)
    assert [row["dataset_id"] for row in rows] == [2, 9]
