from __future__ import annotations

import json
from pathlib import Path

from mini_origin import pmlb_preblind_audit_v80 as audit


def test_extracts_explicit_pmlb_dataset_names():
    text = """
    pmlb.fetch_data('alpha_case')
    fetch_data("beta-case")
    https://github.com/EpistasisLab/pmlb/tree/master/datasets/gamma_case
    datasets/delta-case/delta-case.tsv.gz
    """
    assert audit.extract_pmlb_names(text) == {
        "alpha-case",
        "beta-case",
        "gamma-case",
        "delta-case",
    }


def test_unrelated_strings_do_not_create_pmlb_names():
    assert audit.extract_pmlb_names("PMLB is a benchmark repository") == set()
    assert audit.extract_pmlb_names("datasets are useful") == set()


def test_preregistration_freezes_preblind_boundary():
    data = json.loads(audit.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert data["status"] == "preregistered_before_pmlb_catalogue_access"
    assert data["parent_v79_commit"] == audit.FROZEN_V79_COMMIT
    assert data["parent_v79_evidence_digest"] == audit.V79_EVIDENCE_DIGEST


def test_parent_evidence_matches_frozen_pass():
    preregistration, parent = audit.load_inputs()
    assert preregistration["pmlb_catalogue_or_repository_tree_access_before_audit"] is False
    assert preregistration["pmlb_candidate_names_ids_or_urls_committed_before_audit"] is False
    assert preregistration["pmlb_dataset_bytes_access_before_audit"] is False
    assert preregistration["solver_execution_before_audit"] is False
    assert parent["status"] == "small_query_coverage_development_pass_v79"
    assert parent["development_gate"] is True
    assert parent["rust_mismatch_count"] == 0


def test_source_has_no_pmlb_client_or_network_dependency():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "import pmlb" not in source
    assert "requests" not in source
    assert "urlopen" not in source
    assert "urllib" not in source


def test_extracts_openml_ids_from_all_supported_forms():
    text = """
    {"openml_dataset_id": 1068}
    https://openml.org/d/188
    https://www.openml.org/data/40499
    """
    assert audit.extract_openml_ids(text) == {1068, 188, 40499}


def test_extracts_normalized_names_from_json_and_pairs():
    text = json.dumps({
        "datasets": [
            {"openml_dataset_id": 469, "name": "analcatdata_dmft"},
            {"uci_id": 3, "task": "Annealing Data"},
        ]
    })
    assert audit.extract_dataset_names(text, ".json") == {
        "analcatdata-dmft",
        "annealing-data",
    }


def test_occurrence_rows_are_deterministic():
    mapping = {
        "9": [{"ref": "b", "path": "two"}],
        "2": [{"ref": "a", "path": "one"}],
    }
    rows = audit.occurrence_rows(mapping, "dataset_id", numeric=True)
    assert [row["dataset_id"] for row in rows] == [2, 9]


def test_json_context_recovers_generic_openml_dataset_ids():
    text = json.dumps({
        "datasets": [
            {"dataset_id": 1067, "task_id": 3917, "name": "kc1"},
            {"dataset_id": 1063, "raw_sha256": "abc", "name": "kc2"},
            {"dataset_id": 1049, "raw_md5": "def", "name": "pc4"},
        ]
    })
    assert audit.extract_openml_ids(text, ".json") == {1067, 1063, 1049}


def test_plain_generic_dataset_id_is_not_assumed_openml():
    text = json.dumps({"dataset_id": 999999, "name": "unrelated"})
    assert audit.extract_openml_ids(text, ".json") == set()
