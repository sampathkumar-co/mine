import json

from mini_origin import openml_preblind_audit_v76 as audit


def test_extracts_openml_ids_without_generic_number_matches():
    text = """
    {'openml_dataset_id': 123, 'name': 'Alpha'}
    https://www.openml.org/d/456
    https://openml.org/data/789
    {'workflow_run': 30449862500}
    """
    assert audit.extract_openml_ids(text) == {123, 456, 789}


def test_extracts_normalized_dataset_names_from_json_and_pairs():
    text = json.dumps({
        "datasets": [
            {"uci_id": 27, "name": "Credit Approval"},
            {"openml_dataset_id": 123, "name": "Open ML Example"},
        ],
        "dataset_summaries": [
            {"uci_id": 78, "task": "Page Blocks Classification"},
        ],
    })
    assert audit.extract_dataset_names(text, ".json") == {
        "credit-approval",
        "open-ml-example",
        "page-blocks-classification",
    }


def test_preregistration_contains_no_candidate_metadata():
    row = json.loads(audit.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert row["status"] == "preregistered_before_openml_metadata_access"
    assert row["openml_candidate_ids_names_or_urls_committed_before_audit"] is False
    assert row["openml_api_access_before_audit"] is False
    assert row["openml_dataset_bytes_access_before_audit"] is False
    assert row["solver_execution_before_audit"] is False
    text = audit.PREREGISTRATION.read_text(encoding="utf-8-sig")
    assert "openml.org/d/" not in text
    assert "openml_dataset_id" not in text
