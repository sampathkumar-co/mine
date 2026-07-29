import json
from types import SimpleNamespace

from mini_origin import openml_hash_lock_v77 as lock


SELECTION = {
    "dataset_count": 7,
    "minimum_instances": 500,
    "maximum_instances": 20000,
    "minimum_features": 4,
    "maximum_features": 120,
    "required_task_type_id": 1,
    "required_target_columns": 1,
    "maximum_dataset_bytes": 20000000,
    "reject_sparse_datasets": True,
    "reject_any_frozen_openml_id": True,
    "reject_any_frozen_dataset_name": True,
    "reject_uci_origin_sources": True,
}


def make_row(**changes):
    row = {
        "task_id": 10,
        "task_type_id": 1,
        "dataset_id": 1000,
        "name": "Fresh Example",
        "normalized_name": "fresh-example",
        "format": "ARFF",
        "url": "https://openml.org/data/v1/download/1000/example.arff",
        "target_name": "class",
        "default_target_attribute": "class",
        "original_data_url": "https://example.org/fresh",
        "description": "Independent source",
        "citation": "",
        "paper_url": "",
        "num_instances": 1000,
        "num_features": 12,
    }
    row.update(changes)
    return row


def registry(ids=None, names=None):
    return {
        "excluded_openml_dataset_ids": ids or [],
        "excluded_dataset_names": names or [],
    }


def test_metadata_row_is_stable_and_uses_only_metadata():
    task = SimpleNamespace(
        task_type_id=SimpleNamespace(value=1),
        target_name="class",
    )
    dataset = SimpleNamespace(
        dataset_id=1000,
        name="Fresh Example",
        version=2,
        creator="Researcher",
        licence="CC BY",
        visibility="public",
        format="ARFF",
        url="https://openml.org/data/v1/download/1000/example.arff",
        default_target_attribute="class",
        row_id_attribute=None,
        ignore_attribute=None,
        original_data_url="https://example.org/fresh",
        paper_url="https://example.org/paper",
        description="Independent source",
        citation="citation",
        md5_checksum="ABCDEF",
        qualities={
            "NumberOfInstances": 1000.0,
            "NumberOfFeatures": 12.0,
            "NumberOfClasses": 2.0,
            "NumberOfMissingValues": 0.0,
        },
    )
    first = lock.metadata_row(10, task, dataset)
    second = lock.metadata_row(10, task, dataset)
    assert first == second
    assert first["task_type_id"] == 1
    assert first["num_instances"] == 1000
    assert first["md5_checksum"] == "abcdef"


def test_metadata_filter_accepts_independent_dense_dataset():
    assert lock.metadata_rejection_reasons(
        make_row(), registry(), SELECTION
    ) == []


def test_metadata_filter_rejects_contamination_and_uci_origin():
    reasons = lock.metadata_rejection_reasons(
        make_row(
            dataset_id=123,
            normalized_name="old-name",
            original_data_url="https://archive.ics.uci.edu/example",
            format="Sparse_ARFF",
        ),
        registry([123], ["old-name"]),
        SELECTION,
    )
    assert reasons == [
        "frozen_dataset_name",
        "frozen_openml_dataset_id",
        "sparse",
        "uci_origin",
    ]


def test_preregistration_freezes_rules_before_candidates():
    row = json.loads(lock.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert row["status"] == "preregistered_before_openml_suite_access"
    assert row["benchmark_suite_id"] == 99
    assert row["candidate_task_or_dataset_ids_names_urls_committed_before_execution"] is False
    assert row["openml_suite_or_candidate_metadata_access_before_preregistration"] is False
    assert row["dataset_bytes_access_before_preregistration"] is False
    assert row["record_target_or_feature_parsing_during_lock"] is False
    assert row["solver_execution_during_lock"] is False
    assert row["selection"] == SELECTION
