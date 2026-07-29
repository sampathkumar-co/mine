import json

from mini_origin import fresh_external_lock_v74 as lock


def metadata(uci_id, name, data_url, task="Classification"):
    return {
        "uci_id": uci_id,
        "name": name,
        "repository_url": f"https://archive.ics.uci.edu/dataset/{uci_id}/x",
        "data_url": data_url,
        "tasks": [task],
        "num_instances": 500,
        "num_features": 8,
        "target_col": ["class"],
        "dataset_doi": f"doi-{uci_id}",
        "variables": [
            *[
                {"name": f"f{index}", "role": "Feature"}
                for index in range(8)
            ],
            {"name": "class", "role": "Target"},
        ],
    }


def test_metadata_filter_rejects_contaminated_name():
    reasons = lock.metadata_rejection_reasons(
        metadata(
            10,
            "Heart Example",
            "https://archive.ics.uci.edu/static/public/10/data.csv",
        ),
        {"heart"},
        {
            "required_task": "Classification",
            "minimum_instances": 200,
            "maximum_instances": 20000,
            "minimum_features": 4,
            "maximum_features": 120,
            "required_target_columns": 1,
        },
    )
    assert "contaminated_name_token" in reasons


def test_hash_lock_is_deterministic_and_excludes_registry(tmp_path, monkeypatch):
    prereg = {
        "status": "preregistered_before_metadata_access",
        "parent_v73_registry_digest": "registry-test",
        "frozen_v72_commit": "v72-test",
        "parent_v72_evidence_digest": "evidence-test",
        "metadata_list_url": "https://example.test/list",
        "metadata_dataset_url_template": "https://example.test/meta/{uci_id}",
        "selection_seed": "seed",
        "selection": {
            "dataset_count": 2,
            "required_task": "Classification",
            "minimum_instances": 200,
            "maximum_instances": 20000,
            "minimum_features": 4,
            "maximum_features": 120,
            "required_target_columns": 1,
            "maximum_csv_bytes": 1000,
        },
        "candidate_ids_names_urls_committed_before_execution": False,
        "data_bytes_access_before_preregistration": False,
        "record_or_label_parsing_during_lock": False,
        "solver_execution_during_lock": False,
        "protocol": "test",
        "claim_boundary": "test",
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({
        "status": "repository_dataset_registry_v73_complete",
        "registry_digest": "registry-test",
        "excluded_uci_ids": [1],
    }), encoding="utf-8")
    previous_path = tmp_path / "previous.json"
    previous_path.write_text(json.dumps({
        "pystreed_dataset_tokens": ["heart"],
    }), encoding="utf-8")
    monkeypatch.setattr(lock, "PREREGISTRATION", prereg_path)
    monkeypatch.setattr(lock, "V73_REGISTRY", registry_path)
    monkeypatch.setattr(lock, "V67_REGISTRY", previous_path)
    monkeypatch.setattr(lock, "V73_REGISTRY_DIGEST", "registry-test")
    monkeypatch.setattr(lock, "V72_EVIDENCE_DIGEST", "evidence-test")
    monkeypatch.setattr(lock, "FROZEN_V72_COMMIT", "v72-test")

    list_payload = {"status": 200, "data": [
        {"id": 1, "name": "Excluded", "url": "x"},
        {"id": 2, "name": "Heart Study", "url": "x"},
        {"id": 3, "name": "Gamma", "url": "x"},
        {"id": 4, "name": "Delta", "url": "x"},
        {"id": 5, "name": "Epsilon", "url": "x"},
        {"id": 6, "name": "Stale", "url": "x"},
    ]}
    payloads = {
        "https://example.test/list": json.dumps(list_payload).encode(),
        "https://example.test/meta/6": json.dumps({
            "status": 404,
            "message": "Dataset not found",
        }).encode(),
    }
    for uci_id, name in (
        (2, "Heart Study"),
        (3, "Gamma"),
        (4, "Delta"),
        (5, "Epsilon"),
    ):
        data_url = (
            f"https://archive.ics.uci.edu/static/public/{uci_id}/data.csv"
        )
        payloads[f"https://example.test/meta/{uci_id}"] = json.dumps({
            "status": 200,
            "data": metadata(uci_id, name, data_url),
        }).encode()
        payloads[data_url] = f"rows-{uci_id}".encode()
    monkeypatch.setattr(lock, "download", lambda url: payloads[url])

    first = lock.run(tmp_path / "first.json")
    second = lock.run(tmp_path / "second.json")
    assert first["lock_digest"] == second["lock_digest"]
    assert first["dataset_count"] == 2
    assert first["selected_overlap"] == []
    assert first["metadata_rejection_counts"]["metadata_unavailable"] == 1
    assert {row["uci_id"] for row in first["datasets"]}.isdisjoint({1, 2, 6})
    assert all("csv_sha256" in row for row in first["datasets"])
