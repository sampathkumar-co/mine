import hashlib
import json

from mini_origin import fresh_external_blind_v75 as blind


def test_standardized_csv_uses_only_feature_roles():
    dataset = {
        "uci_id": 999,
        "num_instances": 2,
        "num_features": 2,
        "target_columns": ["class"],
    }
    metadata = {
        "status": 200,
        "data": {
            "uci_id": 999,
            "variables": [
                {"name": "id", "role": "ID"},
                {"name": "f1", "role": "Feature"},
                {"name": "ignored", "role": "Other"},
                {"name": "f2", "role": "Feature"},
                {"name": "class", "role": "Target"},
            ],
        },
    }
    payload = b"id,f1,ignored,f2,class\n1, 3 ,x,,A\n2,4,y,5,B\n"
    rows, summary = blind.parse_standardized_csv(dataset, payload, metadata)
    assert rows == [(('3', ''), 'A'), (('4', '5'), 'B')]
    assert summary["feature_columns"] == ["f1", "f2"]
    assert summary["ignored_columns"] == ["id", "ignored"]


def test_manifest_bytes_match_frozen_commitment():
    row = json.loads(blind.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert hashlib.sha256(blind.MANIFEST.read_bytes()).hexdigest() == row[
        "parent_v74_manifest_sha256"
    ]


def test_v75_reuses_v70_numeric_thresholds():
    root = blind.PREREGISTRATION.parents[1]
    v70 = json.loads(
        (root / "campaigns" / "v70-numeric-threshold-frontier-preregistration.json")
        .read_text(encoding="utf-8")
    )["locked_gate"]
    v75 = json.loads(
        blind.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )["locked_gate"]
    shared = set(v70) - {
        "previously_zero_datasets",
        "minimum_states_from_each_previously_zero_dataset",
    }
    assert {key: v75[key] for key in shared} == {
        key: v70[key] for key in shared
    }
