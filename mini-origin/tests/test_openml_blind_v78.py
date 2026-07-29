import hashlib
import json

import pytest

from mini_origin import openml_blind_v78 as blind


def locked_dataset(payload: bytes) -> dict[str, object]:
    return {
        "dataset_id": 999,
        "name": "synthetic-openml",
        "target_name": "class",
        "default_target_attribute": "class",
        "row_id_attribute": "id",
        "ignore_attribute": None,
        "num_features": 4,
        "num_instances": 2,
        "raw_sha256": hashlib.sha256(payload).hexdigest(),
        "raw_md5": hashlib.md5(payload).hexdigest(),
        "raw_bytes": len(payload),
    }


def test_dense_arff_uses_target_and_ignores_only_committed_id():
    payload = b"""@RELATION test
@ATTRIBUTE id NUMERIC
@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE f2 {x,y}
@ATTRIBUTE class {A,B}
@DATA
1,3,?,A
2,4,y,B
"""
    records, summary = blind.parse_openml_arff(locked_dataset(payload), payload)
    assert records == [(('3.0', ''), 'A'), (('4.0', 'y'), 'B')]
    assert summary["feature_columns"] == ["f1", "f2"]
    assert summary["target_column"] == "class"
    assert summary["ignored_columns"] == ["id"]
    assert summary["missing_feature_cells"] == 1


def test_hash_mismatch_rejects_before_parsing():
    payload = b"@RELATION x\n@ATTRIBUTE class {A}\n@DATA\nA\n"
    dataset = locked_dataset(payload)
    dataset["raw_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        blind.parse_openml_arff(dataset, payload)


def test_empty_target_is_rejected():
    payload = b"""@RELATION test
@ATTRIBUTE id NUMERIC
@ATTRIBUTE f1 NUMERIC
@ATTRIBUTE f2 {x,y}
@ATTRIBUTE class {A,B}
@DATA
1,3,x,?
2,4,y,B
"""
    with pytest.raises(RuntimeError, match="empty target"):
        blind.parse_openml_arff(locked_dataset(payload), payload)


def test_manifest_bytes_match_frozen_commitment():
    row = json.loads(blind.PREREGISTRATION.read_text(encoding="utf-8-sig"))
    assert hashlib.sha256(blind.MANIFEST.read_bytes()).hexdigest() == row[
        "parent_v77_manifest_sha256"
    ]


def test_v78_reuses_v75_scientific_gate():
    root = blind.PREREGISTRATION.parents[1]
    v75 = json.loads(
        (root / "campaigns" / "v75-fresh-external-blind.json")
        .read_text(encoding="utf-8-sig")
    )["locked_gate"]
    v78 = json.loads(
        blind.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )["locked_gate"]
    shared = set(v75) - {"previously_zero_datasets"}
    assert {key: v78[key] for key in shared} == {
        key: v75[key] for key in shared
    }
    assert v78["previously_zero_datasets"] == [
        "pc1",
        "eucalyptus",
        "texture",
        "kc1",
        "kc2",
        "analcatdata_dmft",
        "pc4",
    ]
