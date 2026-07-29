from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import pytest

from mini_origin import pmlb_blind_v82 as blind


def payload(rows: list[list[str]]) -> bytes:
    handle = io.StringIO()
    writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
    writer.writerows(rows)
    return gzip.compress(handle.getvalue().encode("utf-8"), mtime=0)


def dataset_for(raw: bytes, *, instances: int = 2, features: int = 2):
    return {
        "name": "synthetic-pmlb",
        "raw_sha256": blind.sha256_bytes(raw),
        "raw_bytes": len(raw),
        "instances": instances,
        "features": features,
    }


def test_dense_gzip_tsv_parser_preserves_feature_order():
    raw = payload([
        ["first", "target", "second"],
        ["a", "yes", "1"],
        ["", "no", "2"],
    ])
    records, summary = blind.parse_pmlb_table(dataset_for(raw), raw)
    assert records == [(('a', '1'), 'yes'), (('', '2'), 'no')]
    assert summary["feature_columns"] == ["first", "second"]
    assert summary["target_column"] == "target"
    assert summary["record_count"] == 2
    assert summary["missing_feature_cells"] == 1
    assert summary["raw_sha256_verified"] is True
    assert summary["raw_bytes_verified"] is True


def test_parser_rejects_raw_hash_tampering():
    raw = payload([["a", "target"], ["x", "1"]])
    dataset = dataset_for(raw, instances=1, features=1)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        blind.parse_pmlb_table(dataset, raw + b"tamper")


def test_parser_rejects_missing_exact_target():
    raw = payload([["a", "label"], ["x", "1"]])
    with pytest.raises(RuntimeError, match="target column missing"):
        blind.parse_pmlb_table(
            dataset_for(raw, instances=1, features=1), raw
        )


def test_parser_rejects_duplicate_headers():
    raw = payload([["a", "a", "target"], ["x", "y", "1"]])
    with pytest.raises(RuntimeError, match="duplicate TSV header"):
        blind.parse_pmlb_table(
            dataset_for(raw, instances=1, features=2), raw
        )


def test_parser_rejects_malformed_rows():
    raw = payload([["a", "target"], ["x"]])
    with pytest.raises(RuntimeError, match="malformed PMLB row"):
        blind.parse_pmlb_table(
            dataset_for(raw, instances=1, features=1), raw
        )


def test_parser_rejects_empty_targets():
    raw = payload([["a", "target"], ["x", ""]])
    with pytest.raises(RuntimeError, match="empty target"):
        blind.parse_pmlb_table(
            dataset_for(raw, instances=1, features=1), raw
        )


def test_parser_rejects_shape_mismatch():
    raw = payload([["a", "target"], ["x", "1"]])
    with pytest.raises(RuntimeError, match="record count mismatch"):
        blind.parse_pmlb_table(
            dataset_for(raw, instances=2, features=1), raw
        )


def test_frozen_inputs_match_preserved_parents():
    prereg, manifest, parent = blind.load_frozen_inputs()
    assert prereg["parent_v81_commit"] == blind.FROZEN_V81_COMMIT
    assert prereg["frozen_v79_commit"] == blind.FROZEN_V79_COMMIT
    assert manifest["lock_digest"] == blind.V81_LOCK_DIGEST
    assert len(manifest["datasets"]) == 7
    assert parent["evidence_digest"] == blind.V79_EVIDENCE_DIGEST
    assert parent["rust_mismatch_count"] == 0


def test_locked_gate_requires_every_dataset():
    prereg = json.loads(
        blind.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )
    gate = prereg["locked_gate"]
    assert gate["contributing_datasets"] == 7
    assert gate["minimum_states_from_each_dataset"] == 6
    assert gate["minimum_states_from_each_previously_zero_dataset"] == 6
    assert gate["previously_zero_datasets"] == prereg["selected_datasets"]
    assert gate["rust_mismatches"] == 0


def test_adapter_has_no_dataset_specific_exceptions():
    prereg = json.loads(
        blind.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )
    assert prereg["adapter_protocol"]["dataset_specific_exceptions"] is False
    source = Path(blind.__file__).read_text(encoding="utf-8")
    for name in prereg["selected_datasets"]:
        assert name not in source
