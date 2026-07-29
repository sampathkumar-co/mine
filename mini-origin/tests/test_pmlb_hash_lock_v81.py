from __future__ import annotations

import json
from pathlib import Path

from mini_origin import pmlb_hash_lock_v81 as lock


def preregistration():
    return json.loads(lock.PREREGISTRATION.read_text(encoding="utf-8-sig"))


def summary_bytes() -> bytes:
    return (
        "dataset\tn_instances\tn_features\tn_binary_features\t"
        "n_categorical_features\tn_continuous_features\tendpoint_type\t"
        "n_classes\timbalance\ttask\n"
        "alpha\t300\t4\t0\t0\t4\tcategorical\t2\t0.1\tclassification\n"
        "too_small\t299\t4\t0\t0\t4\tcategorical\t2\t0.1\tclassification\n"
        "regression\t500\t5\t0\t0\t5\tcontinuous\t1\t0.0\tregression\n"
    ).encode("utf-8")


def test_summary_parser_and_locked_eligibility():
    rows = lock.parse_summary(summary_bytes())
    assert len(rows) == 3
    assert lock.eligible_summary(rows[0], preregistration()) is True
    assert lock.eligible_summary(rows[1], preregistration()) is False
    assert lock.eligible_summary(rows[2], preregistration()) is False


def valid_metadata(name: str = "alpha") -> bytes:
    return (
        f"dataset: {name}\n"
        "description: synthetic benchmark\n"
        "source: generated locally\n"
        "publication: none\n"
        "task: classification\n"
        "target:\n"
        "  type: categorical\n"
    ).encode("utf-8")


def test_metadata_candidate_is_deterministic():
    row = lock.parse_summary(summary_bytes())[0]
    first, rejection = lock.metadata_candidate(
        row, valid_metadata(), set(), "seed"
    )
    second, _ = lock.metadata_candidate(
        row, valid_metadata(), set(), "seed"
    )
    assert rejection is None
    assert first is not None
    assert first["rank"] == second["rank"]
    assert first["canonical_metadata_digest"] == second[
        "canonical_metadata_digest"
    ]


def test_metadata_rejects_registry_name_and_open_sources():
    row = lock.parse_summary(summary_bytes())[0]
    candidate, rejection = lock.metadata_candidate(
        row, valid_metadata(), {"alpha"}, "seed"
    )
    assert candidate is None
    assert rejection["reason"] == "name_overlap"

    open_source = valid_metadata().replace(
        b"source: generated locally", b"source: UCI repository"
    )
    candidate, rejection = lock.metadata_candidate(
        row, open_source, set(), "seed"
    )
    assert candidate is None
    assert rejection["reason"] == "metadata_mentions_uci_or_openml"


def test_metadata_rejects_other_excluded_name_mentions():
    row = lock.parse_summary(summary_bytes())[0]
    metadata = valid_metadata().replace(
        b"description: synthetic benchmark",
        b"description: derived from old_dataset benchmark",
    )
    candidate, rejection = lock.metadata_candidate(
        row, metadata, {"old-dataset"}, "seed"
    )
    assert candidate is None
    assert rejection["reason"] == "metadata_mentions_excluded_name"
    assert rejection["overlaps"] == ["old-dataset"]


def test_raw_lock_hashes_gzip_bytes_without_parsing(monkeypatch):
    row = lock.parse_summary(summary_bytes())[0]
    candidate, _ = lock.metadata_candidate(
        row, valid_metadata(), set(), "seed"
    )
    payload = b"\x1f\x8b" + b"opaque-compressed-table"
    monkeypatch.setattr(lock, "download", lambda _url: payload)
    selected, rejected = lock.selected_byte_lock([candidate], 1)
    assert rejected == []
    assert selected[0]["raw_sha256"] == lock.sha256_bytes(payload)
    assert selected[0]["decompressed"] is False
    assert selected[0]["records_or_labels_parsed"] is False


def test_frozen_inputs_match_preserved_v80():
    prereg, registry, source = lock.load_inputs()
    assert prereg["parent_v80_commit"] == lock.PARENT_V80_COMMIT
    assert registry["registry_digest"] == lock.V80_REGISTRY_DIGEST
    assert source["source_commit"] == lock.SOURCE_COMMIT
    assert source["dataset_records_or_labels_accessed"] is False
    assert source["dataset_bytes_accessed"] is False


def test_source_contains_no_decompressor_or_table_parser():
    source = Path(lock.__file__).read_text(encoding="utf-8")
    assert "import gzip" not in source
    assert "gzip." not in source
    assert "pandas" not in source
    assert "read_csv" not in source
    assert "decompress(" not in source


def test_preregistration_keeps_zero_revision_budget():
    data = preregistration()
    assert data["algorithm_revisions"] == 0
    assert data["compiler_revisions"] == 0
    assert data["selector_revisions"] == 0
    assert data["scientific_threshold_revisions"] == 0
    assert data["dataset_count"] == 7


def test_raw_byte_url_uses_git_lfs_media_transport():
    url = lock.raw_url("vowel")
    assert url.startswith("https://media.githubusercontent.com/media/")
    assert lock.SOURCE_COMMIT in url
    assert url.endswith("/datasets/vowel/vowel.tsv.gz")
