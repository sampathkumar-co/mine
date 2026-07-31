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
    amendment = json.loads(audit.PROTOCOL_AMENDMENT.read_text(encoding="utf-8"))
    assert (
        amendment["status"]
        == "protocol_amendment_before_completed_audit_or_ucr_catalogue_access"
    )
    assert amendment["catalogue_access_before_amendment"] is False
    assert amendment["candidate_dataset_names_accessed_before_amendment"] is False
    review = json.loads(audit.REVIEW_AMENDMENT.read_text(encoding="utf-8"))
    assert review["status"] == "review_amendment_before_ucr_catalogue_access"
    assert review["catalogue_access_before_amendment"] is False
    assert review["external_dataset_network_access_before_amendment"] is False
    assert data["parent_v85_commit"] == audit.FROZEN_V85_COMMIT
    assert (
        data["parent_v85_authoritative_evidence_digest"]
        == audit.V85_EVIDENCE_DIGEST
    )
    assert data["future_metadata_only_selection"]["dataset_count"] == 7
    assert data["future_blind_gate"]["contributing_datasets"] == 7
    assert data["future_blind_gate"]["rust_mismatches"] == 0
    assert data["future_blind_gate"]["label_independence_mismatches"] == 0
    assert (
        data["future_source"][
            "candidate_dataset_names_ids_or_file_urls_committed_before_audit"
        ]
        is False
    )
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


def test_canonical_metadata_and_ranking_bytes_are_exact():
    metadata = {
        "normalized_dataset_name": "fixture-series",
        "total_instances": 321,
        "series_length": 24,
        "class_count": 3,
        "classification": True,
        "univariate": True,
        "train_url": "  https://example.test/cafe\u0301_TRAIN.tsv ",
        "test_url": "https://example.test/caf\u00e9_TEST.tsv",
    }
    encoded = audit.canonical_metadata_bytes(metadata)
    assert encoded.hex() == (
        "7b22636c6173735f636f756e74223a332c22636c617373696669636174696f6e"
        "223a747275652c226e6f726d616c697a65645f646174617365745f6e616d6522"
        "3a22666978747572652d736572696573222c227365726965735f6c656e677468"
        "223a32342c22746573745f75726c223a2268747470733a2f2f6578616d706c65"
        "2e746573742f636166c3a95f544553542e747376222c22746f74616c5f696e73"
        "74616e636573223a3332312c22747261696e5f75726c223a2268747470733a2f"
        "2f6578616d706c652e746573742f636166c3a95f545241494e2e747376222c22"
        "756e6976617269617465223a747275657d"
    )
    digest = audit.canonical_metadata_digest(metadata)
    assert digest == "76b5342ce3855e63bb9b52e7f0b6ba1938ded0e5b6f62387b837102d310d61cb"
    rank_bytes = audit.ranking_bytes(
        " seed\u0301 ", " release-2026 ", "fixture-series", digest
    )
    assert rank_bytes.hex() == (
        "5b2273656564cc81222c2272656c656173652d32303236222c2266697874757265"
        "2d736572696573222c223736623533343263653338353565363362623962353265"
        "376630623662613139333864656430653562366636323338376238333731303264"
        "3331306436316362225d"
    )
    assert audit.ranking_digest(
        " seed\u0301 ", " release-2026 ", "fixture-series", digest
    ) == "7e99f3b7b0ba41465f7a86cf9ad035dc87aa6db1cf5df20ddfd5836d50560822"

def test_canonical_metadata_rejects_ambiguous_types_and_fields():
    metadata = {
        "normalized_dataset_name": "fixture-series",
        "total_instances": True,
        "series_length": 24,
        "class_count": 3,
        "classification": True,
        "univariate": True,
        "train_url": "https://example.test/train",
        "test_url": "https://example.test/test",
    }
    try:
        audit.canonical_metadata_bytes(metadata)
    except TypeError:
        pass
    else:
        raise AssertionError("boolean instance count was accepted")
    valid = dict(metadata)
    valid["total_instances"] = 321
    digest = audit.canonical_metadata_digest(valid)
    try:
        audit.ranking_bytes("seed", "release", "fixture-series", digest.upper())
    except ValueError:
        pass
    else:
        raise AssertionError("uppercase digest was silently normalized")


def test_audit_ref_snapshots_include_head_and_immutable_shas(monkeypatch):
    monkeypatch.setattr(audit.base, "remote_refs", lambda: ("refs/remotes/origin/a",))
    mapping = {
        "HEAD^{commit}": "1" * 40,
        "refs/remotes/origin/a^{commit}": "2" * 40,
    }
    monkeypatch.setattr(audit.base, "git", lambda *args: mapping[args[1]])
    rows = audit.audit_ref_snapshots()
    assert {row["kind"] for row in rows} == {"checked_out_head", "origin_branch"}
    assert {row["sha"] for row in rows} == {"1" * 40, "2" * 40}


def test_all_tree_blobs_scans_every_suffix_and_keeps_oversized_entries(monkeypatch):
    listing = "\n".join([
        "100644 blob " + "a" * 40 + " 12\tmini-origin/a.sh",
        "100644 blob " + "b" * 40 + " 13\tmini-origin/b.rs",
        "100644 blob " + "c" * 40 + " 14\tmini-origin/c.mjs",
        "100644 blob " + "d" * 40 + " 15\tmini-origin/d.cpp",
        "100644 blob " + "e" * 40 + " 9999999\tresearch-evidence/archive.bin",
    ])
    monkeypatch.setattr(audit.base, "git", lambda *args: listing)
    rows = audit.all_tree_blobs("f" * 40)
    assert {Path(row["path"]).suffix for row in rows} == {
        ".sh", ".rs", ".mjs", ".cpp", ".bin"
    }
    assert max(int(row["size"]) for row in rows) > audit.AUDIT_MAX_FILE_BYTES


def test_download_failure_cannot_replace_a_selected_candidate():
    data = json.loads(audit.PREREGISTRATION.read_text(encoding="utf-8"))
    lock = data["future_metadata_only_selection"]["byte_lock_only"]
    assert lock["selected_file_download_attempts"] == 3
    assert lock["selected_file_retry_delays_seconds"] == [0, 5, 20]
    assert lock["selected_file_unavailability"] == (
        "fail the entire lock; never substitute a lower-ranked candidate"
    )
    assert data["failure_policy"]["candidate_substitution_after_selection"] == "forbidden"


def test_workflow_has_no_path_filter_bypass():
    root = Path(audit.__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/mini-origin-v86-ucr-preblind-audit.yml").read_text()
    trigger_prefix = workflow.split("concurrency:", 1)[0]
    assert "paths:" not in trigger_prefix
    assert "paths-ignore:" not in trigger_prefix
    assert "pull_request:" in trigger_prefix