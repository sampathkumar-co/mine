from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RECORD_PATH = REPO_ROOT / "research-evidence" / "mini-origin-v87-authoritative-preaccess-record.json"
EXPECTED_IMPLEMENTATION_COMMIT = "ca90a8cd98a9af6447639deb4424d01e1c09c4cc"
EXPECTED_FROZEN_PARENT = "912c3ebd933ae39eb05e10467f1ecad56e326b03"
EXPECTED_ARTIFACT_ZIP_SHA256 = "4a43c35b951c3ff276044fd837bd92d337f370fd6c869dc6f5b2f8fc840b57d3"
EXPECTED_REGISTRY_JSON_SHA256 = "f1b50e11ce23e4bdfa42e3575c08379dc99d8bd98bc15fd7d581a464865effe0"
EXPECTED_INTERNAL_REGISTRY_DIGEST = "c986710aa38d89bb9bf00df9a5e5817a26c359915c0f028646208cd9bcfe8ec0"
EXPECTED_RECORD_CANONICAL_SHA256 = "b74e9a35dd6a61cf621ec8ace9ad2e4da612db8940526a604fba08831cc207e9"


def load_record() -> dict[str, object]:
    with RECORD_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_v87_preaccess_record_is_fail_closed_and_exact() -> None:
    record = load_record()

    assert set(record) == {
        "status",
        "protocol_version",
        "implementation_commit",
        "frozen_parent_v85_commit",
        "github_verification",
        "repository_audit",
        "preserved_scientific_constraints",
        "access_boundary",
        "next_authorized_step",
        "claim_boundary",
    }
    assert record["status"] == "authoritative_preaccess_contract_verified"
    assert record["protocol_version"] == "v0.87"
    assert record["implementation_commit"] == EXPECTED_IMPLEMENTATION_COMMIT
    assert record["frozen_parent_v85_commit"] == EXPECTED_FROZEN_PARENT

    verification = record["github_verification"]
    assert set(verification) == {
        "ucr_preblind_audit_run",
        "mini_origin_ci_run",
        "artifact_id",
        "artifact_zip_sha256",
        "registry_json_sha256",
        "internal_registry_digest",
        "conclusions",
    }
    assert verification["ucr_preblind_audit_run"] == 30609072431
    assert verification["mini_origin_ci_run"] == 30609072440
    assert verification["artifact_id"] == 8784724372
    assert verification["artifact_zip_sha256"] == EXPECTED_ARTIFACT_ZIP_SHA256
    assert verification["registry_json_sha256"] == EXPECTED_REGISTRY_JSON_SHA256
    assert verification["internal_registry_digest"] == EXPECTED_INTERNAL_REGISTRY_DIGEST
    assert verification["conclusions"] == {
        "ucr_preblind_audit": "success",
        "mini_origin_ci": "success",
    }

    assert record["repository_audit"] == {
        "immutable_ref_snapshots": 238,
        "unique_commits": 237,
        "tree_entries": 26625,
        "unique_blob_paths": 716,
        "bytes_scanned": 7521288,
        "failures": 0,
        "oversized_omissions": 0,
    }

    constraints = record["preserved_scientific_constraints"]
    assert constraints == {
        "v082_thresholds_and_budgets_unchanged": True,
        "v082_v084_negative_results_binding": True,
        "v085_opened_data_classification_unchanged": True,
        "compiler_selector_planner_solver_changes": 0,
        "threshold_relaxation": False,
        "post_access_substitution_forbidden": True,
    }

    assert record["access_boundary"] == {
        "ucr_catalogue_accessed": False,
        "candidate_metadata_accessed": False,
        "dataset_bytes_accessed": False,
        "records_or_labels_accessed": False,
        "solver_executed": False,
        "fresh_external_evidence": False,
    }
    assert record["next_authorized_step"] == (
        "Execute the frozen v0.87 transport and metadata lock exactly as preregistered. "
        "Any mismatch between the live source and the frozen parser or transport contract "
        "rejects the lock rather than permitting adaptation."
    )
    assert record["claim_boundary"] == (
        "Authoritative preregistration and repository-provenance evidence only; not external "
        "validation, independent replication, novelty proof, or a breakthrough."
    )


def test_v87_preaccess_record_has_pinned_canonical_digest() -> None:
    raw = RECORD_PATH.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_RECORD_CANONICAL_SHA256
    assert raw.endswith(b"\n")
