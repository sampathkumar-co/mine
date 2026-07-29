import json

from mini_origin import repository_dataset_audit_v73 as audit


def test_v73_registry_requires_every_preblind_dataset(monkeypatch):
    previous = json.loads(audit.V67_REGISTRY.read_text(encoding="utf-8"))
    required = set(int(value) for value in previous["excluded_uci_ids"])
    required.update(audit.NEWLY_OPENED_V68_IDS)
    discovered = sorted(required | {999})
    report = {
        "status": "repository_dataset_registry_complete",
        "ref_count": 1,
        "files_scanned": 1,
        "bytes_scanned": 1,
        "uci_id_count": len(discovered),
        "pystreed_token_count": 0,
        "registry_digest": "test-digest",
        "uci_datasets": [{"uci_id": value} for value in discovered],
    }
    monkeypatch.setattr(audit.base, "audit", lambda: dict(report))
    result = audit.audit()
    assert result["status"] == "repository_dataset_registry_v73_complete"
    assert result["excluded_uci_ids"] == discovered
    assert result["excluded_uci_id_count"] == len(discovered)
    assert all(result["required_preblind_uci_checks"].values())


def test_v73_preregistration_contains_no_candidate_metadata():
    preregistration = json.loads(
        audit.PREREGISTRATION.read_text(encoding="utf-8-sig")
    )
    assert preregistration["candidate_metadata_committed_before_registry"] is False
    assert preregistration["external_archive_access_before_registry"] is False
    assert preregistration["record_or_label_access_before_registry"] is False
    assert preregistration["solver_execution_before_registry"] is False
    text = audit.PREREGISTRATION.read_text(encoding="utf-8-sig")
    assert "archive.ics.uci.edu/static/public/" not in text
