from __future__ import annotations

import hashlib
import json

from mini_origin import external_survival_hash_lock_v90 as v90


def test_v90_preregistration_is_frozen_before_raw_access() -> None:
    payload = json.loads(v90.PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["status"] == "preregistered_before_raw_instance_access"
    sources = v90.selected_sources(payload)
    assert len(sources) == 6
    assert [row["name"] for row in sources] == [
        "rail507",
        "rail516",
        "rail582",
        "NY-distance",
        "BAY-distance",
        "COL-distance",
    ]


def test_byte_record_hashes_raw_bytes_without_parsing() -> None:
    payload = b"raw external bytes\x00\xff\n"
    row = v90.byte_record(
        family="synthetic",
        name="raw",
        requested_url="https://people.brunel.ac.uk/raw",
        final_url="https://people.brunel.ac.uk/raw",
        payload=payload,
        headers={},
    )
    assert row["bytes"] == len(payload)
    assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_only_expected_final_hosts_are_allowed() -> None:
    v90._validate_final_url("https://people.brunel.ac.uk/file")
    v90._validate_final_url("https://www.diag.uniroma1.it/file")
    try:
        v90._validate_final_url("https://example.com/file")
    except RuntimeError as exc:
        assert "unexpected final host" in str(exc)
    else:
        raise AssertionError("unexpected host was accepted")
