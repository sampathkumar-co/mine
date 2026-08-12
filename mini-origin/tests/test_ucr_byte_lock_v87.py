from __future__ import annotations

import hashlib
import json

import pytest

from mini_origin import ucr_byte_lock_v87 as lock
from mini_origin import ucr_catalogue_protocol_v87 as catalogue


def fixture_metadata(name: str = "fixture-series") -> catalogue.CandidateMetadata:
    root = "https://www.timeseriesclassification.com/description.php?Dataset=FixtureSeries"
    return catalogue.CandidateMetadata(
        normalized_dataset_name=name,
        total_instances=400,
        series_length=40,
        class_count=3,
        classification=True,
        univariate=True,
        train_url="https://www.timeseriesclassification.com/data/FixtureSeries_TRAIN.ts",
        test_url="https://www.timeseriesclassification.com/data/FixtureSeries_TEST.ts",
        description_url=root,
        metadata_page_sha256="0" * 64,
    )


def test_metadata_rank_matches_frozen_serialization() -> None:
    metadata = fixture_metadata()
    seed = "mini-origin-v87-ucr-untouched-lock-2026-07-31"
    source = "a" * 64
    result = lock.metadata_rank(metadata, seed, source)
    encoded = json.dumps(
        metadata.canonical_fields(), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    rank_bytes = json.dumps(
        [seed, source, metadata.normalized_dataset_name, digest],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    assert result["canonical_metadata_digest"] == digest
    assert result["rank"] == hashlib.sha256(rank_bytes).hexdigest()


def data_spec() -> dict[str, object]:
    return {
        "selected_file_request_headers": {
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
            "Cache-Control": "no-cache",
            "User-Agent": "fixture",
        },
        "selected_file_connect_timeout_seconds": 15,
        "selected_file_read_timeout_seconds": 120,
        "selected_file_total_attempt_deadline_seconds": 180,
        "selected_file_maximum_redirects": 5,
        "maximum_selected_file_bytes": 100000000,
        "selected_file_retry_delays_seconds": [0, 5, 20],
        "selected_file_download_attempts": 3,
    }


def test_selected_file_lock_requires_success_agreement(monkeypatch) -> None:
    payload = b"locked-bytes"

    def fake_attempt(url: str, attempt_index: int, delay: int, spec: dict[str, object]):
        record = {
            "attempt_index": attempt_index,
            "scheduled_delay_seconds": delay,
            "final_url": url,
            "redirect_chain": [],
            "connect_elapsed_seconds": [0.01],
            "read_elapsed_seconds": 0.01,
            "total_elapsed_seconds": 0.03,
            "status_code": 200,
            "body_sha256": hashlib.sha256(payload).hexdigest(),
            "body_bytes": len(payload),
            "failure": None,
        }
        return record, payload

    monkeypatch.setattr(lock, "_data_attempt", fake_attempt)
    monkeypatch.setattr(lock.time, "sleep", lambda _: None)
    url = "https://www.timeseriesclassification.com/data/FixtureSeries_TRAIN.ts"
    result = lock.fetch_selected_files((url,), data_spec())
    assert result[url]["body"] == payload
    assert result[url]["body_sha256"] == hashlib.sha256(payload).hexdigest()
    assert len(result[url]["attempts"]) == 3


def test_selected_file_lock_rejects_disagreement(monkeypatch) -> None:
    def fake_attempt(url: str, attempt_index: int, delay: int, spec: dict[str, object]):
        payload = b"a" if attempt_index != 3 else b"b"
        return ({
            "attempt_index": attempt_index,
            "scheduled_delay_seconds": delay,
            "final_url": url,
            "redirect_chain": [],
            "connect_elapsed_seconds": [0.01],
            "read_elapsed_seconds": 0.01,
            "total_elapsed_seconds": 0.03,
            "status_code": 200,
            "body_sha256": hashlib.sha256(payload).hexdigest(),
            "body_bytes": len(payload),
            "failure": None,
        }, payload)

    monkeypatch.setattr(lock, "_data_attempt", fake_attempt)
    monkeypatch.setattr(lock.time, "sleep", lambda _: None)
    url = "https://www.timeseriesclassification.com/data/FixtureSeries_TEST.ts"
    with pytest.raises(RuntimeError, match="disagree"):
        lock.fetch_selected_files((url,), data_spec())
