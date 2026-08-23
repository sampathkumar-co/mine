from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v92-tsplib-external-holdout.json"
)
TRANSPORT_AMENDMENT = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v92-tsplib-transport-amendment.json"
)
ALLOWED_HOST = "comopt.ifi.uni-heidelberg.de"
CATALOGUE_URL = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/XML-TSPLIB/instances/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)
TIMEOUT_SECONDS = 90
RETRY_DELAYS_SECONDS = (0, 2, 8)
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024


def _verify_transport_amendment() -> None:
    payload = json.loads(TRANSPORT_AMENDMENT.read_text(encoding="utf-8"))
    if payload["status"] != "transport_only_amendment_after_404_before_archive_bytes":
        raise RuntimeError("v0.92 transport amendment status changed")
    failed = payload["failed_attempt"]
    if bool(failed["selected_archive_bytes_obtained"]):
        raise RuntimeError("v0.92 amendment incorrectly claims prior archive access")
    change = payload["transport_change_only"]
    protected = (
        "selected_instance_names_changed",
        "selected_urls_changed",
        "projection_changed",
        "parser_changed",
        "held_karp_baseline_changed",
        "bruteforce_certificate_changed",
        "frozen_v91_rule_changed",
        "five_percent_gate_changed",
        "kill_rule_changed",
    )
    if any(bool(change[key]) for key in protected):
        raise RuntimeError("v0.92 transport amendment changed scientific protocol")


def selected_sources() -> list[dict[str, str]]:
    _verify_transport_amendment()
    payload = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if payload["status"] != "preregistered_before_selected_archive_access":
        raise RuntimeError("v0.92 preregistration status changed")
    rows = payload["source"]["selected_archives"]
    result = [{"name": str(row["name"]), "url": str(row["url"])} for row in rows]
    if [row["name"] for row in result] != ["gr21", "gr24", "p43"]:
        raise RuntimeError("v0.92 selected source names changed")
    return result


def _validate_final_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != ALLOWED_HOST:
        raise RuntimeError(f"unexpected v0.92 final URL: {url}")


def download_raw(url: str) -> tuple[bytes, str, dict[str, str]]:
    last_error: Exception | None = None
    for delay in RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Referer": CATALOGUE_URL,
                    "Accept": "application/zip,application/octet-stream,*/*;q=0.1",
                    "Accept-Encoding": "identity",
                },
            )
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                final_url = response.geturl()
                _validate_final_url(final_url)
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_ARCHIVE_BYTES:
                    raise RuntimeError("v0.92 archive exceeds declared byte cap")
                chunks = []
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise RuntimeError("v0.92 archive exceeds byte cap")
                    chunks.append(chunk)
                return b"".join(chunks), final_url, {
                    "content_type": response.headers.get("Content-Type", ""),
                    "content_length": response.headers.get("Content-Length", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                    "etag": response.headers.get("ETag", ""),
                }
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"v0.92 raw archive download failed: {url}: {last_error}")


def build_manifest() -> dict[str, object]:
    records = []
    for source in selected_sources():
        payload, final_url, headers = download_raw(source["url"])
        records.append({
            "name": source["name"],
            "requested_url": source["url"],
            "final_url": final_url,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "transport_headers": headers,
        })
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "v92_tsplib_archives_hash_locked_not_opened",
        "version": "v0.92",
        "archive_count": len(records),
        "records": records,
        "records_digest": digest,
        "access_boundary": {
            "archive_bytes_downloaded": True,
            "zip_opened": False,
            "xml_parsed": False,
            "edge_costs_inspected": False,
            "tsp_states_constructed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "records_digest": result["records_digest"],
        "records": [
            {key: row[key] for key in ("name", "bytes", "sha256", "final_url")}
            for row in result["records"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
