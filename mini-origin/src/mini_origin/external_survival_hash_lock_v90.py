from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v90-external-survival-preregistration.json"
)
ALLOWED_FINAL_HOSTS = frozenset({
    "people.brunel.ac.uk",
    "www.diag.uniroma1.it",
    "diag.uniroma1.it",
})
USER_AGENT = "Mini-ORIGIN-v0.90-hash-lock/1.0"
TIMEOUT_SECONDS = 90
RETRY_DELAYS_SECONDS = (0, 2, 8)
MAX_FILE_BYTES = 32 * 1024 * 1024


def selected_sources(preregistration: dict[str, object]) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    external = preregistration["external_sources"]
    if not isinstance(external, dict):
        raise RuntimeError("v0.90 external_sources malformed")
    for family in ("or_library", "dimacs9"):
        block = external[family]
        if not isinstance(block, dict):
            raise RuntimeError(f"v0.90 family block malformed: {family}")
        selected = block["selected_instances"]
        if not isinstance(selected, list):
            raise RuntimeError(f"v0.90 selected_instances malformed: {family}")
        for row in selected:
            if not isinstance(row, dict):
                raise RuntimeError("v0.90 source row malformed")
            sources.append({
                "family": family,
                "name": str(row["name"]),
                "url": str(row["url"]),
            })
    return sources


def _validate_final_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"non-HTTPS final URL rejected: {url}")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_FINAL_HOSTS:
        raise RuntimeError(f"unexpected final host rejected: {host}")


def download_raw(url: str) -> tuple[bytes, str, dict[str, str]]:
    last_error: Exception | None = None
    for delay in RETRY_DELAYS_SECONDS:
        if delay:
            time.sleep(delay)
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/octet-stream,text/plain,*/*;q=0.1",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                final_url = response.geturl()
                _validate_final_url(final_url)
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > MAX_FILE_BYTES:
                    raise RuntimeError(
                        f"declared file exceeds v0.90 byte cap: {content_length}"
                    )
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_FILE_BYTES:
                        raise RuntimeError("download exceeded v0.90 byte cap")
                    chunks.append(chunk)
                payload = b"".join(chunks)
                headers = {
                    "content_type": response.headers.get("Content-Type", ""),
                    "content_length": response.headers.get("Content-Length", ""),
                    "last_modified": response.headers.get("Last-Modified", ""),
                    "etag": response.headers.get("ETag", ""),
                }
                return payload, final_url, headers
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"raw download failed after retries: {url}: {last_error}")


def byte_record(
    *,
    family: str,
    name: str,
    requested_url: str,
    final_url: str,
    payload: bytes,
    headers: dict[str, str],
) -> dict[str, object]:
    return {
        "family": family,
        "name": name,
        "requested_url": requested_url,
        "final_url": final_url,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "transport_headers": headers,
    }


def lock_sources(sources: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    records = []
    for source in sources:
        payload, final_url, headers = download_raw(source["url"])
        records.append(byte_record(
            family=source["family"],
            name=source["name"],
            requested_url=source["url"],
            final_url=final_url,
            payload=payload,
            headers=headers,
        ))
    return records


def build_manifest() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_raw_instance_access":
        raise RuntimeError("v0.90 preregistration status changed")
    sources = selected_sources(preregistration)
    if len(sources) != 6:
        raise RuntimeError(f"v0.90 requires exactly six raw streams, found {len(sources)}")
    records = lock_sources(sources)
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "status": "v90_raw_bytes_hash_locked_not_parsed",
        "version": "v0.90",
        "raw_stream_count": len(records),
        "records": records,
        "records_digest": hashlib.sha256(canonical).hexdigest(),
        "access_boundary": {
            "raw_bytes_downloaded": True,
            "decompression_performed": False,
            "tokenization_performed": False,
            "instance_records_parsed": False,
            "quotient_statistics_computed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": manifest["status"],
        "raw_stream_count": manifest["raw_stream_count"],
        "records_digest": manifest["records_digest"],
        "records": [
            {
                "family": row["family"],
                "name": row["name"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "final_url": row["final_url"],
            }
            for row in manifest["records"]
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
