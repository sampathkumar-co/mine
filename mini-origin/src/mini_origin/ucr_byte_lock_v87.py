from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import ssl
import subprocess
import time
import unicodedata
from urllib.parse import urljoin, urlsplit

from mini_origin import ucr_catalogue_protocol_v87 as catalogue

PREREGISTRATION = Path(__file__).resolve().parents[2] / "campaigns" / "v86-ucr-preblind-audit.json"
PREACCESS_RECORD = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v87-authoritative-preaccess-record.json"
EXPECTED_REGISTRY_SHA256 = "f1b50e11ce23e4bdfa42e3575c08379dc99d8bd98bc15fd7d581a464865effe0"
EXPECTED_REGISTRY_DIGEST = "c986710aa38d89bb9bf00df9a5e5817a26c359915c0f028646208cd9bcfe8ec0"
EXPECTED_IMPLEMENTATION_COMMIT = "ca90a8cd98a9af6447639deb4424d01e1c09c4cc"
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ascii_trim_nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip(catalogue.ASCII_EDGE_WHITESPACE)


def _failure_text(error: BaseException) -> str:
    text = catalogue.normalize_visible_text(f"{type(error).__name__}: {error}")
    return text[:1000] or type(error).__name__


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("frozen total attempt deadline exceeded")
    return remaining


def _read_capped(response: http.client.HTTPResponse, connection: http.client.HTTPSConnection, maximum_bytes: int, read_timeout: float, total_deadline: float) -> tuple[bytes, float]:
    start = time.monotonic()
    deadline = start + read_timeout
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = min(deadline - time.monotonic(), _remaining(total_deadline))
        if remaining <= 0:
            raise TimeoutError("frozen whole-body read deadline exceeded")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        chunk = response.read(min(65536, maximum_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum_bytes:
            raise OverflowError("response exceeds frozen byte cap")
    elapsed = time.monotonic() - start
    if elapsed > read_timeout:
        raise TimeoutError("frozen whole-body read deadline exceeded")
    return b"".join(chunks), elapsed


def _raw_get(url: str, *, headers: tuple[tuple[str, str], ...], connect_timeout: float, read_timeout: float, total_timeout: float, maximum_redirects: int, maximum_bytes: int) -> dict[str, object]:
    requested = catalogue.canonical_url(catalogue.ROOT_URL, url)
    current = requested
    redirects: list[str] = []
    connect_elapsed: list[float] = []
    started = time.monotonic()
    total_deadline = started + total_timeout
    context = ssl.create_default_context()
    while True:
        parts = urlsplit(current)
        connection = http.client.HTTPSConnection(
            parts.hostname,
            port=parts.port or 443,
            timeout=min(connect_timeout, _remaining(total_deadline)),
            context=context,
        )
        connect_started = time.monotonic()
        connection.connect()
        connection_elapsed = time.monotonic() - connect_started
        if connection_elapsed > connect_timeout:
            connection.close()
            raise TimeoutError("frozen connect timeout exceeded")
        connect_elapsed.append(connection_elapsed)
        if connection.sock is not None:
            connection.sock.settimeout(min(read_timeout, _remaining(total_deadline)))
        target = parts.path or "/"
        if parts.query:
            target += "?" + parts.query
        connection.request("GET", target, headers=dict(headers))
        response = connection.getresponse()
        status = int(response.status)
        if status in REDIRECT_STATUSES:
            location = response.getheader("Location")
            response.close()
            connection.close()
            if location is None:
                raise RuntimeError("redirect response omitted Location")
            if len(redirects) >= maximum_redirects:
                raise RuntimeError("frozen redirect limit exceeded")
            current = catalogue.canonical_url(current, location)
            redirects.append(current)
            continue
        if status != 200:
            response.close()
            connection.close()
            raise RuntimeError(f"HTTP {status}")
        body, read_elapsed = _read_capped(
            response, connection, maximum_bytes, read_timeout, total_deadline
        )
        response.close()
        connection.close()
        if not body:
            raise RuntimeError("empty response body")
        total_elapsed = time.monotonic() - started
        return {
            "final_url": catalogue.canonical_url(requested, current),
            "body": body,
            "redirect_chain": tuple(redirects),
            "connect_elapsed_seconds": tuple(connect_elapsed),
            "read_elapsed_seconds": read_elapsed,
            "total_elapsed_seconds": total_elapsed,
            "status_code": 200,
        }


def _html_attempt(url: str, attempt_index: int, delay: int) -> catalogue.HTMLPageAttempt:
    try:
        result = _raw_get(
            url,
            headers=catalogue.HTML_PAGE_REQUEST_HEADERS,
            connect_timeout=catalogue.HTML_PAGE_CONNECT_TIMEOUT_SECONDS,
            read_timeout=catalogue.HTML_PAGE_READ_TIMEOUT_SECONDS,
            total_timeout=catalogue.HTML_PAGE_TOTAL_DEADLINE_SECONDS,
            maximum_redirects=catalogue.HTML_PAGE_MAX_REDIRECTS,
            maximum_bytes=catalogue.HTML_PAGE_MAX_RESPONSE_BYTES,
        )
    except Exception as error:
        return catalogue.HTMLPageAttempt(
            attempt_index, delay, catalogue.HTML_PAGE_REQUEST_METHOD,
            catalogue.HTML_PAGE_REQUEST_HEADERS, False, (), (), None,
            None, None, None, None, _failure_text(error),
        )
    return catalogue.HTMLPageAttempt(
        attempt_index=attempt_index,
        scheduled_delay_seconds=delay,
        request_method=catalogue.HTML_PAGE_REQUEST_METHOD,
        request_headers=catalogue.HTML_PAGE_REQUEST_HEADERS,
        tls_certificate_validated=True,
        redirect_chain=result["redirect_chain"],
        connect_elapsed_seconds=result["connect_elapsed_seconds"],
        read_elapsed_seconds=result["read_elapsed_seconds"],
        total_elapsed_seconds=result["total_elapsed_seconds"],
        status_code=result["status_code"],
        final_url=result["final_url"],
        body=result["body"],
        failure=None,
    )


def fetch_html_batch(urls: tuple[str, ...]) -> dict[str, catalogue.AuthoritativeHTMLPage]:
    ordered = tuple(sorted(catalogue.canonical_url(catalogue.ROOT_URL, url) for url in urls))
    attempts: dict[str, list[catalogue.HTMLPageAttempt]] = {url: [] for url in ordered}
    for attempt_index, delay in zip(catalogue.HTML_PAGE_ATTEMPT_INDICES, catalogue.HTML_PAGE_RETRY_DELAYS_SECONDS):
        if delay:
            time.sleep(delay)
        for url in ordered:
            attempts[url].append(_html_attempt(url, attempt_index, delay))
    return {
        url: catalogue.authoritative_html_page(url, tuple(attempts[url]))
        for url in ordered
    }


def fetch_catalogue() -> dict[str, catalogue.AuthoritativeHTMLPage]:
    root = catalogue.canonical_url(catalogue.ROOT_URL, catalogue.ROOT_URL)
    pages: dict[str, catalogue.AuthoritativeHTMLPage] = {}
    frontier = (root,)
    visited: set[str] = set()
    candidates: set[str] = set()
    for depth in range(catalogue.MAX_CRAWL_DEPTH + 1):
        pending = tuple(sorted(url for url in frontier if url not in visited))
        fetched = fetch_html_batch(pending)
        pages.update(fetched)
        next_frontier: set[str] = set()
        for url in pending:
            visited.add(url)
            traversal, found = catalogue.catalogue_links(
                pages[url].final_url, pages[url].body
            )
            candidates.update(found)
            if depth < catalogue.MAX_CRAWL_DEPTH:
                next_frontier.update(link for link in traversal if link not in visited)
        if len(visited | candidates | next_frontier) > catalogue.MAX_HTML_PAGES:
            raise OverflowError("complete catalogue HTML page limit exceeded")
        frontier = tuple(sorted(next_frontier))
    candidate_pending = tuple(sorted(candidates - set(pages)))
    if candidate_pending:
        pages.update(fetch_html_batch(candidate_pending))
    if len(pages) > catalogue.MAX_HTML_PAGES:
        raise OverflowError("complete catalogue HTML page limit exceeded")
    return pages


def load_contract(registry_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_ucr_catalogue_access":
        raise RuntimeError("v0.86 preregistration status changed")
    preaccess = json.loads(PREACCESS_RECORD.read_text(encoding="utf-8"))
    if preaccess["status"] != "authoritative_preaccess_contract_verified":
        raise RuntimeError("v0.87 authoritative preaccess record changed")
    if preaccess["implementation_commit"] != EXPECTED_IMPLEMENTATION_COMMIT:
        raise RuntimeError("v0.87 implementation anchor changed")
    registry_bytes = registry_path.read_bytes()
    if sha256_bytes(registry_bytes) != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("authoritative v0.86 registry bytes changed")
    registry = json.loads(registry_bytes.decode("utf-8"))
    if registry["status"] != "ucr_preblind_registry_v86_complete":
        raise RuntimeError("authoritative v0.86 registry is incomplete")
    if registry["registry_digest"] != EXPECTED_REGISTRY_DIGEST:
        raise RuntimeError("authoritative v0.86 registry digest changed")
    if preaccess["github_verification"]["registry_json_sha256"] != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("preaccess registry SHA anchor changed")
    if preaccess["github_verification"]["internal_registry_digest"] != EXPECTED_REGISTRY_DIGEST:
        raise RuntimeError("preaccess registry digest anchor changed")
    return preregistration, registry


def metadata_rank(metadata: catalogue.CandidateMetadata, seed: str, source_release: str) -> dict[str, object]:
    fields = metadata.canonical_fields()
    encoded = json.dumps(
        fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = sha256_bytes(encoded)
    rank_input = [
        _ascii_trim_nfc(seed),
        _ascii_trim_nfc(source_release),
        metadata.normalized_dataset_name,
        digest,
    ]
    rank_bytes = json.dumps(
        rank_input, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "metadata": fields,
        "canonical_metadata_digest": digest,
        "rank": sha256_bytes(rank_bytes),
    }


def _data_attempt(url: str, attempt_index: int, delay: int, spec: dict[str, object]) -> tuple[dict[str, object], bytes | None]:
    headers = tuple((str(k), str(v)) for k, v in spec["selected_file_request_headers"].items())
    try:
        result = _raw_get(
            url,
            headers=headers,
            connect_timeout=float(spec["selected_file_connect_timeout_seconds"]),
            read_timeout=float(spec["selected_file_read_timeout_seconds"]),
            total_timeout=float(spec["selected_file_total_attempt_deadline_seconds"]),
            maximum_redirects=int(spec["selected_file_maximum_redirects"]),
            maximum_bytes=int(spec["maximum_selected_file_bytes"]),
        )
    except Exception as error:
        return ({
            "attempt_index": attempt_index,
            "scheduled_delay_seconds": delay,
            "failure": _failure_text(error),
        }, None)
    body = result["body"]
    assert isinstance(body, bytes)
    return ({
        "attempt_index": attempt_index,
        "scheduled_delay_seconds": delay,
        "final_url": result["final_url"],
        "redirect_chain": list(result["redirect_chain"]),
        "connect_elapsed_seconds": list(result["connect_elapsed_seconds"]),
        "read_elapsed_seconds": result["read_elapsed_seconds"],
        "total_elapsed_seconds": result["total_elapsed_seconds"],
        "status_code": 200,
        "body_sha256": sha256_bytes(body),
        "body_bytes": len(body),
        "failure": None,
    }, body)


def fetch_selected_files(urls: tuple[str, ...], spec: dict[str, object]) -> dict[str, dict[str, object]]:
    ordered = tuple(sorted(catalogue.canonical_url(catalogue.ROOT_URL, url) for url in urls))
    delays = tuple(int(v) for v in spec["selected_file_retry_delays_seconds"])
    if delays != (0, 5, 20) or int(spec["selected_file_download_attempts"]) != 3:
        raise RuntimeError("selected-file retry schedule changed")
    records: dict[str, list[dict[str, object]]] = {url: [] for url in ordered}
    bodies: dict[str, list[bytes | None]] = {url: [] for url in ordered}
    for attempt_index, delay in enumerate(delays, start=1):
        if delay:
            time.sleep(delay)
        for url in ordered:
            record, body = _data_attempt(url, attempt_index, delay, spec)
            records[url].append(record)
            bodies[url].append(body)
    locked: dict[str, dict[str, object]] = {}
    for url in ordered:
        successes = [
            (record, body)
            for record, body in zip(records[url], bodies[url])
            if record["failure"] is None and body is not None
        ]
        if not successes:
            raise RuntimeError(f"all selected-file attempts failed: {url}")
        agreement = {
            (str(record["final_url"]), str(record["body_sha256"]))
            for record, _ in successes
        }
        if len(agreement) != 1:
            raise RuntimeError(f"successful selected-file attempts disagree: {url}")
        authoritative_record, authoritative_body = successes[0]
        if any(body != authoritative_body for _, body in successes):
            raise RuntimeError(f"successful selected-file bodies disagree: {url}")
        locked[url] = {
            "authoritative_attempt_index": authoritative_record["attempt_index"],
            "final_url": authoritative_record["final_url"],
            "body_sha256": authoritative_record["body_sha256"],
            "body_bytes": authoritative_record["body_bytes"],
            "attempts": records[url],
            "body": authoritative_body,
        }
    return locked


def _page_evidence(page: catalogue.AuthoritativeHTMLPage) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    for attempt in page.attempts:
        row: dict[str, object] = {
            "attempt_index": attempt.attempt_index,
            "scheduled_delay_seconds": attempt.scheduled_delay_seconds,
            "failure": attempt.failure,
        }
        if attempt.failure is None:
            assert attempt.body is not None
            row.update({
                "final_url": attempt.final_url,
                "redirect_chain": list(attempt.redirect_chain),
                "connect_elapsed_seconds": list(attempt.connect_elapsed_seconds),
                "read_elapsed_seconds": attempt.read_elapsed_seconds,
                "total_elapsed_seconds": attempt.total_elapsed_seconds,
                "status_code": attempt.status_code,
                "body_sha256": sha256_bytes(attempt.body),
                "body_bytes": len(attempt.body),
            })
        attempts.append(row)
    return {
        "requested_url": page.requested_url,
        "final_url": page.final_url,
        "body_sha256": page.body_sha256,
        "body_bytes": len(page.body),
        "authoritative_attempt_index": page.authoritative_attempt_index,
        "attempts": attempts,
    }


def execute_lock(registry_path: Path, output_dir: Path) -> dict[str, object]:
    preregistration, registry = load_contract(registry_path)
    selection = preregistration["future_metadata_only_selection"]
    if int(selection["dataset_count"]) != 7:
        raise RuntimeError("frozen UCR dataset count changed")
    if selection["selection_seed"] != "mini-origin-v87-ucr-untouched-lock-2026-07-31":
        raise RuntimeError("frozen UCR selection seed changed")
    pages = fetch_catalogue()
    exclusions = frozenset(str(value) for value in registry["excluded_dataset_names"])
    snapshot = catalogue.finalize_metadata_snapshot(
        catalogue.ROOT_URL, pages, exclusions
    )
    ranked = [
        metadata_rank(metadata, str(selection["selection_seed"]), snapshot.frozen_source_release)
        for metadata in snapshot.eligible_metadata
    ]
    ranked.sort(key=lambda row: (str(row["rank"]), str(row["metadata"]["normalized_dataset_name"])))
    if len(ranked) < 7:
        raise RuntimeError(f"only {len(ranked)} eligible untouched UCR datasets; seven required")
    selected = ranked[:7]
    selected_urls: list[str] = []
    for row in selected:
        metadata = row["metadata"]
        selected_urls.extend([str(metadata["train_url"]), str(metadata["test_url"])])
    if len(set(selected_urls)) != 14:
        raise RuntimeError("selected TRAIN/TEST URLs are not fourteen distinct files")
    byte_spec = selection["byte_lock_only"]
    locked_files = fetch_selected_files(tuple(selected_urls), byte_spec)

    output_dir.mkdir(parents=True, exist_ok=False)
    page_dir = output_dir / "source-pages"
    data_dir = output_dir / "selected-files"
    page_dir.mkdir()
    data_dir.mkdir()
    for page in pages.values():
        destination = page_dir / f"{page.body_sha256}.html"
        if destination.exists() and destination.read_bytes() != page.body:
            raise RuntimeError("source-page digest collision")
        destination.write_bytes(page.body)

    source_manifest_path = output_dir / "source-release-manifest.json"
    source_manifest_path.write_bytes(snapshot.source_manifest_bytes)
    transport = [_page_evidence(pages[url]) for url in sorted(pages)]
    (output_dir / "html-transport-evidence.json").write_text(
        json.dumps(transport, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    selected_records: list[dict[str, object]] = []
    for index, row in enumerate(selected, start=1):
        metadata = row["metadata"]
        name = str(metadata["normalized_dataset_name"])
        item = dict(row)
        file_records: dict[str, object] = {}
        for kind in ("train", "test"):
            url = str(metadata[f"{kind}_url"])
            locked = locked_files[url]
            body = locked.pop("body")
            assert isinstance(body, bytes)
            destination = data_dir / f"{index:02d}-{name}-{kind}.bin"
            destination.write_bytes(body)
            file_records[kind] = locked
        item["files"] = file_records
        selected_records.append(item)

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
    ).strip()
    manifest: dict[str, object] = {
        "status": "ucr_byte_lock_v87_complete",
        "execution_commit": head,
        "frozen_implementation_commit": EXPECTED_IMPLEMENTATION_COMMIT,
        "registry_json_sha256": EXPECTED_REGISTRY_SHA256,
        "registry_digest": EXPECTED_REGISTRY_DIGEST,
        "source": {
            "root_url": catalogue.ROOT_URL,
            "frozen_source_release": snapshot.frozen_source_release,
            "crawl_page_count": len(snapshot.crawl_page_urls),
            "candidate_page_count": len(snapshot.candidate_urls),
            "retained_page_count": len(pages),
            "source_manifest_sha256": sha256_bytes(snapshot.source_manifest_bytes),
        },
        "eligible_dataset_count": len(ranked),
        "selected_dataset_count": len(selected_records),
        "selected": selected_records,
        "rejections": [[url, list(reasons)] for url, reasons in snapshot.rejections],
        "access_boundary": {
            "selected_bytes_downloaded": True,
            "selected_bytes_decompressed": False,
            "records_or_labels_parsed": False,
            "solver_executed": False,
            "fresh_external_evaluation": False,
        },
        "claim_boundary": (
            "Deterministic untouched UCR metadata and byte lock only. This is not solver "
            "evaluation, independent replication, novelty proof, or a breakthrough."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = execute_lock(args.registry, args.output_dir)
    summary = {
        "status": manifest["status"],
        "execution_commit": manifest["execution_commit"],
        "eligible_dataset_count": manifest["eligible_dataset_count"],
        "selected_dataset_count": manifest["selected_dataset_count"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
