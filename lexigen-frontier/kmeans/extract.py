from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPORT_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
EXPECTED_GIT_BLOB_SHA1 = "c603aa3341133ba43725b64692c08e89760d8654"
BLOB_API = f"https://api.github.com/repos/oripress/AlgoTune/git/blobs/{EXPECTED_GIT_BLOB_SHA1}"
TASK = "kmeans"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_blob() -> bytes:
    last_error: Exception | None = None
    for delay in (0, 3, 10, 30):
        if delay:
            time.sleep(delay)
        request = urllib.request.Request(
            BLOB_API,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "LEXIGEN-task-scoped-frontier",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.loads(response.read())
            if str(payload.get("sha")) != EXPECTED_GIT_BLOB_SHA1:
                raise RuntimeError("GitHub blob response SHA changed")
            if payload.get("encoding") != "base64":
                raise RuntimeError(f"unexpected blob encoding {payload.get('encoding')!r}")
            raw = base64.b64decode(str(payload["content"]), validate=False)
            actual = git_blob(raw)
            if actual != EXPECTED_GIT_BLOB_SHA1:
                raise RuntimeError(f"decoded report identity changed: {actual}")
            return raw
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"pinned blob fetch failed: {type(last_error).__name__}: {last_error}")


def collect(value: Any, path: list[str], matches: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + [str(key)]
            if str(key) == TASK:
                matches.append({"path": child_path, "value": child})
            elif str(child) == TASK and str(key) in {"task", "task_name", "name", "problem"}:
                matches.append({"path": path, "value": value})
            collect(child, child_path, matches)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            collect(child, path + [str(index)], matches)


def main() -> None:
    diagnostic: dict[str, Any] = {
        "task": TASK,
        "report_repository": "oripress/AlgoTune",
        "report_commit": REPORT_COMMIT,
        "expected_report_git_blob_sha1": EXPECTED_GIT_BLOB_SHA1,
        "unrelated_report_sections_emitted": False,
    }
    try:
        raw = fetch_blob()
        report = json.loads(raw)
        matches: list[dict[str, Any]] = []
        collect(report, [], matches)
        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for match in matches:
            encoded = json.dumps(match, sort_keys=True, separators=(",", ":"))
            if encoded not in seen:
                seen.add(encoded)
                unique.append(match)
        diagnostic.update({
            "status": "success" if unique else "no_match",
            "report_git_blob_sha1": git_blob(raw),
            "match_count": len(unique),
            "matches": unique,
        })
    except Exception as exc:
        diagnostic.update({
            "status": "infrastructure_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "match_count": 0,
            "matches": [],
        })
    Path("kmeans-frontier.json").write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "task": TASK,
        "status": diagnostic["status"],
        "match_count": diagnostic["match_count"],
        "unrelated_report_sections_emitted": False,
    }, indent=2))
    if diagnostic["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
