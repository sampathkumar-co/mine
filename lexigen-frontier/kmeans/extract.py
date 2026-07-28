from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

REPORT_COMMIT = "dff9914c10800c7a031c9e8c3d4d1c8cd1b38906"
REPORT_URL = f"https://raw.githubusercontent.com/oripress/AlgoTune/{REPORT_COMMIT}/reports/agent_summary.json"
EXPECTED_GIT_BLOB_SHA1 = "c603aa3341133ba43725b64692c08e89760d8654"
TASK = "kmeans"


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


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
    request = urllib.request.Request(REPORT_URL, headers={"User-Agent": "LEXIGEN-task-scoped-frontier"})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    actual = git_blob(raw)
    if actual != EXPECTED_GIT_BLOB_SHA1:
        raise RuntimeError(f"official report identity changed: {actual} != {EXPECTED_GIT_BLOB_SHA1}")
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
    if not unique:
        raise RuntimeError("no exact kmeans frontier record found")
    output = {
        "task": TASK,
        "report_repository": "oripress/AlgoTune",
        "report_commit": REPORT_COMMIT,
        "report_git_blob_sha1": actual,
        "matches": unique,
        "unrelated_report_sections_emitted": false,
    }
    Path("kmeans-frontier.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task": TASK, "match_count": len(unique)}, indent=2))


if __name__ == "__main__":
    main()
