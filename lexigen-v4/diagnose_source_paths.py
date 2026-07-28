from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

from selector import SOURCE_COMMIT

URL = f"https://api.github.com/repos/oripress/AlgoTune/git/trees/{SOURCE_COMMIT}?recursive=1"


def main() -> None:
    request = urllib.request.Request(
        URL,
        headers={"User-Agent": "LEXIGEN-v4-source-path-diagnostic", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        raw = response.read()
    payload = json.loads(raw)
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError("source tree metadata lacks a tree list")

    blobs = [str(entry.get("path", "")) for entry in tree if isinstance(entry, dict) and entry.get("type") == "blob"]
    task_suffix = sorted(path for path in blobs if path.lower().endswith("task.py"))
    task_named = sorted(path for path in blobs if "task" in Path(path).name.lower() and path.lower().endswith(".py"))
    roots = Counter(path.split("/", 1)[0] for path in blobs if path)
    depth_counts = Counter(len(path.split("/")) for path in task_suffix)

    report = {
        "source_commit": SOURCE_COMMIT,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "tree_sha": payload.get("sha"),
        "truncated": payload.get("truncated"),
        "blob_count": len(blobs),
        "top_level_blob_roots": dict(sorted(roots.items())),
        "task_py_path_count": len(task_suffix),
        "task_py_depth_counts": {str(key): value for key, value in sorted(depth_counts.items())},
        "task_py_paths": task_suffix,
        "other_task_named_python_paths": task_named,
        "file_contents_opened": False,
        "task_source_opened": False,
    }
    output = Path("source-path-diagnostic-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-paths.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"blob_count": len(blobs), "task_py_path_count": len(task_suffix), "task_py_depth_counts": report["task_py_depth_counts"]}, indent=2))


if __name__ == "__main__":
    main()
