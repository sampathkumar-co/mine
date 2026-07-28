from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

from selector import SOURCE_COMMIT

URL = f"https://api.github.com/repos/oripress/AlgoTune/git/trees/{SOURCE_COMMIT}?recursive=1"


def main() -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "LEXIGEN-v4-source-layout", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=240) as response:
        raw = response.read()
    payload = json.loads(raw)
    tree = payload.get("tree")
    if not isinstance(tree, list) or payload.get("truncated"):
        raise RuntimeError("source tree metadata unavailable or truncated")

    paths = sorted(
        str(entry.get("path", ""))
        for entry in tree
        if isinstance(entry, dict)
        and entry.get("type") == "blob"
        and str(entry.get("path", "")).startswith("AlgoTuneTasks/")
    )
    depth_counts = Counter(len(path.split("/")) for path in paths)
    suffix_counts = Counter(Path(path).suffix for path in paths)
    basename_counts = Counter(Path(path).name for path in paths)
    report = {
        "source_commit": SOURCE_COMMIT,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "tree_sha": payload.get("sha"),
        "algotune_task_blob_count": len(paths),
        "depth_counts": {str(key): value for key, value in sorted(depth_counts.items())},
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "most_common_basenames": basename_counts.most_common(30),
        "paths": paths,
        "file_contents_opened": False,
        "task_source_opened": False
    }
    output = Path("source-layout-diagnostic-evidence")
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-layout.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"algotune_task_blob_count": len(paths), "depth_counts": report["depth_counts"], "suffix_counts": report["suffix_counts"]}, indent=2))


if __name__ == "__main__":
    main()
