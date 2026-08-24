from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

TEXT_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".rs", ".toml", ".md", ".txt"}
MAX_FILE_BYTES = 2_000_000
MAX_MATCHED_FILES = 32
IGNORED_PARTS = {".git", ".venv", "venv", "site-packages", "build", "dist", "__pycache__", ".tox", ".nox"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def score(path: Path, text: str, tail: str, owner: str) -> tuple[int, str]:
    rel = str(path).lower()
    s = 0
    if f"def {tail}" in text or f"fn {tail}" in text or f" {tail}(" in text:
        s += 20
    if owner and owner in text:
        s += 5
    if any(part in rel for part in ("/tests/", "/test_", "/doc/", "/docs/")):
        s -= 4
    if any(part in rel for part in ("/src/", "/numpy/", "/pandas/", "/llama_cpp/", "/tokenizers/", "/transformers/")):
        s += 3
    return (-s, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tail = args.api.split(".")[-1]
    owner = args.api.split(".")[-2] if "." in args.api else ""

    def collect(token: str) -> list[tuple[Path, str]]:
        hits: list[tuple[Path, str]] = []
        for p in root.rglob("*"):
            if not p.is_file() or any(part in IGNORED_PARTS for part in p.parts):
                continue
            if p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if token and token in text:
                hits.append((p, text))
        hits.sort(key=lambda pair: score(pair[0].relative_to(root), pair[1], tail, owner))
        return hits[:MAX_MATCHED_FILES]

    matched = collect(tail)
    if not matched and owner and owner != tail:
        matched = collect(owner)
    if not matched:
        raise RuntimeError(f"no project-source context matched API {args.api}")

    manifest = []
    for src, text in matched:
        rel = src.relative_to(root)
        dest = out / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        manifest.append({"path": str(rel), "bytes": src.stat().st_size, "sha256": sha256(src), "contains_primary_token": tail in text})
    report = {
        "api": args.api,
        "primary_token": tail,
        "fallback_token": owner or None,
        "matched_file_count": len(manifest),
        "matched_files": manifest,
        "ignored_parts": sorted(IGNORED_PARTS),
        "virtualenv_or_site_packages_scanned": False,
        "git_directory_scanned": False,
        "git_objects_read": False,
        "expert_diff_accessed": False
    }
    (out / "worktree-context.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
