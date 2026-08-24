from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

TEXT_SUFFIXES = {".py", ".pyi", ".pyx", ".pxd", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".rs", ".toml", ".md", ".txt"}
MAX_FILE_BYTES = 2_000_000
MAX_MATCHED_FILES = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    primary = [tail]
    fallback = [owner] if owner and owner != tail else []

    def candidates(tokens: list[str]) -> list[Path]:
        matches: list[Path] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file() or ".git" in p.parts or p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if p.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(token and token in text for token in tokens):
                matches.append(p)
                if len(matches) >= MAX_MATCHED_FILES:
                    break
        return matches

    matched = candidates(primary)
    if not matched and fallback:
        matched = candidates(fallback)
    if not matched:
        raise RuntimeError(f"no base-worktree source context matched API {args.api}")
    manifest = []
    for src in matched:
        rel = src.relative_to(root)
        dest = out / "files" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        manifest.append({"path": str(rel), "bytes": src.stat().st_size, "sha256": sha256(src), "contains_primary_token": tail in src.read_text(encoding="utf-8", errors="ignore")})
    report = {"api": args.api, "primary_token": tail, "fallback_token": owner or None, "matched_file_count": len(manifest), "matched_files": manifest, "git_directory_scanned": False, "git_objects_read": False, "expert_diff_accessed": False}
    (out / "worktree-context.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
