#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXCLUDED_PARTS = {
    ".git",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".test-data",
    "__pycache__",
    "backups",
    "dist",
    "node_modules",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=20,
        )
        values = [root / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
        return sorted(path for path in values if path.is_file())
    except (OSError, subprocess.SubprocessError):
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and not (set(path.relative_to(root).parts) & EXCLUDED_PARTS)
        )


def git_sha(root: Path) -> str:
    configured = os.getenv("GITHUB_SHA") or os.getenv("DIRECTOR_RELEASE_SHA")
    if configured:
        return configured
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def backend_inventory(root: Path) -> dict[str, Any]:
    pyproject = root / "backend" / "pyproject.toml"
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = payload.get("project", {})
    return {
        "name": project.get("name"),
        "version": project.get("version"),
        "requires_python": project.get("requires-python"),
        "dependencies": project.get("dependencies", []),
        "development_dependencies": project.get("optional-dependencies", {}).get("dev", []),
    }


def frontend_inventory(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "frontend" / "package.json").read_text(encoding="utf-8"))
    return {
        "name": payload.get("name"),
        "version": payload.get("version"),
        "dependencies": payload.get("dependencies", {}),
        "development_dependencies": payload.get("devDependencies", {}),
    }


def build_manifest(root: Path) -> dict[str, Any]:
    version = (root / "VERSION").read_text(encoding="utf-8").strip()
    files = []
    for path in tracked_files(root):
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "product": "Director OS",
        "version": version,
        "git_sha": git_sha(root),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_file_count": len(files),
        "source_size_bytes": sum(item["size_bytes"] for item in files),
        "source_files": files,
        "backend": backend_inventory(root),
        "frontend": frontend_inventory(root),
        "qualification": {
            "normal_backend_ci": "required",
            "normal_frontend_ci": "required",
            "real_ffmpeg_media_render": "required",
            "full_stack_rehearsal": "required",
            "security_and_sbom": "required",
            "production_release_doctor": "required before deployment",
        },
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Director OS release manifest.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="dist/release-manifest.json")
    parser.add_argument("--checksums", default="dist/CHECKSUMS.sha256")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / args.output
    checksums = root / args.checksums
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums.write_text(
        f"{sha256_file(output)}  {output.relative_to(root).as_posix()}\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(f"Wrote {checksums}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
