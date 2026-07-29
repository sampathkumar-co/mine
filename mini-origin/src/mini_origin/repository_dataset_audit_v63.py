from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess


TEXT_SUFFIXES = {
    ".json",
    ".py",
    ".yml",
    ".yaml",
    ".md",
    ".txt",
    ".toml",
    ".csv",
}
MAX_FILE_BYTES = 2_000_000
MAX_OCCURRENCES_PER_KEY = 40

UCI_PATTERNS = (
    re.compile(r'["\']uci_id["\']\s*:\s*(\d+)'),
    re.compile(r"archive\.ics\.uci\.edu/static/public/(\d+)/", re.IGNORECASE),
    re.compile(r"cdn\.uci[^\s\"']*/(\d+)/", re.IGNORECASE),
    re.compile(r"\bUCI(?:\s+dataset)?\s*[-:#]?\s*(\d+)\b", re.IGNORECASE),
)
PYSTREED_PATTERN = re.compile(
    r"data/(?:cost-sensitive|classification)/([^/\s\"']+)",
    re.IGNORECASE,
)
NAME_ID_PATTERN = re.compile(
    r'["\']name["\']\s*:\s*["\']([^"\']+)["\'][\s\S]{0,300}?'
    r'["\']uci_id["\']\s*:\s*(\d+)',
    re.IGNORECASE,
)
ID_NAME_PATTERN = re.compile(
    r'["\']uci_id["\']\s*:\s*(\d+)[\s\S]{0,300}?'
    r'["\']name["\']\s*:\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout


def remote_refs() -> list[str]:
    rows = git(
        "for-each-ref",
        "--format=%(refname)",
        "refs/remotes/origin/",
    ).splitlines()
    return sorted(
        ref for ref in rows
        if ref and not ref.endswith("/HEAD")
    )


def tree_files(ref: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line in git("ls-tree", "-r", "--long", ref).splitlines():
        metadata, path = line.split("\t", 1)
        parts = metadata.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        size = int(parts[3])
        suffix = Path(path).suffix.lower()
        if suffix not in TEXT_SUFFIXES or size > MAX_FILE_BYTES:
            continue
        if not (
            path.startswith("mini-origin/")
            or path.startswith("research-evidence/")
            or path.startswith(".github/workflows/")
        ):
            continue
        rows.append((path, size))
    return rows


def show_text(ref: str, path: str) -> str:
    return git("show", f"{ref}:{path}")


def append_occurrence(
    mapping: dict[str, list[dict[str, object]]],
    key: str,
    occurrence: dict[str, object],
) -> None:
    rows = mapping[key]
    if len(rows) < MAX_OCCURRENCES_PER_KEY and occurrence not in rows:
        rows.append(occurrence)


def audit() -> dict[str, object]:
    refs = remote_refs()
    uci_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    token_occurrences: dict[str, list[dict[str, object]]] = defaultdict(list)
    names_by_id: dict[str, set[str]] = defaultdict(set)
    files_scanned = 0
    bytes_scanned = 0
    failures: list[dict[str, str]] = []

    for ref in refs:
        for path, size in tree_files(ref):
            try:
                text = show_text(ref, path)
            except subprocess.CalledProcessError as error:
                failures.append({
                    "ref": ref,
                    "path": path,
                    "error": error.stderr[-500:],
                })
                continue
            files_scanned += 1
            bytes_scanned += size
            occurrence = {"ref": ref, "path": path}

            ids = set()
            for pattern in UCI_PATTERNS:
                ids.update(pattern.findall(text))
            for uci_id in sorted(ids, key=int):
                append_occurrence(uci_occurrences, uci_id, occurrence)

            for name, uci_id in NAME_ID_PATTERN.findall(text):
                names_by_id[uci_id].add(name.strip())
                append_occurrence(uci_occurrences, uci_id, occurrence)
            for uci_id, name in ID_NAME_PATTERN.findall(text):
                names_by_id[uci_id].add(name.strip())
                append_occurrence(uci_occurrences, uci_id, occurrence)

            for token in PYSTREED_PATTERN.findall(text):
                canonical = token.strip().lower().replace("_", "-")
                append_occurrence(token_occurrences, canonical, occurrence)

    uci_rows = [
        {
            "uci_id": int(uci_id),
            "names": sorted(names_by_id.get(uci_id, set())),
            "occurrence_count_capped": len(occurrences),
            "occurrences": occurrences,
        }
        for uci_id, occurrences in sorted(
            uci_occurrences.items(), key=lambda item: int(item[0])
        )
    ]
    token_rows = [
        {
            "token": token,
            "occurrence_count_capped": len(occurrences),
            "occurrences": occurrences,
        }
        for token, occurrences in sorted(token_occurrences.items())
    ]
    protocol = {
        "refs": refs,
        "text_suffixes": sorted(TEXT_SUFFIXES),
        "max_file_bytes": MAX_FILE_BYTES,
        "scanned_roots": [
            "mini-origin/",
            "research-evidence/",
            ".github/workflows/",
        ],
        "uci_patterns": [pattern.pattern for pattern in UCI_PATTERNS],
        "pystreed_pattern": PYSTREED_PATTERN.pattern,
    }
    digest = hashlib.sha256(
        json.dumps(
            {
                "protocol": protocol,
                "uci_rows": uci_rows,
                "token_rows": token_rows,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    known_checks = {
        "uci_3_annealing": any(row["uci_id"] == 3 for row in uci_rows),
        "uci_83_primary_tumor": any(row["uci_id"] == 83 for row in uci_rows),
        "uci_90_soybean_large": any(row["uci_id"] == 90 for row in uci_rows),
        "pystreed_annealing": any(
            row["token"] == "annealing" for row in token_rows
        ),
    }
    return {
        "status": (
            "repository_dataset_registry_complete"
            if all(known_checks.values()) and not failures
            else "repository_dataset_registry_incomplete"
        ),
        "protocol": protocol,
        "ref_count": len(refs),
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "uci_id_count": len(uci_rows),
        "pystreed_token_count": len(token_rows),
        "known_contamination_checks": known_checks,
        "failures": failures,
        "registry_digest": digest,
        "uci_datasets": uci_rows,
        "pystreed_dataset_tokens": token_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "refs": report["ref_count"],
        "files": report["files_scanned"],
        "uci_ids": report["uci_id_count"],
        "pystreed_tokens": report["pystreed_token_count"],
        "known_checks": report["known_contamination_checks"],
    }, indent=2))
    if report["status"] != "repository_dataset_registry_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
