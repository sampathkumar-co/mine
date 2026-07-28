from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp",
    ".py", ".rs", ".java", ".scala", ".jl",
}
MAX_FILE_BYTES = 2_000_000

# Broad retrieval terms. These collect candidate evidence; they do not decide
# novelty by themselves.
SEARCH_TERMS = (
    "equivalent", "equivalence", "identical", "duplicate", "redundant",
    "similarity", "similar support", "support", "partition", "feature",
    "attribute", "test", "split", "branch", "cache", "lower bound",
)

# A plausible implementation of the v0.44 mechanism should discuss a query-like
# object and equality/canonicalisation of the response partition it induces in
# the same local context. This deliberately errs toward false positives.
QUERY_TERMS = re.compile(
    r"\b(feature|attribute|test|query|split|predicate|column)\w*\b",
    re.IGNORECASE,
)
PARTITION_TERMS = re.compile(
    r"\b(partition|support|mask|outcome|response|child(?:ren)?|subset)\w*\b",
    re.IGNORECASE,
)
EQUIVALENCE_TERMS = re.compile(
    r"\b(equivalent|identical|duplicate|redundant|canonical|dedup|same)\w*\b",
    re.IGNORECASE,
)
ACTION_TERMS = re.compile(
    r"\b(skip|remove|prune|collapse|merge|representative|canonical|ignore|dedup)\w*\b",
    re.IGNORECASE,
)
SUBPROBLEM_TERMS = re.compile(
    r"\b(subproblem|data(?:set|view)?|branch|cache|archive|assignment|lower.?bound)\w*\b",
    re.IGNORECASE,
)
INSTANCE_TERMS = re.compile(
    r"\b(instance|sample|row|point|observation|record)\w*\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RepositorySpec:
    name: str
    path: str
    source_url: str
    expected_commit: str | None
    family: str


@dataclass(frozen=True)
class EvidenceHit:
    repository: str
    path: str
    line: int
    category: str
    excerpt: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        parts = set(path.parts)
        if parts & {".git", "build", "dist", "vendor", "third_party", "node_modules"}:
            continue
        yield path


def normalise_excerpt(lines: list[str], index: int, radius: int = 2) -> str:
    start = max(0, index - radius)
    end = min(len(lines), index + radius + 1)
    return "\n".join(
        f"{line_no + 1}: {lines[line_no].rstrip()}"
        for line_no in range(start, end)
    )[:2400]


def classify_window(text: str) -> set[str]:
    categories: set[str] = set()
    query = bool(QUERY_TERMS.search(text))
    partition = bool(PARTITION_TERMS.search(text))
    equivalent = bool(EQUIVALENCE_TERMS.search(text))
    action = bool(ACTION_TERMS.search(text))
    subproblem = bool(SUBPROBLEM_TERMS.search(text))
    instance = bool(INSTANCE_TERMS.search(text))

    if subproblem and (equivalent or "similar" in text.lower()):
        categories.add("subproblem_similarity_or_cache_equivalence")
    if instance and (equivalent or "similar support" in text.lower()):
        categories.add("instance_or_support_equivalence")
    if query and partition and equivalent:
        categories.add("possible_local_test_partition_equivalence")
    if query and partition and equivalent and action:
        categories.add("strong_local_test_partition_candidate")
    return categories


def audit_repository(spec: RepositorySpec, base: Path) -> dict[str, object]:
    root = base / spec.path
    if not root.exists():
        return {
            "name": spec.name,
            "family": spec.family,
            "source_url": spec.source_url,
            "available": False,
            "reason": "repository directory not present",
            "expected_commit": spec.expected_commit,
        }

    actual_commit = git_value(root, "rev-parse", "HEAD")
    remote = git_value(root, "remote", "get-url", "origin")
    hits: list[EvidenceHit] = []
    files_scanned = 0
    lines_scanned = 0
    keyword_file_count = 0
    tree_digest = hashlib.sha256()

    for path in source_files(root):
        files_scanned += 1
        relative = path.relative_to(root).as_posix()
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(sha256_file(path).encode("ascii"))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        lines_scanned += len(lines)
        file_has_keyword = False
        for index, line in enumerate(lines):
            lowered = line.lower()
            if not any(term in lowered for term in SEARCH_TERMS):
                continue
            file_has_keyword = True
            start = max(0, index - 5)
            end = min(len(lines), index + 6)
            window = "\n".join(lines[start:end])
            categories = classify_window(window)
            for category in sorted(categories):
                hits.append(EvidenceHit(
                    repository=spec.name,
                    path=relative,
                    line=index + 1,
                    category=category,
                    excerpt=normalise_excerpt(lines, index),
                ))
        keyword_file_count += int(file_has_keyword)

    # Deduplicate overlapping windows while preserving the first exact location.
    unique: dict[tuple[str, str, int], EvidenceHit] = {}
    for hit in hits:
        key = (hit.path, hit.category, hit.line)
        unique.setdefault(key, hit)
    hits = sorted(unique.values(), key=lambda row: (row.category, row.path, row.line))

    category_counts: dict[str, int] = {}
    for hit in hits:
        category_counts[hit.category] = category_counts.get(hit.category, 0) + 1

    strong = [
        hit for hit in hits
        if hit.category == "strong_local_test_partition_candidate"
    ]
    possible = [
        hit for hit in hits
        if hit.category == "possible_local_test_partition_equivalence"
    ]
    return {
        "name": spec.name,
        "family": spec.family,
        "source_url": spec.source_url,
        "available": True,
        "expected_commit": spec.expected_commit,
        "actual_commit": actual_commit,
        "commit_matches": (
            actual_commit == spec.expected_commit
            if spec.expected_commit is not None else None
        ),
        "origin": remote,
        "files_scanned": files_scanned,
        "lines_scanned": lines_scanned,
        "keyword_file_count": keyword_file_count,
        "source_tree_digest": tree_digest.hexdigest(),
        "category_counts": category_counts,
        "strong_candidate_count": len(strong),
        "possible_candidate_count": len(possible),
        "hits": [asdict(hit) for hit in hits[:500]],
        "truncated_hit_count": max(0, len(hits) - 500),
    }


def run(manifest_path: Path, source_root: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = [RepositorySpec(**row) for row in manifest["repositories"]]
    repositories = [audit_repository(spec, source_root) for spec in specs]
    available = [row for row in repositories if row["available"]]
    pinned = [
        row for row in available
        if row["expected_commit"] is not None
    ]
    strong_candidates = sum(
        int(row.get("strong_candidate_count", 0)) for row in available
    )
    possible_candidates = sum(
        int(row.get("possible_candidate_count", 0)) for row in available
    )
    gate = (
        len(available) >= 4
        and all(row["commit_matches"] for row in pinned)
        and all(int(row["files_scanned"]) > 0 for row in available)
    )
    return {
        "status": "audit_completed" if gate else "audit_incomplete",
        "audit_gate": gate,
        "method": {
            "scope": (
                "Pinned public source snapshots are scanned for within-state "
                "test/query equivalence implemented through equality of induced "
                "response partitions. Subproblem similarity and instance/support "
                "equivalence are reported separately."
            ),
            "important_limitation": (
                "A source audit cannot prove universal novelty. Zero candidates "
                "means no equivalent implementation was found by this transparent "
                "scan in the pinned snapshots; manual review remains required."
            ),
            "strong_candidate_rule": (
                "A local context contains a query-like term, partition/support/mask "
                "term, equivalence/canonicalisation term, and a skip/remove/prune/"
                "representative action term."
            ),
        },
        "repository_count": len(repositories),
        "available_repository_count": len(available),
        "strong_candidate_count": strong_candidates,
        "possible_candidate_count": possible_candidates,
        "repositories": repositories,
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.manifest, args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "available_repositories": report["available_repository_count"],
        "strong_candidates": report["strong_candidate_count"],
        "possible_candidates": report["possible_candidate_count"],
    }, indent=2))
    if not report["audit_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
