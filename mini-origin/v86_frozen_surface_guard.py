#!/usr/bin/env python3
"""Fail-closed guard for the Mini-ORIGIN v0.86 frozen scientific surface.

The v0.86 protocol permits only pre-access protocol, manifest, gate, guard, and
workflow files to differ from the frozen v0.85 parent. Any change to the frozen
compiler, selectors, planners, solver, campaign evidence, thresholds, budgets,
or prior negative-result records is an automatic protocol violation.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

FROZEN_PARENT = "912c3ebd933ae39eb05e10467f1ecad56e326b03"

ALLOWED_CHANGED_PATHS = frozenset(
    {
        ".github/workflows/mini-origin-v86-preaccess.yml",
        "mini-origin/campaigns/v86-untouched-external-validation.json",
        "mini-origin/v86_frozen_surface_guard.py",
        "mini-origin/v86_manifest_generator.py",
        "mini-origin/v86_transferable_gate.py",
    }
)

PROTECTED_EVIDENCE_PATHS = frozenset(
    {
        "research-evidence/mini-origin-v82-pmlb-blind-rejection.json",
        "research-evidence/mini-origin-v83-near-small-query-coverage-rejection.json",
        "research-evidence/mini-origin-v84-partition-signature-coverage-rejection.json",
        "research-evidence/mini-origin-v85-authoritative-opened-data-development-pass.json",
        "research-evidence/mini-origin-v85-exact-rerun-reproducibility.json",
    }
)


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def changed_paths(parent: str = FROZEN_PARENT, head: str = "HEAD") -> set[str]:
    output = _git("diff", "--name-only", "--diff-filter=ACDMRTUXB", parent, head)
    return {line.strip() for line in output.splitlines() if line.strip()}


def validate_changed_paths(paths: Iterable[str]) -> None:
    paths = set(paths)
    unexpected = sorted(paths - ALLOWED_CHANGED_PATHS)
    if unexpected:
        raise RuntimeError(
            "frozen v0.85 scientific surface changed outside the preregistered "
            "v0.86 pre-access files: " + ", ".join(unexpected)
        )

    missing_guard = "mini-origin/v86_frozen_surface_guard.py" not in paths
    if missing_guard:
        raise RuntimeError("guard file is not part of the v0.86 change set")


def verify_protected_evidence(parent: str = FROZEN_PARENT, head: str = "HEAD") -> None:
    changed = changed_paths(parent, head)
    modified_evidence = sorted(changed & PROTECTED_EVIDENCE_PATHS)
    if modified_evidence:
        raise RuntimeError(
            "binding negative/opened-data evidence was modified: "
            + ", ".join(modified_evidence)
        )

    missing = sorted(path for path in PROTECTED_EVIDENCE_PATHS if not Path(path).is_file())
    if missing:
        raise RuntimeError("protected evidence file missing from checkout: " + ", ".join(missing))


def verify_repository(parent: str = FROZEN_PARENT, head: str = "HEAD") -> None:
    _git("cat-file", "-e", f"{parent}^{{commit}}")
    paths = changed_paths(parent, head)
    validate_changed_paths(paths)
    verify_protected_evidence(parent, head)
    print(f"frozen surface guard: PASS ({len(paths)} allowed changed paths)")


def self_test() -> None:
    validate_changed_paths(ALLOWED_CHANGED_PATHS)

    try:
        validate_changed_paths(ALLOWED_CHANGED_PATHS | {"mini-origin/compiler.py"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("unexpected frozen-source change was accepted")

    try:
        validate_changed_paths(ALLOWED_CHANGED_PATHS | {next(iter(PROTECTED_EVIDENCE_PATHS))})
    except RuntimeError:
        pass
    else:
        raise AssertionError("evidence mutation was accepted")

    print("v0.86 frozen surface guard self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--parent", default=FROZEN_PARENT)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        verify_repository(args.parent, args.head)


if __name__ == "__main__":
    main()
