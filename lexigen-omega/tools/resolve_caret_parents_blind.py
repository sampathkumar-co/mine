from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def run_quiet(args: list[str], *, cwd: str | None = None) -> None:
    subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=False,
    )


def output_quiet(args: list[str], *, cwd: str | None = None) -> str:
    proc = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.stdout.strip()


def resolve(repo: str, child_sha: str) -> str:
    if not HEX40.fullmatch(child_sha):
        raise RuntimeError("child SHA is not canonical 40-hex")
    if repo.count("/") != 1 or any(x in repo for x in ("..", " ", "\n", "\r")):
        raise RuntimeError("unsafe repository identifier")
    remote = f"https://github.com/{repo}.git"
    with tempfile.TemporaryDirectory(prefix="lexigen-omega-parent-") as td:
        run_quiet(["git", "init", "-q"], cwd=td)
        run_quiet(
            ["git", "fetch", "--quiet", "--no-tags", "--depth=2", remote, child_sha],
            cwd=td,
        )
        parent = output_quiet(["git", "rev-parse", "FETCH_HEAD^"], cwd=td)
        if not HEX40.fullmatch(parent):
            raise RuntimeError("resolved parent is not canonical 40-hex")
        if parent == child_sha:
            raise RuntimeError("parent equals child")
        return parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen_before_parent_resolution":
        raise RuntimeError("parent-resolution lock not frozen")
    if any(lock.get("boundary", {}).values()):
        raise RuntimeError("parent-resolution boundary is already contaminated")
    contract = lock["resolver_contract"]
    if contract["allowed_git_query"] != "git rev-parse FETCH_HEAD^":
        raise RuntimeError("unexpected resolver query")
    if contract["network_fetch_output"] != "suppressed":
        raise RuntimeError("network output must be suppressed")
    if not contract["temporary_object_store_deleted_before_artifact_upload"]:
        raise RuntimeError("temporary object store deletion required")

    rows = []
    for target in lock["targets"]:
        child = str(target["child_sha"])
        expected = f"{child}^"
        if target["safe_metadata_base_commit"] != expected:
            raise RuntimeError(f"base syntax mismatch for {target['replication']}")
        parent = resolve(str(target["repo"]), child)
        rows.append(
            {
                "replication": target["replication"],
                "repo": target["repo"],
                "child_sha": child,
                "parent_sha": parent,
            }
        )

    result = {
        "project": "LEXIGEN OMEGA",
        "stage": lock["stage"],
        "status": "parent_sha_only_resolution_complete",
        "results": rows,
        "emitted_commit_messages": False,
        "emitted_changed_filenames": False,
        "emitted_patches": False,
        "emitted_source_contents": False,
        "emitted_tree_listings": False,
        "temporary_object_store_retained": False,
        "claim_boundary": lock["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
