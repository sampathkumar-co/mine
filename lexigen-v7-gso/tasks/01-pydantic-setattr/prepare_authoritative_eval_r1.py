from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from datasets import load_dataset

INSTANCE_ID = "pydantic__pydantic-addf1f9"
DATASET_REVISION = "c2e4f1a58427cccd15e0e542f136bd204fb19284"
IMAGE = "slimshetty/gso:gso.eval.x86_64.pydantic__pydantic-addf1f9"
SELECTED = {
    "v7_full": ("F1", "63d68ecb88a6911926a5d60236808643d507d07a1e2fc19a943b62fee5bebaae"),
    "v7_no_library": ("N2", "83ea4aac767729ba325b2bb0d7535eb9fc53b494d647236494c90edde52253a2"),
    "v7_random_library": ("R1", "83ea4aac767729ba325b2bb0d7535eb9fc53b494d647236494c90edde52253a2"),
}


def run(cmd: list[str], *, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=capture, check=check)


def make_patch(builder: Path, arm: str, candidate: str, expected_sha: str, output: Path) -> Path:
    name = f"lexigen-v7-authoritative-{candidate.lower()}"
    run(["docker", "rm", "-f", name], check=False, capture=True)
    run(["docker", "create", "--name", name, IMAGE, "tail", "-f", "/dev/null"])
    run(["docker", "start", name])
    try:
        run(["docker", "cp", str(builder.resolve()), f"{name}:/tmp/build_candidate.py"])
        run(["docker", "exec", name, "bash", "-lc", f"python /tmp/build_candidate.py --root /testbed --candidate {candidate}"])
        run(["docker", "exec", name, "bash", "-lc", "python -m py_compile /testbed/pydantic/main.py /testbed/pydantic/_internal/_model_construction.py"])
        patch = run(
            ["docker", "exec", name, "bash", "-lc", "cd /testbed && git diff -- pydantic/main.py pydantic/_internal/_model_construction.py"],
            capture=True,
        ).stdout
        if not patch.strip():
            raise RuntimeError(f"empty frozen patch for {arm}/{candidate}")
        digest = hashlib.sha256(patch.encode()).hexdigest()
        if digest != expected_sha:
            raise RuntimeError(f"frozen patch digest mismatch for {arm}/{candidate}: {digest} != {expected_sha}")
        patch_path = output / "patches" / f"{arm}.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(patch, encoding="utf-8")
        pred_path = output / "predictions" / f"{arm}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pred = {
            "instance_id": INSTANCE_ID,
            "model_patch": patch,
            "model_name_or_path": f"LEXIGEN-V7-{arm}-{candidate}",
        }
        pred_path.write_text(json.dumps(pred) + "\n", encoding="utf-8")
        return pred_path
    finally:
        run(["docker", "rm", "-f", name], check=False, capture=True)


def materialize_pinned_dataset(dataset_output: Path) -> None:
    ds = load_dataset("gso-bench/gso", revision=DATASET_REVISION, split="test")
    rows = [dict(row) for row in ds if row["instance_id"] == INSTANCE_ID]
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one pinned GSO row for {INSTANCE_ID}, got {len(rows)}")
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.write_text(json.dumps(rows[0], default=str) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--builder", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dataset-output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run(["docker", "pull", IMAGE])
    materialize_pinned_dataset(args.dataset_output)
    manifest = {
        "instance_id": INSTANCE_ID,
        "dataset_revision": DATASET_REVISION,
        "docker_image": IMAGE,
        "selected": {},
        "search_closed": True,
        "expert_reference_execution_allowed_only_in_official_grader": True,
        "expert_diff_used_for_candidate_generation": False,
        "hints_used_for_candidate_generation": False,
    }
    for arm, (candidate, expected_sha) in SELECTED.items():
        make_patch(args.builder, arm, candidate, expected_sha, args.output)
        manifest["selected"][arm] = {"candidate_id": candidate, "patch_sha256": expected_sha}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
