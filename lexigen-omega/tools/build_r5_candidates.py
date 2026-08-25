from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
from pathlib import Path

TARGET = Path("PIL/Image.py")
CANDIDATES = ("F1", "F2", "F4", "N1", "N3", "N6", "R1", "R3", "R6")
PROPOSAL_BLOB = "1cbfca32e5e37cddd4ae9cb771058a65e5ad1828"
EXPECTED_BASE_BLOB = "49c9f6ab204a16044355d1d043c16ecef9e044a4"

ORIGINAL = '''        self.load()\n        if self.im.bands == 1:\n            ims = [self.copy()]\n        else:\n            ims = []\n            for i in range(self.im.bands):\n                ims.append(self._new(self.im.getband(i)))\n        return tuple(ims)\n'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one replacement target, got {count}")
    return text.replace(old, new, 1)


def write_patch(path: Path, before: str, after: str, patch_path: Path) -> None:
    diff = difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=f"a/{path.as_posix()}", tofile=f"b/{path.as_posix()}"
    )
    patch_path.write_text("".join(diff), encoding="utf-8")


def verify_protocol_inputs() -> None:
    got = subprocess.check_output(
        ["git", "hash-object", "lexigen-omega/evidence/PROSPECTIVE_R5_SOURCE_PROPOSALS.json"],
        text=True,
    ).strip()
    if got != PROPOSAL_BLOB:
        raise RuntimeError(f"R5 proposal blob drift: {got} != {PROPOSAL_BLOB}")


def transform(before: str, cid: str) -> str:
    if cid == "F1":
        new = '''        self.load()\n        im = self.im\n        band_count = im.bands\n        if band_count == 1:\n            ims = [self.copy()]\n        else:\n            ims = [self._new(im.getband(i)) for i in range(band_count)]\n        return tuple(ims)\n'''
    elif cid == "F2":
        new = '''        self.load()\n        im = self.im\n        if im.bands == 1:\n            try:\n                ims = [self._new(im.copy())]\n            except AttributeError:\n                ims = [self.copy()]\n        else:\n            ims = []\n            for i in range(im.bands):\n                ims.append(self._new(im.getband(i)))\n        return tuple(ims)\n'''
    elif cid == "F4":
        new = '''        self.load()\n        if self.im.bands == 1:\n            return (self.copy(),)\n        ims = []\n        for i in range(self.im.bands):\n            ims.append(self._new(self.im.getband(i)))\n        return tuple(ims)\n'''
    elif cid == "N1":
        new = '''        self.load()\n        im = self.im\n        new_image = self._new\n        band_count = im.bands\n        if band_count == 1:\n            return (self.copy(),)\n        if band_count == 3:\n            return (new_image(im.getband(0)), new_image(im.getband(1)), new_image(im.getband(2)))\n        if band_count == 4:\n            return (\n                new_image(im.getband(0)), new_image(im.getband(1)),\n                new_image(im.getband(2)), new_image(im.getband(3))\n            )\n        ims = []\n        for i in range(band_count):\n            ims.append(new_image(im.getband(i)))\n        return tuple(ims)\n'''
    elif cid == "N3":
        new = '''        self.load()\n        im = self.im\n        new_image = self._new\n        band_count = im.bands\n        if band_count == 1:\n            ims = [self.copy()]\n        else:\n            ims = []\n            for i in range(band_count):\n                ims.append(new_image(im.getband(i)))\n        return tuple(ims)\n'''
    elif cid == "N6":
        new = '''        self.load()\n        band_count = self.im.bands\n        if band_count == 1:\n            ims = [self.copy()]\n        else:\n            ims = []\n            for i in range(band_count):\n                ims.append(self._new(self.im.getband(i)))\n        return tuple(ims)\n'''
    elif cid == "R1":
        new = '''        self.load()\n        im = self.im\n        new_image = self._new\n        band_count = im.bands\n        if band_count == 1:\n            return (self.copy(),)\n        return tuple([new_image(im.getband(i)) for i in range(band_count)])\n'''
    elif cid == "R3":
        new = '''        self.load()\n        if self.im.bands == 1:\n            return (self._new(self.im.copy()),)\n        ims = []\n        for i in range(self.im.bands):\n            ims.append(self._new(self.im.getband(i)))\n        return tuple(ims)\n'''
    elif cid == "R6":
        new = '''        self.load()\n        im = self.im\n        band_count = im.bands\n        if band_count == 1:\n            return (self.copy(),)\n        getband = im.getband\n        new_image = self._new\n        return tuple(new_image(getband(i)) for i in range(band_count))\n'''
    else:
        raise RuntimeError(f"unknown candidate {cid}")
    after = replace_once(before, ORIGINAL, new, cid)
    if after == before:
        raise RuntimeError(f"{cid}: transform made no change")
    return after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=CANDIDATES, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    verify_protocol_inputs()
    path = args.root / TARGET
    before = path.read_text(encoding="utf-8")
    base_blob = subprocess.check_output(["git", "hash-object", str(path)], text=True).strip()
    if base_blob != EXPECTED_BASE_BLOB:
        raise RuntimeError(f"R5 base source drift: {base_blob} != {EXPECTED_BASE_BLOB}")

    after = transform(before, args.candidate)
    path.write_text(after, encoding="utf-8")

    args.output.mkdir(parents=True, exist_ok=True)
    patch_path = args.output / f"R5-{args.candidate}.patch"
    write_patch(TARGET, before, after, patch_path)

    report = {
        "project": "LEXIGEN OMEGA",
        "stage": "R5_candidate_materialization_before_execution",
        "instance_id": "python-pillow__Pillow-d8af3fc",
        "candidate": args.candidate,
        "target": TARGET.as_posix(),
        "base_git_blob_sha1": base_blob,
        "proposal_git_blob_sha1": PROPOSAL_BLOB,
        "before_sha256": sha256_bytes(before.encode()),
        "after_sha256": sha256_bytes(after.encode()),
        "patch_sha256": sha256_bytes(patch_path.read_bytes()),
        "candidate_execution_count": 0,
        "candidate_timing_observed": False,
        "expert_opt_commit_accessed": False,
        "expert_diff_accessed": False,
        "hints_accessed": False,
        "R5_test_or_outcome_accessed": False,
    }
    (args.output / f"R5-{args.candidate}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
