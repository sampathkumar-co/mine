from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

EXPECTED_SNAPSHOT_MD5 = "b2c626b07f216aac830d344eff5ad523"


def verify_record(result: dict[str, object]) -> dict[str, object]:
    target = result["target"]
    v = int(target["v"])
    k = int(target["k"])
    t = int(target["t"])
    prior_upper = int(target["upper"])
    raw_blocks = result.get("blocks_zero_based") or []
    blocks = [tuple(int(point) for point in block) for block in raw_blocks]
    problems: list[str] = []

    if len(blocks) != len(set(blocks)):
        problems.append("duplicate blocks")
    for index, block in enumerate(blocks):
        if len(block) != k:
            problems.append(f"block {index} has size {len(block)}, expected {k}")
        if tuple(sorted(block)) != block:
            problems.append(f"block {index} is not strictly sorted")
        if len(set(block)) != len(block):
            problems.append(f"block {index} repeats a point")
        if any(point < 0 or point >= v for point in block):
            problems.append(f"block {index} contains a point outside [0,{v})")

    covered: set[tuple[int, ...]] = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    required = math.comb(v, t)
    if len(covered) != required:
        problems.append(f"covers {len(covered)} of {required} required t-subsets")
    if len(blocks) >= prior_upper:
        problems.append(f"block count {len(blocks)} is not below prior upper bound {prior_upper}")

    serialized_hash = hashlib.sha256(
        json.dumps(raw_blocks, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "target": target["name"],
        "valid": not problems,
        "block_count": len(blocks),
        "prior_upper_bound": prior_upper,
        "covered_t_subsets": len(covered),
        "required_t_subsets": required,
        "blocks_sha256": serialized_hash,
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    problems: list[str] = []
    if summary.get("snapshot_md5") != EXPECTED_SNAPSHOT_MD5:
        problems.append("summary snapshot MD5 differs from frozen snapshot")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 3:
        problems.append("summary must contain exactly three selected-target results")
        results = []

    candidates = [result for result in results if bool(result.get("record_candidate"))]
    verifications = [verify_record(result) for result in candidates]
    if any(not row["valid"] for row in verifications):
        problems.append("one or more claimed record candidates failed independent verification")

    report = {
        "protocol": "LEXIGEN World Covering Record v2 independent verification",
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_md5": summary.get("snapshot_md5"),
        "selected_results": len(results),
        "claimed_record_candidates": len(candidates),
        "verified_record_candidates": sum(bool(row["valid"]) for row in verifications),
        "candidate_verifications": verifications,
        "problems": problems,
        "verification_passes": not problems,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
