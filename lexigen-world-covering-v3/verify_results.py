from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

EXPECTED_MD5 = "b2c626b07f216aac830d344eff5ad523"


def verify_record(result: dict[str, object]) -> dict[str, object]:
    target = result["target"]
    v = int(target["v"])
    k = int(target["k"])
    t = int(target["t"])
    upper = int(target["upper"])
    raw = result.get("blocks_zero_based") or []
    blocks = [tuple(int(x) for x in block) for block in raw]
    problems = []
    if len(blocks) != len(set(blocks)):
        problems.append("duplicate blocks")
    for i, block in enumerate(blocks):
        if len(block) != k:
            problems.append(f"block {i} size")
        if tuple(sorted(block)) != block:
            problems.append(f"block {i} unsorted")
        if len(set(block)) != len(block):
            problems.append(f"block {i} repeats point")
        if any(x < 0 or x >= v for x in block):
            problems.append(f"block {i} out of range")
    covered = set()
    for block in blocks:
        covered.update(itertools.combinations(block, t))
    required = math.comb(v, t)
    if len(covered) != required:
        problems.append(f"covers {len(covered)} of {required}")
    if len(blocks) >= upper:
        problems.append(f"block count {len(blocks)} not below {upper}")
    return {
        "target": target["name"],
        "valid": not problems,
        "block_count": len(blocks),
        "prior_upper_bound": upper,
        "covered_t_subsets": len(covered),
        "required_t_subsets": required,
        "blocks_sha256": hashlib.sha256(
            json.dumps(raw, separators=(",", ":")).encode()
        ).hexdigest(),
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.summary.read_bytes()
    summary = json.loads(raw)
    problems = []
    if summary.get("snapshot_md5") != EXPECTED_MD5:
        problems.append("snapshot MD5 mismatch")
    results = summary.get("results")
    if not isinstance(results, list) or len(results) != 3:
        problems.append("summary must contain exactly three results")
        results = []
    candidates = [x for x in results if bool(x.get("record_candidate"))]
    rows = [verify_record(x) for x in candidates]
    if any(not row["valid"] for row in rows):
        problems.append("candidate failed independent verification")
    report = {
        "protocol": "LEXIGEN World Covering Record v3 independent verification",
        "summary_sha256": hashlib.sha256(raw).hexdigest(),
        "snapshot_md5": summary.get("snapshot_md5"),
        "selected_results": len(results),
        "claimed_record_candidates": len(candidates),
        "verified_record_candidates": sum(bool(row["valid"]) for row in rows),
        "candidate_verifications": rows,
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
