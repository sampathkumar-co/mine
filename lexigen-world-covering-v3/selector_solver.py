from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from common import (
    SNAPSHOT_MD5,
    V1_SEED,
    V2_SEED,
    V3_SEED,
    file_sha256,
    load_snapshot_json,
    select_target_lineage,
)
from solver_engine import solve_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    v1, v2, v3 = select_target_lineage(load_snapshot_json())
    selection = {
        "snapshot_md5": SNAPSHOT_MD5,
        "seed_hashes": {
            "v1": hashlib.sha256(V1_SEED.encode()).hexdigest(),
            "v2": hashlib.sha256(V2_SEED.encode()).hexdigest(),
            "v3": hashlib.sha256(V3_SEED.encode()).hexdigest(),
        },
        "v1_reserved": [asdict(x) for x in v1],
        "v2_reserved": [asdict(x) for x in v2],
        "v3_targets": [asdict(x) for x in v3],
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("FROZEN_SELECTED_TARGETS_V3")
    print(json.dumps(selection, indent=2), flush=True)

    results = []
    for target in v3:
        print(f"START {target.name} upper={target.upper} lower={target.lower}", flush=True)
        result = solve_target(target, args.output)
        results.append(result)
        print(
            f"FINISH {target.name} valid={result['valid']} blocks={result['result_blocks']} "
            f"record={result['record_candidate']}",
            flush=True,
        )

    root = Path(__file__).resolve().parent
    names = [
        "selector_solver.py",
        "common.py",
        "greedy.py",
        "repair.py",
        "solver_engine.py",
        "verify_results.py",
        "PROTOCOL.md",
        "requirements.txt",
    ]
    summary = {
        "protocol": "LEXIGEN World Covering Record v3",
        "snapshot_md5": SNAPSHOT_MD5,
        "selected_count": len(v3),
        "record_candidates": sum(bool(x["record_candidate"]) for x in results),
        "results": results,
        "code_hashes": {name: file_sha256(root / name) for name in names},
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
