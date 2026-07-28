from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from common import (
    SEED_MATERIAL,
    SNAPSHOT_MD5,
    file_sha256,
    load_snapshot_json,
    select_targets,
)
from solver_engine import solve_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    reserved, targets = select_targets(load_snapshot_json())
    selection = {
        "snapshot_md5": SNAPSHOT_MD5,
        "seed_material_sha256": hashlib.sha256(SEED_MATERIAL.encode()).hexdigest(),
        "v1_reserved_targets": [asdict(target) for target in reserved],
        "v2_targets": [asdict(target) for target in targets],
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    print("FROZEN_SELECTED_TARGETS_V2")
    print(json.dumps(selection, indent=2), flush=True)

    results = []
    for target in targets:
        print(f"START {target.name} upper={target.upper} lower={target.lower}", flush=True)
        result = solve_target(target, args.output)
        results.append(result)
        print(
            f"FINISH {target.name} valid={result['valid']} blocks={result['result_blocks']} "
            f"record={result['record_candidate']}",
            flush=True,
        )

    root = Path(__file__).resolve().parent
    code_names = [
        "selector_solver.py",
        "common.py",
        "greedy.py",
        "local_search.py",
        "solver_engine.py",
        "PROTOCOL.md",
        "requirements.txt",
    ]
    summary = {
        "protocol": "LEXIGEN World Covering Record v2",
        "snapshot_md5": SNAPSHOT_MD5,
        "selected_count": len(targets),
        "record_candidates": sum(bool(result["record_candidate"]) for result in results),
        "results": results,
        "code_hashes": {name: file_sha256(root / name) for name in code_names},
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
