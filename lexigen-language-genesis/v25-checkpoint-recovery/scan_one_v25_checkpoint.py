from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RECOVERY = HERE.parent / "v25-recovery"
for folder in (HERE, RECOVERY):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import scan_one_v25_recovery as frozen_scanner
from enumerator_v25_checkpoint import (
    ControlledCheckpointStop,
    enumerate_programs_checkpointed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--split",
        choices=("discovery", "validation"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stop-after-processed-candidates",
        type=int,
        default=None,
    )
    args = parser.parse_args()
    def adapter(
        examples: Any,
        *,
        maximum_depth: int,
        maximum_unique_per_type_per_depth: int,
        maximum_total_unique: int,
        maximum_raw_candidates: int,
    ) -> dict[str, Any]:
        return enumerate_programs_checkpointed(
            examples,
            maximum_depth=maximum_depth,
            maximum_unique_per_type_per_depth=(
                maximum_unique_per_type_per_depth
            ),
            maximum_total_unique=maximum_total_unique,
            maximum_raw_candidates=maximum_raw_candidates,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            checkpoint_interval_processed_candidates=25000,
            stop_after_processed_candidates=(
                args.stop_after_processed_candidates
            ),
        )

    original_enumerator = frozen_scanner.enumerate_programs
    original_argv = sys.argv[:]
    frozen_scanner.enumerate_programs = adapter
    sys.argv = [
        "scan_one_v25_recovery.py",
        "--arcgen-root",
        str(args.arcgen_root),
        "--task-id",
        args.task_id,
        "--split",
        args.split,
        "--output",
        str(args.output),
    ]
    try:
        frozen_scanner.main()
    except ControlledCheckpointStop as error:
        print(json.dumps({
            "task_id": args.task_id,
            "status": "checkpointed_stop",
            "checkpoint": str(args.checkpoint),
            "message": str(error),
        }, sort_keys=True))
        raise SystemExit(75)
    finally:
        frozen_scanner.enumerate_programs = original_enumerator
        sys.argv = original_argv


if __name__ == "__main__":
    main()
