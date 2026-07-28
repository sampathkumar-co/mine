from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import aggregate_blind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    args.diagnostic.mkdir(parents=True, exist_ok=True)
    try:
        sys.argv = ["aggregate_blind.py", "--input", str(args.input), "--output", str(args.output)]
        aggregate_blind.main()
        report = {
            "status": "aggregation_succeeded",
            "original_blind_run_id": 30376286181,
            "candidate_executions": 0,
            "reference_executions": 0,
            "blind_records_rerun": 0
        }
        (args.diagnostic / "aggregation-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        report = {
            "status": "aggregation_failed",
            "original_blind_run_id": 30376286181,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "candidate_executions": 0,
            "reference_executions": 0,
            "blind_records_rerun": 0
        }
        (args.diagnostic / "aggregation-diagnostic.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise


if __name__ == "__main__":
    main()
