from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import exact_tail_v36 as v36


THRESHOLDS = (4, 6, 8, 10, 12, 16)
FEATURE_LIMITS = (4, 6, 8, 10, 12)


def run() -> dict[str, object]:
    v36.THRESHOLDS = THRESHOLDS
    v36.FEATURE_LIMITS = FEATURE_LIMITS
    report = v36.run()
    report["search_bounds"] = {
        "candidate_thresholds": list(THRESHOLDS),
        "feature_limits": list(FEATURE_LIMITS),
        "fallbacks": list(v36.v34.OBJECTIVE_NAMES),
    }
    report["bounded_protocol_digest"] = hashlib.sha256(
        json.dumps(
            {
                "search_bounds": report["search_bounds"],
                "selected_policy": report["selected_policy"],
                "planner": "exact_diagnosis_worst_mean_v1",
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "selected_policy": report["selected_policy"],
        "profile": report["profile"],
        "search_bounds": report["search_bounds"],
    }, indent=2))


if __name__ == "__main__":
    main()
