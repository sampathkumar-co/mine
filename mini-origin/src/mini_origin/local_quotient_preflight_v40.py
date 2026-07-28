from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import local_quotient_runner_v40 as runner
from . import local_quotient_v40 as v40


PREFLIGHT_THRESHOLDS = (4, 6, 8, 10, 12, 16)


def run() -> dict[str, object]:
    original = v40.THRESHOLDS
    try:
        v40.THRESHOLDS = PREFLIGHT_THRESHOLDS
        report = runner.run()
    finally:
        v40.THRESHOLDS = original
    report["preflight_only"] = True
    report["preflight_thresholds"] = list(PREFLIGHT_THRESHOLDS)
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
        "selected_threshold": report["selected_threshold"],
        "profile": report["profile"],
        "preflight_only": True,
    }, indent=2))


if __name__ == "__main__":
    main()
