from __future__ import annotations

import json
import traceback
from pathlib import Path

import select_holdouts

OUTPUT = Path("selection-diagnostic-evidence")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        select_holdouts.main()
        report = {
            "status": "selection_completed",
            "locked_selector_changed": False,
            "task_contents_opened": False,
            "data_manifests_opened": False,
            "data_payloads_opened": False,
        }
        (OUTPUT / "diagnostic-success.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        report = {
            "status": "selection_infrastructure_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "locked_selector_changed": False,
            "scientific_rules_changed": False,
            "task_contents_opened": False,
            "data_manifests_opened": False,
            "data_payloads_opened": False,
            "reports_opened": False,
            "public_solvers_opened": False
        }
        (OUTPUT / "diagnostic-failure.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise


if __name__ == "__main__":
    main()
