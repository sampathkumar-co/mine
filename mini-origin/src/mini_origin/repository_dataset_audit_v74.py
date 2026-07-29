from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import repository_dataset_audit_v63 as base


REQUIRED_V64_IDS = (26, 50, 59, 72, 81, 107, 149)
REQUIRED_V68_IDS = (52, 74, 94, 151, 166, 181, 295)
V68_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v68-clean-lower-bound-rejected.json"
)
V68_DIGEST = "b2ff35cbc40d0c2828fa26a3057c245d5c794f4ea9164b3f560c7bcfba50448b"
V71_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v71-label-free-selector-pass.json"
)
V71_DIGEST = "58382c7aa2bcd7c28fec54642fcf76cc88e5059e0eb62e942d6cccf33d6ddfe2"


def audit() -> dict[str, object]:
    v68 = json.loads(V68_EVIDENCE.read_text(encoding="utf-8"))
    v71 = json.loads(V71_EVIDENCE.read_text(encoding="utf-8"))
    if v68.get("development_gate"):
        raise RuntimeError("v0.68 must remain a rejected external gate")
    if v68.get("evidence_digest") != V68_DIGEST:
        raise RuntimeError("unexpected v0.68 evidence digest")
    if not v71.get("development_gate") or v71.get("evidence_digest") != V71_DIGEST:
        raise RuntimeError("unexpected v0.71 evidence")

    report = base.audit()
    discovered = {int(row["uci_id"]) for row in report["uci_datasets"]}
    required_checks = {
        str(uci_id): uci_id in discovered
        for uci_id in (*REQUIRED_V64_IDS, *REQUIRED_V68_IDS)
    }
    report["required_opened_uci_checks"] = required_checks
    report["required_v64_uci_ids"] = list(REQUIRED_V64_IDS)
    report["required_v68_uci_ids"] = list(REQUIRED_V68_IDS)
    report["parent_v68_evidence_digest"] = V68_DIGEST
    report["parent_v71_evidence_digest"] = V71_DIGEST
    report["status"] = (
        "repository_dataset_registry_v74_complete"
        if report["status"] == "repository_dataset_registry_complete"
        and all(required_checks.values())
        and not report.get("unreadable_files")
        else "repository_dataset_registry_v74_incomplete"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "refs": report["ref_count"],
        "files": report["files_scanned"],
        "bytes": report["bytes_scanned"],
        "uci_ids": report["uci_id_count"],
        "pystreed_tokens": report["pystreed_token_count"],
        "opened_checks": report["required_opened_uci_checks"],
        "unreadable": len(report.get("unreadable_files", [])),
    }, indent=2))
    if report["status"] != "repository_dataset_registry_v74_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
