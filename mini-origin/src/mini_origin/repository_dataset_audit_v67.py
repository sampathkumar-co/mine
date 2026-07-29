from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import repository_dataset_audit_v63 as base


REQUIRED_V64_IDS = (26, 50, 59, 72, 81, 107, 149)
V66_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v66-rust-lower-bound-pass.json"
V66_DIGEST = "3b2bb026556ff9f6321ad6a8375854ae46931e64080329c76f86f31d12c0d643"


def audit() -> dict[str, object]:
    evidence = json.loads(V66_EVIDENCE.read_text(encoding="utf-8"))
    if not evidence["development_gate"] or evidence["evidence_digest"] != V66_DIGEST:
        raise RuntimeError("unexpected v0.66 evidence")
    report = base.audit()
    discovered = {int(row["uci_id"]) for row in report["uci_datasets"]}
    v64_checks = {
        str(uci_id): uci_id in discovered for uci_id in REQUIRED_V64_IDS
    }
    report["required_v64_uci_checks"] = v64_checks
    report["parent_v66_evidence_digest"] = V66_DIGEST
    report["status"] = (
        "repository_dataset_registry_v67_complete"
        if report["status"] == "repository_dataset_registry_complete"
        and all(v64_checks.values())
        else "repository_dataset_registry_v67_incomplete"
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
        "v64_checks": report["required_v64_uci_checks"],
    }, indent=2))
    if report["status"] != "repository_dataset_registry_v67_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
