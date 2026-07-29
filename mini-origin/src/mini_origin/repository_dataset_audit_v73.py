from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import repository_dataset_audit_v63 as base


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v73-frozen-dataset-audit.json"
)
V67_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v67-updated-dataset-registry.json"
)
V72_EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "research-evidence"
    / "mini-origin-v72-label-free-frontier-pass.json"
)
V67_REGISTRY_DIGEST = "b88fcb352c2f80af8bc89a3a7576b9cd384800b67d1b168534ad26df9985b6c1"
V72_EVIDENCE_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
FROZEN_V72_COMMIT = "dae02829efc4819935a4ec87c31ea5eee3305d83"
NEWLY_OPENED_V68_IDS = (52, 74, 94, 151, 166, 181, 295)


def audit() -> dict[str, object]:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if preregistration["status"] != "preregistered_before_registry_execution":
        raise RuntimeError("v0.73 preregistration status changed")
    if preregistration["frozen_v72_commit"] != FROZEN_V72_COMMIT:
        raise RuntimeError("frozen v0.72 commit changed")
    if preregistration["parent_v72_evidence_digest"] != V72_EVIDENCE_DIGEST:
        raise RuntimeError("v0.72 evidence commitment changed")
    for key in (
        "candidate_metadata_committed_before_registry",
        "external_archive_access_before_registry",
        "record_or_label_access_before_registry",
        "solver_execution_before_registry",
    ):
        if preregistration[key] is not False:
            raise RuntimeError(f"pre-registry boundary violated: {key}")

    parent = json.loads(V72_EVIDENCE.read_text(encoding="utf-8"))
    if not parent["development_gate"]:
        raise RuntimeError("v0.72 parent must remain a pass")
    if parent["evidence_digest"] != V72_EVIDENCE_DIGEST:
        raise RuntimeError("unexpected v0.72 evidence")

    previous = json.loads(V67_REGISTRY.read_text(encoding="utf-8"))
    if previous["status"] != "repository_dataset_registry_v67_complete":
        raise RuntimeError("v0.67 registry must remain complete")
    if previous["registry_digest"] != V67_REGISTRY_DIGEST:
        raise RuntimeError("unexpected v0.67 registry digest")

    report = base.audit()
    discovered = {int(row["uci_id"]) for row in report["uci_datasets"]}
    required = set(int(value) for value in previous["excluded_uci_ids"])
    required.update(NEWLY_OPENED_V68_IDS)
    checks = {str(uci_id): uci_id in discovered for uci_id in sorted(required)}

    report["parent_v72_evidence_digest"] = V72_EVIDENCE_DIGEST
    report["frozen_v72_commit"] = FROZEN_V72_COMMIT
    report["parent_v67_registry_digest"] = V67_REGISTRY_DIGEST
    report["required_preblind_uci_checks"] = checks
    report["excluded_uci_ids"] = sorted(discovered)
    report["excluded_uci_id_count"] = len(discovered)
    report["claim_scope"] = preregistration["claim_boundary"]
    report["status"] = (
        "repository_dataset_registry_v73_complete"
        if report["status"] == "repository_dataset_registry_complete"
        and all(checks.values())
        else "repository_dataset_registry_v73_incomplete"
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
        "excluded_uci_ids": report["excluded_uci_id_count"],
        "pystreed_tokens": report["pystreed_token_count"],
        "required_checks_passed": all(report["required_preblind_uci_checks"].values()),
        "registry_digest": report["registry_digest"],
    }, indent=2))
    if report["status"] != "repository_dataset_registry_v73_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
