from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import openml_hash_lock_v77 as parent

PREREGISTRATION = Path(__file__).resolve().parents[2] / "campaigns" / "v88-final-fresh-openml-gate.json"
V85_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v85-authoritative-opened-data-development-pass.json"
V87_REJECTION = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v87-ucr-lock-rejection.json"
V85_COMMIT = "912c3ebd933ae39eb05e10467f1ecad56e326b03"
V85_EVIDENCE_DIGEST = "ca99dd822bc57fca55ffaf6de3614c7403cdbc0c85f3db81e0652dfe0acc0c20"
V86_REGISTRY_DIGEST = "c986710aa38d89bb9bf00df9a5e5817a26c359915c0f028646208cd9bcfe8ec0"
V86_REGISTRY_SHA256 = "f1b50e11ce23e4bdfa42e3575c08379dc99d8bd98bc15fd7d581a464865effe0"
V87_ARTIFACT_SHA256 = "7b99df65a69e04ce7e4b4bacf7e87b73920ed11f4c610162335628c624bd29fc"


def load_inputs(registry_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if prereg["status"] != "preregistered_before_v88_openml_suite_access":
        raise RuntimeError("v0.88 preregistration status changed")
    if prereg["parent_v85_commit"] != V85_COMMIT:
        raise RuntimeError("v0.85 commit anchor changed")
    if prereg["parent_v85_evidence_digest"] != V85_EVIDENCE_DIGEST:
        raise RuntimeError("v0.85 evidence anchor changed")
    if prereg["parent_v86_registry_digest"] != V86_REGISTRY_DIGEST:
        raise RuntimeError("v0.86 registry anchor changed")
    if prereg["v87_rejection_artifact_sha256"] != V87_ARTIFACT_SHA256:
        raise RuntimeError("v0.87 rejection anchor changed")
    if any(bool(value) for value in prereg["preaccess_boundary"].values()):
        raise RuntimeError("v0.88 preaccess boundary changed")
    if any(int(value) != 0 for value in prereg["revision_budget_after_suite_access"].values()):
        raise RuntimeError("v0.88 revision budget changed")

    evidence = json.loads(V85_EVIDENCE.read_text(encoding="utf-8"))
    if evidence["evidence_digest"] != V85_EVIDENCE_DIGEST or evidence["verdict"] != "development_pass":
        raise RuntimeError("unexpected v0.85 evidence")
    rejection = json.loads(V87_REJECTION.read_text(encoding="utf-8"))
    if rejection["status"] != "ucr_untouched_lock_rejected_v87" or rejection["solver_executed"] is not False:
        raise RuntimeError("unexpected v0.87 rejection record")

    raw = registry_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != V86_REGISTRY_SHA256:
        raise RuntimeError("authoritative v0.86 registry bytes changed")
    registry = json.loads(raw.decode("utf-8-sig"))
    if registry["registry_digest"] != V86_REGISTRY_DIGEST:
        raise RuntimeError("authoritative v0.86 registry digest changed")
    return prereg, registry


def run(registry_path: Path, output: Path) -> dict[str, object]:
    prereg, registry = load_inputs(registry_path)
    parent.V76_REGISTRY_DIGEST = V86_REGISTRY_DIGEST
    parent.FROZEN_V75_COMMIT = V85_COMMIT
    parent.V75_EVIDENCE_DIGEST = V85_EVIDENCE_DIGEST
    parent.load_frozen_inputs = lambda: (prereg, registry)
    result = parent.run(output)
    result["status"] = "openml_fresh_dataset_hash_lock_v88_complete"
    result["parent_v85_commit"] = V85_COMMIT
    result["parent_v85_evidence_digest"] = V85_EVIDENCE_DIGEST
    result["parent_v86_registry_digest"] = V86_REGISTRY_DIGEST
    result["v87_rejection_artifact_sha256"] = V87_ARTIFACT_SHA256
    result["claim_scope"] = prereg["claim_boundary"]
    result["lock_digest"] = parent.canonical_digest({k: v for k, v in result.items() if k != "lock_digest"})
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.registry, args.output)
    print(json.dumps({
        "status": result["status"],
        "datasets": result["dataset_count"],
        "selected_ids": [row["dataset_id"] for row in result["datasets"]],
        "lock_digest": result["lock_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
