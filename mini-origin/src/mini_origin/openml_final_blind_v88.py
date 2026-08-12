from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import openml_blind_v78 as parent
from . import response_lattice_closure_v85 as repair

PREREGISTRATION = Path(__file__).resolve().parents[2] / "campaigns" / "v88-final-fresh-openml-gate.json"
V86_REGISTRY_DIGEST = "c986710aa38d89bb9bf00df9a5e5817a26c359915c0f028646208cd9bcfe8ec0"
V85_EVIDENCE_DIGEST = "ca99dd822bc57fca55ffaf6de3614c7403cdbc0c85f3db81e0652dfe0acc0c20"
V85_COMMIT = "912c3ebd933ae39eb05e10467f1ecad56e326b03"


def canonical_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_inputs(manifest_path: Path) -> tuple[dict[str, object], dict[str, object], str]:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8-sig"))
    if prereg["status"] != "preregistered_before_v88_openml_suite_access":
        raise RuntimeError("v0.88 preregistration changed")
    if any(int(value) != 0 for value in prereg["revision_budget_after_suite_access"].values()):
        raise RuntimeError("v0.88 zero revision budget changed")
    raw = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw.decode("utf-8-sig"))
    if manifest["status"] != "openml_fresh_dataset_hash_lock_v88_complete":
        raise RuntimeError("v0.88 hash lock incomplete")
    if manifest["parent_v85_commit"] != V85_COMMIT:
        raise RuntimeError("v0.85 commit anchor changed")
    if manifest["parent_v85_evidence_digest"] != V85_EVIDENCE_DIGEST:
        raise RuntimeError("v0.85 evidence anchor changed")
    if manifest["parent_v86_registry_digest"] != V86_REGISTRY_DIGEST:
        raise RuntimeError("v0.86 registry anchor changed")
    if int(manifest["dataset_count"]) != 7:
        raise RuntimeError("v0.88 must lock exactly seven datasets")
    if manifest["selected_id_overlap"] or manifest["selected_name_overlap"]:
        raise RuntimeError("v0.88 lock overlaps frozen repository registry")
    return prereg, manifest, manifest_sha


def install(manifest_path: Path) -> tuple[dict[str, object], dict[str, object], str]:
    prereg, manifest, manifest_sha = load_inputs(manifest_path)
    lock_digest = str(manifest["lock_digest"])
    parent.PREREGISTRATION = PREREGISTRATION
    parent.MANIFEST = manifest_path
    parent.V77_LOCK_DIGEST = lock_digest
    parent.V77_MANIFEST_SHA256 = manifest_sha
    parent.V76_REGISTRY_DIGEST = V86_REGISTRY_DIGEST
    parent.V75_EVIDENCE_DIGEST = V85_EVIDENCE_DIGEST
    parent.FROZEN_V75_COMMIT = V85_COMMIT
    parent.FROZEN_V77_COMMIT = "v88-manifest-artifact"
    parent.load_frozen_inputs = lambda: (prereg, manifest)
    parent.frontier = repair
    return prereg, manifest, manifest_sha


def run_reference(manifest_path: Path, states_path: Path, reference_path: Path):
    prereg, manifest, manifest_sha = install(manifest_path)
    result = parent.run_reference(states_path, reference_path)
    result["status"] = "openml_fresh_dataset_python_reference_v88"
    result["selector_revision"] = "response-lattice-closure-v85"
    result["v88_lock_digest"] = manifest["lock_digest"]
    result["v88_manifest_sha256"] = manifest_sha
    result["parent_v85_evidence_digest"] = V85_EVIDENCE_DIGEST
    result["selected_openml_dataset_ids"] = [row["dataset_id"] for row in manifest["datasets"]]
    result["claim_scope"] = prereg["claim_boundary"]
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def validate(manifest_path: Path, reference_path: Path, rust_path: Path, output_path: Path):
    prereg, manifest, manifest_sha = install(manifest_path)
    result = parent.validate(reference_path, rust_path, output_path)
    summaries = {str(row["task"]): row for row in result["dataset_summaries"]}
    selected_names = [str(row["name"]) for row in manifest["datasets"]]
    per_dataset_minimum = int(prereg["blind_gate"]["minimum_states_from_each_dataset"])
    all_seven_covered = len(selected_names) == 7 and all(
        name in summaries and int(summaries[name]["selected_states"]) >= per_dataset_minimum
        for name in selected_names
    )
    parent_gate = bool(result["external_gate"])
    gate = parent_gate and all_seven_covered
    result["status"] = "openml_fresh_dataset_blind_pass_v88" if gate else "openml_fresh_dataset_blind_rejected_v88"
    result["external_gate"] = gate
    result["parent_v82_gate_passed"] = parent_gate
    result["all_seven_fresh_datasets_minimum_states_passed"] = all_seven_covered
    result["selector_revision"] = "response-lattice-closure-v85"
    result["v88_lock_digest"] = manifest["lock_digest"]
    result["v88_manifest_sha256"] = manifest_sha
    result["parent_v85_evidence_digest"] = V85_EVIDENCE_DIGEST
    result["selected_openml_dataset_ids"] = [row["dataset_id"] for row in manifest["datasets"]]
    result["claim_scope"] = prereg["claim_boundary"]
    result["evidence_digest"] = canonical_digest({k: v for k, v in result.items() if k != "evidence_digest"})
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ref = sub.add_parser("reference")
    ref.add_argument("--manifest", type=Path, required=True)
    ref.add_argument("--states", type=Path, required=True)
    ref.add_argument("--reference", type=Path, required=True)
    val = sub.add_parser("validate")
    val.add_argument("--manifest", type=Path, required=True)
    val.add_argument("--reference", type=Path, required=True)
    val.add_argument("--rust", type=Path, required=True)
    val.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reference":
        result = run_reference(args.manifest, args.states, args.reference)
        print(json.dumps({"status": result["status"], "datasets": result["contributing_dataset_count"], "base_states": result["base_state_count"], "profiled_states": result["profiled_state_count"]}, indent=2))
        return
    result = validate(args.manifest, args.reference, args.rust, args.output)
    print(json.dumps({"status": result["status"], "gate": result["external_gate"], "datasets": result["contributing_dataset_count"], "base_states": result["base_state_count"], "median_ratio": result["expansion_ratio_median"], "rust_mismatches": result["rust_mismatch_count"]}, indent=2))
    if not result["external_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
