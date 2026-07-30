from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mini_origin import pmlb_blind_v82 as parent
from mini_origin import pmlb_development_v85 as development

SOURCE_RUN_ID = 30566963272
SOURCE_ARTIFACT_ID = 8770724605
SOURCE_ARTIFACT_DIGEST = "sha256:e5b2a769903b701184d59f12a9e2783e58aa5e0c818ea9bbcd95e2ff70b6f95a"
STATES_SHA256 = "3e784b85ff4cf38ec4908fa1da4c57b164ab62b6f3885e6e9c57881436f0c7ac"
REFERENCE_SHA256 = "03daea4bf64cf56543a5e81f0319d142781ee87843262e165d45105017120bd1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path("results/v85-salvage")
    states_path = root / "states.txt"
    reference_path = root / "python-reference.json"
    if sha256(states_path) != STATES_SHA256:
        raise RuntimeError("preserved v0.85 state bytes changed")
    if sha256(reference_path) != REFERENCE_SHA256:
        raise RuntimeError("preserved v0.85 reference bytes changed")

    preregistration, manifest, parent_evidence = parent.load_frozen_inputs()
    development._install()
    result = json.loads(reference_path.read_text(encoding="utf-8"))
    if result["status"] != "clean_lower_bound_python_reference_v68":
        raise RuntimeError("unexpected pre-crash reference status")
    if int(result["contributing_dataset_count"]) != 7:
        raise RuntimeError("preserved Python reference did not contribute seven datasets")
    if sha256(states_path) != str(result["state_input_sha256"]):
        raise RuntimeError("state input digest does not match preserved reference")

    result["status"] = "pmlb_opened_data_python_reference_v85_salvaged_metadata"
    result["v81_lock_digest"] = parent.V81_LOCK_DIGEST
    result["v81_manifest_sha256"] = parent.V81_MANIFEST_SHA256
    result["parent_v79_evidence_digest"] = parent.V79_EVIDENCE_DIGEST
    result["parent_v68_evidence_digest"] = parent.LEGACY_PARENT_V68_EVIDENCE_DIGEST
    result["compiler_protocol"] = parent.selector.frontier.compiler_protocol()
    result["selector_protocol"] = parent.selector.protocol()["state_selector"]
    result["selected_pmlb_datasets"] = [str(row["name"]) for row in manifest["datasets"]]
    result["frozen_v79_label_independence"] = {
        "mismatch_count": int(parent_evidence["label_independence_certificate"]["mismatch_count"]),
        "evidence_digest": parent.V79_EVIDENCE_DIGEST,
    }
    result["frozen_external_digest"] = parent.canonical_digest({
        "v81_lock_digest": parent.V81_LOCK_DIGEST,
        "parent_v79_evidence_digest": parent.V79_EVIDENCE_DIGEST,
        "protocol": result["protocol"],
        "dataset_summaries": result["dataset_summaries"],
        "state_input_sha256": result["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in result["rows"]],
    })
    result["development_data_status"] = "opened"
    result["fresh_external_evidence"] = False
    result["selector_revision"] = "response-lattice-closure-v85"
    result["implementation_amendment"] = development.IMPLEMENTATION_AMENDMENT.name
    result["claim_scope"] = preregistration["claim_boundary"]
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    provenance = {
        "status": "v85_failed_artifact_metadata_salvaged_for_diagnostic_replay",
        "authority": "non-authoritative diagnostic; replacement PR #208 workflow remains decisive",
        "source_run_id": SOURCE_RUN_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_states_sha256": STATES_SHA256,
        "source_reference_sha256": REFERENCE_SHA256,
        "salvaged_reference_sha256": sha256(reference_path),
        "scientific_components_changed": False,
    }
    provenance["evidence_digest"] = parent.canonical_digest(provenance)
    (root / "salvage-provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
