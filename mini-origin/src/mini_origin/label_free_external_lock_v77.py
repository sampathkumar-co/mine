from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "research-evidence" / "mini-origin-v74-refreshed-dataset-registry.json"
V72_EVIDENCE = ROOT / "research-evidence" / "mini-origin-v72-label-free-frontier-pass.json"
REGISTRY_DIGEST = "2a089db9d801f30fd3072b0eea8be6ef7c34cf1bedc9144d6da39a22bb4ddd71"
V72_EVIDENCE_DIGEST = "b1fc70852a2ad35d91972889eb853856cde18bca0ed02db37cd37ac333639090"
EXPECTED_EXCLUDED_UCI_IDS = (
    3, 8, 12, 14, 19, 22, 26, 33, 44, 46, 50, 52, 59, 62, 63, 67,
    69, 70, 72, 73, 74, 76, 80, 81, 83, 90, 94, 95, 96, 100, 101, 105,
    107, 111, 146, 149, 151, 166, 171, 181, 295, 373, 468, 475, 519, 544,
    545, 571, 602, 697, 759, 850, 863, 864, 891,
)

DATASETS = (
    {
        "name": "Glass Identification",
        "uci_id": 42,
        "url": "https://archive.ics.uci.edu/static/public/42/glass%2Bidentification.zip",
        "doi": "10.24432/C5WW2P",
    },
    {
        "name": "Haberman's Survival",
        "uci_id": 43,
        "url": "https://archive.ics.uci.edu/static/public/43/haberman%2Bs%2Bsurvival.zip",
        "doi": "10.24432/C5XK51",
    },
    {
        "name": "Parkinsons",
        "uci_id": 174,
        "url": "https://archive.ics.uci.edu/static/public/174/parkinsons.zip",
        "doi": "10.24432/C59C74",
    },
    {
        "name": "Cardiotocography",
        "uci_id": 193,
        "url": "https://archive.ics.uci.edu/static/public/193/cardiotocography.zip",
        "doi": "10.24432/C51S4N",
    },
    {
        "name": "Vertebral Column",
        "uci_id": 212,
        "url": "https://archive.ics.uci.edu/static/public/212/vertebral%2Bcolumn.zip",
        "doi": "10.24432/C5K89B",
    },
    {
        "name": "Seeds",
        "uci_id": 236,
        "url": "https://archive.ics.uci.edu/static/public/236/seeds.zip",
        "doi": "10.24432/C5H30K",
    },
    {
        "name": "Banknote Authentication",
        "uci_id": 267,
        "url": "https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip",
        "doi": "10.24432/C55P57",
    },
)


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.77-hash-lock/1"})
    with urlopen(request, timeout=300) as response:
        return response.read()


def run(output: Path) -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("status") != "repository_dataset_registry_v74_complete":
        raise RuntimeError("v0.74 registry is not complete")
    if registry.get("registry_digest") != REGISTRY_DIGEST:
        raise RuntimeError("v0.74 registry digest changed")
    if int(registry.get("failure_count", -1)) != 0:
        raise RuntimeError("v0.74 registry contains scan failures")
    excluded_values = tuple(sorted(int(value) for value in registry["excluded_uci_ids"]))
    if excluded_values != EXPECTED_EXCLUDED_UCI_IDS:
        raise RuntimeError("v0.74 audited exclusion list changed")

    parent = json.loads(V72_EVIDENCE.read_text(encoding="utf-8"))
    if not parent.get("development_gate"):
        raise RuntimeError("v0.72 parent did not pass")
    if parent.get("evidence_digest") != V72_EVIDENCE_DIGEST:
        raise RuntimeError("v0.72 evidence digest changed")
    if int(parent.get("rust_mismatch_count", -1)) != 0:
        raise RuntimeError("v0.72 Rust reproduction is not exact")

    candidate_ids = tuple(int(row["uci_id"]) for row in DATASETS)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("duplicate candidate UCI IDs")
    overlap = sorted(set(candidate_ids) & set(excluded_values))
    if overlap:
        raise RuntimeError(f"candidate IDs occur in v0.74 registry: {overlap}")

    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        print(f"hashing {dataset['name']}: {dataset['url']}", flush=True)
        payload = download(str(dataset["url"]))
        rows.append({
            **dataset,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })

    result: dict[str, object] = {
        "status": "label_free_external_archive_hash_lock_v77",
        "protocol": "v74_all_branch_exclusion_then_hash_only_no_archive_open_no_member_listing_no_record_parse",
        "license": "CC BY 4.0",
        "parent_v72_evidence_digest": V72_EVIDENCE_DIGEST,
        "repository_registry_digest": REGISTRY_DIGEST,
        "excluded_uci_ids": list(excluded_values),
        "excluded_uci_id_count": len(excluded_values),
        "candidate_overlap": overlap,
        "dataset_count": len(rows),
        "datasets": rows,
    }
    result["lock_digest"] = hashlib.sha256(
        json.dumps(result, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
