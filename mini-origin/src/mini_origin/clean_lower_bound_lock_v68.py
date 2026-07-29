from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REGISTRY = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v67-updated-dataset-registry.json"
REGISTRY_DIGEST = "b88fcb352c2f80af8bc89a3a7576b9cd384800b67d1b168534ad26df9985b6c1"
V66_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v66-rust-lower-bound-pass.json"
V66_DIGEST = "3b2bb026556ff9f6321ad6a8375854ae46931e64080329c76f86f31d12c0d643"

DATASETS = (
    {
        "name": "Ionosphere",
        "uci_id": 52,
        "url": "https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        "doi": "10.24432/C5W01B",
    },
    {
        "name": "Musk (Version 1)",
        "uci_id": 74,
        "url": "https://archive.ics.uci.edu/static/public/74/musk%2Bversion%2B1.zip",
        "doi": "10.24432/C5ZK5B",
    },
    {
        "name": "Spambase",
        "uci_id": 94,
        "url": "https://archive.ics.uci.edu/static/public/94/spambase.zip",
        "doi": "10.24432/C53G6X",
    },
    {
        "name": "Connectionist Bench (Sonar, Mines vs. Rocks)",
        "uci_id": 151,
        "url": "https://archive.ics.uci.edu/static/public/151/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks.zip",
        "doi": "10.24432/C5T01Q",
    },
    {
        "name": "Hill-Valley",
        "uci_id": 166,
        "url": "https://archive.ics.uci.edu/static/public/166/hill%2Bvalley.zip",
        "doi": "10.24432/C5JC8P",
    },
    {
        "name": "Libras Movement",
        "uci_id": 181,
        "url": "https://archive.ics.uci.edu/static/public/181/libras%2Bmovement.zip",
        "doi": "10.24432/C5GC82",
    },
    {
        "name": "Urban Land Cover",
        "uci_id": 295,
        "url": "https://archive.ics.uci.edu/static/public/295/urban%2Bland%2Bcover.zip",
        "doi": "10.24432/C53S48",
    },
)


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.68-hash-lock/1"})
    with urlopen(request, timeout=300) as response:
        return response.read()


def run(output: Path) -> dict[str, object]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["status"] != "repository_dataset_registry_v67_complete":
        raise RuntimeError("updated registry is not complete")
    if registry["registry_digest"] != REGISTRY_DIGEST:
        raise RuntimeError("updated registry digest changed")
    evidence = json.loads(V66_EVIDENCE.read_text(encoding="utf-8"))
    if not evidence["development_gate"] or evidence["evidence_digest"] != V66_DIGEST:
        raise RuntimeError("unexpected v0.66 evidence")

    excluded = {int(value) for value in registry["excluded_uci_ids"]}
    candidate_ids = [int(row["uci_id"]) for row in DATASETS]
    overlap = sorted(set(candidate_ids) & excluded)
    if overlap:
        raise RuntimeError(f"candidate IDs occur in updated registry: {overlap}")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("duplicate candidate UCI IDs")

    rows = []
    for dataset in DATASETS:
        print(f"hashing {dataset['name']}: {dataset['url']}", flush=True)
        payload = download(str(dataset["url"]))
        rows.append({
            **dataset,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    result = {
        "status": "clean_lower_bound_archive_hash_lock_v68",
        "protocol": "updated_repository_registry_exclusion_then_hash_only_no_archive_open_no_record_parse",
        "license": "CC BY 4.0",
        "parent_v66_evidence_digest": V66_DIGEST,
        "repository_registry_digest": REGISTRY_DIGEST,
        "excluded_uci_id_count": len(excluded),
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
