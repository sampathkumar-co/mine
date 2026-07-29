from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


REGISTRY_DIGEST = "870daa9b28450c1717a266ca73b7717ba24abba11b7f4c537e28e77ab1c0cc0d"
EXCLUDED_UCI_IDS = {
    3, 8, 12, 14, 19, 22, 33, 44, 46, 62, 63, 67, 69, 70, 73, 76,
    80, 83, 90, 95, 96, 100, 101, 105, 111, 146, 171, 373, 468, 475,
    519, 544, 545, 571, 602, 697, 759, 850, 863, 864, 891,
}
DATASETS = (
    {
        "name": "Connect-4",
        "uci_id": 26,
        "url": "https://archive.ics.uci.edu/static/public/26/connect+4.zip",
        "doi": "10.24432/C59P43",
    },
    {
        "name": "Image Segmentation",
        "uci_id": 50,
        "url": "https://archive.ics.uci.edu/static/public/50/image+segmentation.zip",
        "doi": "10.24432/C5GP4N",
    },
    {
        "name": "Letter Recognition",
        "uci_id": 59,
        "url": "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip",
        "doi": "10.24432/C5ZP40",
    },
    {
        "name": "Multiple Features",
        "uci_id": 72,
        "url": "https://archive.ics.uci.edu/static/public/72/multiple+features.zip",
        "doi": "10.24432/C5HC70",
    },
    {
        "name": "Pen-Based Recognition of Handwritten Digits",
        "uci_id": 81,
        "url": "https://archive.ics.uci.edu/static/public/81/pen+based+recognition+of+handwritten+digits.zip",
        "doi": "10.24432/C5MG6K",
    },
    {
        "name": "Waveform Database Generator (Version 1)",
        "uci_id": 107,
        "url": "https://archive.ics.uci.edu/static/public/107/waveform+database+generator+version+1.zip",
        "doi": "10.24432/C5CS3C",
    },
    {
        "name": "Statlog (Vehicle Silhouettes)",
        "uci_id": 149,
        "url": "https://archive.ics.uci.edu/static/public/149/statlog+vehicle+silhouettes.zip",
        "doi": "10.24432/C5HG6N",
    },
)


def download(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "Mini-ORIGIN-v0.64-hash-lock/1"},
    )
    with urlopen(request, timeout=300) as response:
        return response.read()


def run(output: Path) -> dict[str, object]:
    candidate_ids = [int(row["uci_id"]) for row in DATASETS]
    overlap = sorted(set(candidate_ids) & EXCLUDED_UCI_IDS)
    if overlap:
        raise RuntimeError(f"candidate UCI IDs occur in repository registry: {overlap}")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("duplicate candidate UCI ID")

    rows = []
    for dataset in DATASETS:
        print(f"hashing {dataset['name']}: {dataset['url']}", flush=True)
        payload = download(str(dataset["url"]))
        rows.append({
            **dataset,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    frozen = {
        "status": "clean_external_archive_hash_lock_v1",
        "protocol": (
            "repository_registry_exclusion_then_download_archive_bytes_and_hash_only_"
            "no_archive_open_no_record_parse"
        ),
        "license": "CC BY 4.0",
        "parent_v60_commit": "f73ad897377ae1189844ea9ab49aa429c6b1c1c3",
        "repository_registry_digest": REGISTRY_DIGEST,
        "excluded_uci_id_count": len(EXCLUDED_UCI_IDS),
        "candidate_overlap": overlap,
        "dataset_count": len(rows),
        "datasets": rows,
    }
    frozen["lock_digest"] = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode("utf-8")
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
