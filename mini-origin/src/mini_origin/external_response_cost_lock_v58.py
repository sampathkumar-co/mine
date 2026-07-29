from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DATASETS = (
    {
        "name": "Zoo",
        "uci_id": 111,
        "url": "https://archive.ics.uci.edu/static/public/111/zoo.zip",
        "doi": "10.24432/C5R59V",
    },
    {
        "name": "Lymphography",
        "uci_id": 63,
        "url": "https://archive.ics.uci.edu/static/public/63/lymphography.zip",
        "doi": "10.24432/C54598",
    },
    {
        "name": "Congressional Voting Records",
        "uci_id": 105,
        "url": "https://archive.ics.uci.edu/static/public/105/congressional+voting+records.zip",
        "doi": "10.24432/C5C01P",
    },
    {
        "name": "Mushroom",
        "uci_id": 73,
        "url": "https://archive.ics.uci.edu/static/public/73/mushroom.zip",
        "doi": "10.24432/C5959T",
    },
    {
        "name": "Molecular Biology (Promoter Gene Sequences)",
        "uci_id": 67,
        "url": "https://archive.ics.uci.edu/static/public/67/molecular+biology+promoter+gene+sequences.zip",
        "doi": "10.24432/C5S01D",
    },
    {
        "name": "Lung Cancer",
        "uci_id": 62,
        "url": "https://archive.ics.uci.edu/static/public/62/lung+cancer.zip",
        "doi": "10.24432/C57596",
    },
    {
        "name": "SPECTF Heart",
        "uci_id": 96,
        "url": "https://archive.ics.uci.edu/static/public/96/spectf+heart.zip",
        "doi": "10.24432/C5N015",
    },
)


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.58-hash-lock/1"})
    with urlopen(request, timeout=90) as response:
        return response.read()


def run(output: Path) -> dict[str, object]:
    rows = []
    for dataset in DATASETS:
        payload = download(str(dataset["url"]))
        rows.append({
            **dataset,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        })
    frozen = {
        "status": "external_response_cost_archive_hash_lock_v1",
        "protocol": "download_archive_bytes_and_hash_only_no_archive_open_no_record_parse",
        "license": "CC BY 4.0",
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
