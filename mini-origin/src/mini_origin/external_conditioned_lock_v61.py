from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


DATASETS = (
    {"name":"Soybean (Large)","uci_id":90,"url":"https://archive.ics.uci.edu/static/public/90/soybean+large.zip","doi":"10.24432/C5JG6Z"},
    {"name":"Annealing","uci_id":3,"url":"https://archive.ics.uci.edu/static/public/3/annealing.zip","doi":"10.24432/C5RW2F"},
    {"name":"Primary Tumor","uci_id":83,"url":"https://archive.ics.uci.edu/static/public/83/primary+tumor.zip","doi":"10.24432/C5WK5Q"},
    {"name":"Optical Recognition of Handwritten Digits","uci_id":80,"url":"https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip","doi":"10.24432/C50P49"},
    {"name":"Statlog (Landsat Satellite)","uci_id":146,"url":"https://archive.ics.uci.edu/static/public/146/statlog+landsat+satellite.zip","doi":"10.24432/C55887"},
    {"name":"Madelon","uci_id":171,"url":"https://archive.ics.uci.edu/static/public/171/madelon.zip","doi":"10.24432/C5602H"},
    {"name":"Glioma Grading Clinical and Mutation Features","uci_id":759,"url":"https://archive.ics.uci.edu/static/public/759/glioma+grading+clinical+and+mutation+features.zip","doi":"10.24432/C5R62J"},
)


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent":"Mini-ORIGIN-v0.61-hash-lock/1"})
    with urlopen(request, timeout=180) as response:
        return response.read()


def run(output: Path) -> dict[str, object]:
    rows=[]
    for dataset in DATASETS:
        payload=download(str(dataset["url"]))
        rows.append({**dataset,"sha256":hashlib.sha256(payload).hexdigest(),"bytes":len(payload)})
    frozen={"status":"external_conditioned_archive_hash_lock_v1","protocol":"download_archive_bytes_and_hash_only_no_archive_open_no_record_parse","license":"CC BY 4.0","dataset_count":len(rows),"datasets":rows}
    frozen["lock_digest"]=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode("utf-8")).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(frozen,indent=2),encoding="utf-8")
    return frozen


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.output),indent=2))


if __name__=="__main__":
    main()
