from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile


MANIFEST = Path(__file__).resolve().parents[2] / "external-data" / "uci-v64" / "manifest.json"
MAX_SAMPLE_BYTES = 256_000
MAX_SAMPLE_LINES = 8


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.64-schema/1"})
    with urlopen(request, timeout=300) as response:
        return response.read()


def text_sample(payload: bytes) -> dict[str, object]:
    prefix = payload[:MAX_SAMPLE_BYTES]
    text = prefix.decode("utf-8-sig", errors="replace")
    lines = [line.rstrip("\r\n") for line in text.splitlines() if line.strip()]
    sample = lines[:MAX_SAMPLE_LINES]
    return {
        "sample_lines": sample,
        "sample_line_count": len(sample),
        "comma_counts": [line.count(",") for line in sample],
        "space_token_counts": [len(line.split()) for line in sample],
        "tab_counts": [line.count("\t") for line in sample],
    }


def run(output: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["lock_digest"] != "5f48634b5e4f020b7dec6ec23b98141b4221f4495e3c1cfc8eb8ff44ba51609b":
        raise RuntimeError("unexpected v0.64 archive-lock digest")

    datasets = []
    for dataset in manifest["datasets"]:
        raw = download(str(dataset["url"]))
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != dataset["sha256"] or len(raw) != int(dataset["bytes"]):
            raise RuntimeError(f"archive mismatch for {dataset['name']}")

        members = []
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            for info in archive.infolist():
                row: dict[str, object] = {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "is_directory": info.is_dir(),
                }
                if not info.is_dir() and info.file_size <= 20_000_000:
                    member_payload = archive.read(info.filename)
                    row.update(text_sample(member_payload))
                members.append(row)
        datasets.append({
            "name": dataset["name"],
            "uci_id": dataset["uci_id"],
            "sha256": actual_hash,
            "bytes": len(raw),
            "members": members,
        })

    result = {
        "status": "clean_external_schema_inventory_v1",
        "claim_scope": "Hash-verified archive member and small text-sample inventory only. No state selection, cost generation, exact search or performance measurement.",
        "archive_lock_digest": manifest["lock_digest"],
        "dataset_count": len(datasets),
        "datasets": datasets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({
        "status": result["status"],
        "datasets": result["dataset_count"],
        "members": sum(len(row["members"]) for row in result["datasets"]),
    }, indent=2))


if __name__ == "__main__":
    main()
