from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from . import conditioned_cell_frontier_v60 as conditioned
from . import external_response_cost_v58 as external
from . import response_cost_export_v57 as export_v57
from . import response_cost_pareto_v56 as response


MANIFEST = Path(__file__).resolve().parents[2] / "external-data" / "uci-v64" / "manifest.json"
PROFILE_SEEDS = conditioned.PROFILE_SEEDS
BUDGET_LADDER = conditioned.BUDGET_LADDER
LOCK_DIGEST = "5f48634b5e4f020b7dec6ec23b98141b4221f4495e3c1cfc8eb8ff44ba51609b"
REGISTRY_DIGEST = "870daa9b28450c1717a266ca73b7717ba24abba11b7f4c537e28e77ab1c0cc0d"
PARENT_V60_DIGEST = "fe0c56d83095bfc6607f9bfffb480ca829f369d6bd9db5ff47860184f7e246fb"


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.64-evaluation/1"})
    with urlopen(request, timeout=300) as response_handle:
        return response_handle.read()


def archive_member(archive: zipfile.ZipFile, expected: str) -> str:
    matches = [
        name for name in archive.namelist()
        if name.rsplit("/", 1)[-1].lower() == expected.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {expected!r}; matches={matches!r}; "
            f"members={archive.namelist()!r}"
        )
    return matches[0]


def decompress_unix_compress(payload: bytes) -> bytes:
    errors: list[str] = []
    for command in (("gzip", "-dc"), ("uncompress", "-c")):
        try:
            completed = subprocess.run(
                command,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            if completed.stdout:
                return completed.stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            errors.append(f"{command!r}: {exc}")
    raise RuntimeError(f"unable to decompress Unix .Z payload: {errors}")


def nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def delimited_values(line: str) -> list[str]:
    if "," in line:
        return [value.strip() for value in line.split(",")]
    return line.split()


def label_first(values: list[str]) -> tuple[tuple[str, ...], str]:
    if len(values) < 2:
        raise RuntimeError(f"bad label-first row: {values!r}")
    return tuple(values[1:]), values[0]


def label_last(values: list[str]) -> tuple[tuple[str, ...], str]:
    if len(values) < 2:
        raise RuntimeError(f"bad label-last row: {values!r}")
    return tuple(values[:-1]), values[-1]


def parse_connect4(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    compressed = archive.read(archive_member(archive, "connect-4.data.Z"))
    text = decompress_unix_compress(compressed).decode("utf-8", errors="strict")
    records = [label_last(delimited_values(line)) for line in nonempty_lines(text)]
    if len(records) != 67_557 or {len(row[0]) for row in records} != {42}:
        raise RuntimeError(f"unexpected Connect-4 shape: rows={len(records)} widths={sorted({len(row[0]) for row in records})}")
    return records


def parse_image_segmentation(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    records: list[tuple[tuple[str, ...], str]] = []
    for expected in ("segmentation.data", "segmentation.test"):
        text = archive.read(archive_member(archive, expected)).decode("utf-8", errors="strict")
        for line in nonempty_lines(text):
            if line.startswith(";"):
                continue
            values = [value.strip() for value in line.split(",")]
            if values[0] == "REGION-CENTROID-COL":
                continue
            records.append(label_first(values))
    if len(records) != 2_310 or {len(row[0]) for row in records} != {19}:
        raise RuntimeError(f"unexpected Image Segmentation shape: rows={len(records)} widths={sorted({len(row[0]) for row in records})}")
    return records


def parse_letter(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    text = archive.read(archive_member(archive, "letter-recognition.data")).decode("utf-8", errors="strict")
    records = [label_first(delimited_values(line)) for line in nonempty_lines(text)]
    if len(records) != 20_000 or {len(row[0]) for row in records} != {16}:
        raise RuntimeError(f"unexpected Letter Recognition shape: rows={len(records)} widths={sorted({len(row[0]) for row in records})}")
    return records


def parse_multiple_features(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    feature_members = (
        "mfeat-fac",
        "mfeat-fou",
        "mfeat-kar",
        "mfeat-mor",
        "mfeat-pix",
        "mfeat-zer",
    )
    views: list[list[tuple[str, ...]]] = []
    for expected in feature_members:
        text = archive.read(archive_member(archive, expected)).decode("utf-8", errors="strict")
        rows = [tuple(line.split()) for line in nonempty_lines(text)]
        if len(rows) != 2_000 or len({len(row) for row in rows}) != 1:
            raise RuntimeError(
                f"unexpected Multiple Features view {expected}: "
                f"rows={len(rows)} widths={sorted({len(row) for row in rows})}"
            )
        views.append(rows)
    records = []
    for index in range(2_000):
        features = tuple(value for view in views for value in view[index])
        label = str(index // 200)
        records.append((features, label))
    if {len(row[0]) for row in records} != {649}:
        raise RuntimeError("unexpected combined Multiple Features width")
    return records


def parse_pendigits(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    records: list[tuple[tuple[str, ...], str]] = []
    for expected in ("pendigits.tra", "pendigits.tes"):
        text = archive.read(archive_member(archive, expected)).decode("utf-8", errors="strict")
        records.extend(label_last(delimited_values(line)) for line in nonempty_lines(text))
    if len(records) != 10_992 or {len(row[0]) for row in records} != {16}:
        raise RuntimeError(f"unexpected Pen Digits shape: rows={len(records)} widths={sorted({len(row[0]) for row in records})}")
    return records


def parse_waveform(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    compressed = archive.read(archive_member(archive, "waveform.data.Z"))
    text = decompress_unix_compress(compressed).decode("utf-8", errors="strict")
    records = [label_last(delimited_values(line)) for line in nonempty_lines(text)]
    if len(records) != 5_000 or {len(row[0]) for row in records} != {21}:
        raise RuntimeError(f"unexpected Waveform shape: rows={len(records)} widths={sorted({len(row[0]) for row in records})}")
    return records


def parse_vehicle(archive: zipfile.ZipFile) -> list[tuple[tuple[str, ...], str]]:
    members = sorted(
        name for name in archive.namelist()
        if re.fullmatch(r"(?:.*/)?xa[a-i]\.dat", name, flags=re.IGNORECASE)
    )
    if len(members) != 9:
        raise RuntimeError(f"expected nine Vehicle data members, found {members!r}")
    records: list[tuple[tuple[str, ...], str]] = []
    for member in members:
        text = archive.read(member).decode("utf-8", errors="strict")
        records.extend(label_last(line.split()) for line in nonempty_lines(text))
    if len(records) != 846 or {len(row[0]) for row in records} != {18}:
        raise RuntimeError(f"unexpected Vehicle shape: rows={len(records)} widths={sorted({len(row[0]) for row in records})}")
    return records


PARSERS = {
    "Connect-4": parse_connect4,
    "Image Segmentation": parse_image_segmentation,
    "Letter Recognition": parse_letter,
    "Multiple Features": parse_multiple_features,
    "Pen-Based Recognition of Handwritten Digits": parse_pendigits,
    "Waveform Database Generator (Version 1)": parse_waveform,
    "Statlog (Vehicle Silhouettes)": parse_vehicle,
}


def parse_records(name: str, payload: bytes) -> list[tuple[tuple[str, ...], str]]:
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        records = PARSERS[name](archive)
    widths = {len(features) for features, _ in records}
    if not records or len(widths) != 1:
        raise RuntimeError(f"bad records for {name}: count={len(records)} widths={widths}")
    return records


def compact_state(task: object, allowed: int, remaining: int, seed: int) -> dict[str, object]:
    row = export_v57.compact_state(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v64:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:clean-external-conditioned-v64".encode("utf-8")
    ).hexdigest()
    return row


def protocol() -> dict[str, object]:
    return {
        "profile_seeds": list(conditioned.PROFILE_SEEDS),
        "path_seeds": list(conditioned.PATH_SEEDS),
        "maximum_depth": conditioned.MAX_DEPTH,
        "maximum_query_choices": conditioned.MAX_QUERY_CHOICES,
        "maximum_cells_per_depth": conditioned.MAX_CELLS_PER_DEPTH,
        "sample_sizes": list(conditioned.SAMPLE_SIZES),
        "max_states_per_task": conditioned.MAX_STATES_PER_TASK,
        "partition_class_range": [
            conditioned.MIN_PARTITION_CLASSES,
            conditioned.MAX_PARTITION_CLASSES,
        ],
        "raw_query_range": [
            conditioned.MIN_RAW_QUERIES,
            conditioned.MAX_RAW_QUERIES,
        ],
        "minimum_redundancy": conditioned.MIN_REDUNDANCY,
        "budget": response.BUDGET,
        "budget_ladder": list(BUDGET_LADDER),
    }


def run(states_path: Path, reference_path: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["lock_digest"] != LOCK_DIGEST:
        raise RuntimeError("unexpected archive-lock digest")
    if manifest["repository_registry_digest"] != REGISTRY_DIGEST:
        raise RuntimeError("unexpected repository-registry digest")

    tasks = []
    summaries = []
    verification = []
    for dataset in manifest["datasets"]:
        payload = download(str(dataset["url"]))
        actual_hash = hashlib.sha256(payload).hexdigest()
        actual_bytes = len(payload)
        matched = actual_hash == dataset["sha256"] and actual_bytes == int(dataset["bytes"])
        verification.append({
            "name": dataset["name"],
            "uci_id": dataset["uci_id"],
            "expected_sha256": dataset["sha256"],
            "actual_sha256": actual_hash,
            "expected_bytes": dataset["bytes"],
            "actual_bytes": actual_bytes,
            "matched": matched,
        })
        if not matched:
            raise RuntimeError(f"archive mismatch for {dataset['name']}")
        records = parse_records(str(dataset["name"]), payload)
        task, summary = external.task_from_records(str(dataset["name"]), records)
        selected, selection = conditioned.select_states(task)
        summary.update(selection)
        summary["task"] = task.name
        summary["uci_id"] = dataset["uci_id"]
        summaries.append(summary)
        tasks.append((task, selected))

    state_rows = []
    reference_rows = []
    base_states: set[str] = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hashlib.sha256(
                f"v64:{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_states.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile_row = response.profile_for_task(task, seed)
                result = response.evaluate_state(task, profile_row, allowed, remaining)
                compact = compact_state(task, allowed, remaining, seed)
                state_rows.append(compact)
                result.update({
                    "task": task.name,
                    "base_state_digest": base_digest,
                    "state_digest": compact["digest"],
                    "structural_partition_representatives": representatives,
                })
                reference_rows.append(result)

    state_rows.sort(key=lambda row: str(row["digest"]))
    reference_rows.sort(key=lambda row: str(row["state_digest"]))
    export_v57.write_text(state_rows, states_path)

    solved = [row for row in reference_rows if row["pareto_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    pareto_only = [row for row in solved if not row["plain_solved"]]
    ratios = [float(row["expansion_ratio_lower_bound"]) for row in solved]
    ladder = {
        str(budget): {
            key: sum(
                int(row["budget_ladder"][str(budget)][key])
                for row in reference_rows
            )
            for key in ("pareto_solved", "plain_solved")
        }
        for budget in BUDGET_LADDER
    }
    result = {
        "status": "clean_external_conditioned_python_reference_v64",
        "parent_v60_digest": PARENT_V60_DIGEST,
        "repository_registry_digest": REGISTRY_DIGEST,
        "archive_lock_digest": LOCK_DIGEST,
        "archive_verification": {
            "all_hashes_match": all(row["matched"] for row in verification),
            "rows": verification,
        },
        "protocol": protocol(),
        "dataset_summaries": summaries,
        "contributing_dataset_count": sum(int(row["selected_states"] > 0) for row in summaries),
        "base_state_count": len(base_states),
        "profiled_state_count": len(reference_rows),
        "pareto_solved_count": len(solved),
        "both_solved_count": len(both),
        "pareto_only_count": len(pareto_only),
        "plan_mismatch_count": sum(int(not row["matched_if_both"]) for row in both),
        "dominated_queries_removed": sum(
            int(row["pareto_stats"]["dominated_queries_removed"])
            for row in solved
        ),
        "root_incomparable_classes": sum(
            int(row["root_pareto_certificate"]["incomparable_pareto_classes"])
            for row in reference_rows
        ),
        "expansion_ratio_median": float(np.median(ratios)) if ratios else None,
        "expansion_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "budget_ladder_summary": ladder,
        "state_input_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "rows": reference_rows,
    }
    result["frozen_external_digest"] = hashlib.sha256(json.dumps({
        "archive_lock_digest": result["archive_lock_digest"],
        "repository_registry_digest": result["repository_registry_digest"],
        "parent_v60_digest": result["parent_v60_digest"],
        "protocol": result["protocol"],
        "dataset_summaries": summaries,
        "state_input_sha256": result["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in reference_rows],
    }, sort_keys=True).encode("utf-8")).hexdigest()
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.states, args.reference)
    print(json.dumps({
        "status": result["status"],
        "datasets": result["contributing_dataset_count"],
        "base_states": result["base_state_count"],
        "profiled_states": result["profiled_state_count"],
        "pareto_solved": result["pareto_solved_count"],
        "plain_solved": result["both_solved_count"],
        "pareto_only": result["pareto_only_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
