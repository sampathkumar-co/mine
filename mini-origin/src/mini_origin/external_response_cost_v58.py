from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from . import average_odt_frontier_v44 as frontier
from . import exact_quotient_certificate_v42 as corpus
from . import response_cost_export_v57 as export_v57
from . import response_cost_pareto_v56 as response
from . import state_policy_v34 as state


MANIFEST = Path(__file__).resolve().parents[2] / "external-data" / "uci-v58" / "manifest.json"
PROFILE_SEEDS = (5801, 5802, 5803)
MAX_RECORDS = 384
MAX_STATES_PER_TASK = 15
MIN_CANDIDATES = 8
MAX_CANDIDATES = 24
MIN_RAW_QUERIES = 12
MAX_PARTITION_REPRESENTATIVES = 18
MIN_REDUNDANCY = 4
BUDGET_LADDER = (10_000, 50_000, 250_000, 500_000)


PARSERS = {
    "Zoo": {"files": ("zoo.data",), "kind": "zoo"},
    "Lymphography": {"files": ("lymphography.data",), "kind": "label-first"},
    "Congressional Voting Records": {"files": ("house-votes-84.data",), "kind": "label-first"},
    "Mushroom": {"files": ("agaricus-lepiota.data",), "kind": "label-first"},
    "Molecular Biology (Promoter Gene Sequences)": {"files": ("promoters.data",), "kind": "promoter"},
    "Lung Cancer": {"files": ("lung-cancer.data",), "kind": "label-first"},
    "SPECTF Heart": {"files": ("SPECTF.train", "SPECTF.test"), "kind": "label-first"},
}


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.58-evaluation/1"})
    with urlopen(request, timeout=120) as response_handle:
        return response_handle.read()


def archive_member(archive: zipfile.ZipFile, expected: str) -> str:
    candidates = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == expected]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one {expected!r}, found {candidates!r}")
    return candidates[0]


def parse_csv_line(line: str) -> list[str]:
    return [value.strip() for value in line.strip().split(",")]


def parse_records(name: str, payload: bytes) -> list[tuple[tuple[str, ...], str]]:
    parser = PARSERS[name]
    records: list[tuple[tuple[str, ...], str]] = []
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        for expected in parser["files"]:
            member = archive_member(archive, expected)
            text = archive.read(member).decode("utf-8", errors="replace")
            for raw_line in text.splitlines():
                if not raw_line.strip():
                    continue
                values = parse_csv_line(raw_line)
                kind = parser["kind"]
                if kind == "zoo":
                    if len(values) < 3:
                        raise RuntimeError(f"bad Zoo row: {raw_line!r}")
                    features = tuple(values[1:-1])
                    label = values[-1]
                elif kind == "label-first":
                    if len(values) < 2:
                        raise RuntimeError(f"bad label-first row: {raw_line!r}")
                    label = values[0]
                    features = tuple(values[1:])
                elif kind == "promoter":
                    if len(values) < 3:
                        raise RuntimeError(f"bad promoter row: {raw_line!r}")
                    label = values[0]
                    sequence = "".join(values[2:]).replace(" ", "").lower()
                    if len(sequence) != 57:
                        raise RuntimeError(f"expected 57 promoter bases, got {len(sequence)}")
                    features = tuple(sequence)
                else:
                    raise AssertionError(kind)
                records.append((features, label))
    widths = {len(features) for features, _ in records}
    if len(widths) != 1:
        raise RuntimeError(f"inconsistent feature widths for {name}: {sorted(widths)}")
    return records


def deterministic_sample(
    name: str,
    records: list[tuple[tuple[str, ...], str]],
) -> list[tuple[tuple[str, ...], str]]:
    distinct = sorted(set(records))
    distinct.sort(key=lambda row: hashlib.sha256(
        json.dumps([name, row[0], row[1]], separators=(",", ":")).encode("utf-8")
    ).hexdigest())
    return distinct[:MAX_RECORDS]


def task_from_records(name: str, records: list[tuple[tuple[str, ...], str]]):
    sampled = deterministic_sample(name, records)
    if not sampled:
        raise RuntimeError(f"no records for {name}")
    width = len(sampled[0][0])
    rows = tuple(features for features, _ in sampled)
    labels = tuple(label for _, label in sampled)
    task = state.base.make_task(
        name,
        tuple(f"q{index}" for index in range(width)),
        rows,
        labels,
    )
    return task, {
        "raw_records": len(records),
        "distinct_records": len(set(records)),
        "sampled_records": len(sampled),
        "features": width,
        "labels": len(set(labels)),
    }


def deterministic_rank(task_name: str, allowed: int, remaining: int, representatives: int) -> tuple[int, int, int, str]:
    return (
        -allowed.bit_count(),
        -representatives,
        -remaining.bit_count(),
        hashlib.sha256(f"v58:{task_name}:{allowed}:{remaining}".encode("utf-8")).hexdigest(),
    )


def select_states(task: object) -> tuple[list[tuple[int, int, int]], dict[str, int]]:
    harvested = corpus.collect_policy_states(task)
    candidates: list[tuple[int, int, int]] = []
    for allowed, remaining in harvested:
        size = allowed.bit_count()
        raw = remaining.bit_count()
        representatives = frontier.quotient_representative_count(task, allowed, remaining)
        if (
            MIN_CANDIDATES <= size <= MAX_CANDIDATES
            and raw >= MIN_RAW_QUERIES
            and representatives <= MAX_PARTITION_REPRESENTATIVES
            and raw - representatives >= MIN_REDUNDANCY
        ):
            candidates.append((allowed, remaining, representatives))
    candidates.sort(key=lambda row: deterministic_rank(task.name, row[0], row[1], row[2]))
    return candidates[:MAX_STATES_PER_TASK], {
        "harvested_states": len(harvested),
        "eligible_states": len(candidates),
        "selected_states": min(len(candidates), MAX_STATES_PER_TASK),
    }


def compact_state(task: object, allowed: int, remaining: int, seed: int) -> dict[str, object]:
    row = export_v57.compact_state(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v58:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:external-response-cost-v58".encode("utf-8")
    ).hexdigest()
    return row


def run(state_output: Path, reference_output: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verification_rows = []
    tasks = []
    dataset_summaries = []
    for dataset in manifest["datasets"]:
        payload = download(str(dataset["url"]))
        actual_hash = hashlib.sha256(payload).hexdigest()
        actual_bytes = len(payload)
        matched = actual_hash == dataset["sha256"] and actual_bytes == dataset["bytes"]
        verification_rows.append({
            "name": dataset["name"],
            "expected_sha256": dataset["sha256"],
            "actual_sha256": actual_hash,
            "expected_bytes": dataset["bytes"],
            "actual_bytes": actual_bytes,
            "matched": matched,
        })
        if not matched:
            raise RuntimeError(f"archive mismatch for {dataset['name']}")
        records = parse_records(str(dataset["name"]), payload)
        task, summary = task_from_records(str(dataset["name"]), records)
        selected, selection = select_states(task)
        summary.update(selection)
        summary["task"] = task.name
        dataset_summaries.append(summary)
        tasks.append((task, selected))

    compact_rows: list[dict[str, object]] = []
    reference_rows = []
    base_digests: set[str] = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hashlib.sha256(
                f"v58:{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_digests.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = response.profile_for_task(task, seed)
                result = response.evaluate_state(task, profile, allowed, remaining)
                compact = compact_state(task, allowed, remaining, seed)
                compact_rows.append(compact)
                result.update({
                    "task": task.name,
                    "base_state_digest": base_digest,
                    "state_digest": compact["digest"],
                    "structural_partition_representatives": representatives,
                })
                reference_rows.append(result)

    compact_rows.sort(key=lambda row: str(row["digest"]))
    reference_rows.sort(key=lambda row: str(row["state_digest"]))
    export_v57.write_text(compact_rows, state_output)

    solved = [row for row in reference_rows if row["pareto_solved"]]
    both = [row for row in solved if row["plain_solved"]]
    pareto_only = [row for row in solved if not row["plain_solved"]]
    ratios = [float(row["expansion_ratio_lower_bound"]) for row in solved]
    ladder = {
        str(budget): {
            key: sum(int(row["budget_ladder"][str(budget)][key]) for row in reference_rows)
            for key in ("pareto_solved", "plain_solved")
        }
        for budget in BUDGET_LADDER
    }
    reference = {
        "status": "external_response_cost_python_reference_v58",
        "parent_v56_digest": "5d0c281c66200ae378af25e6e6214e6b8b49b9fd144d4060fe29afc0a795a1cc",
        "parent_v57_artifact_digest": "sha256:ba5b26e95a7a95081866f1de32d14d5b9218f33b1019abc3994eddb054231462",
        "archive_lock_digest": manifest["lock_digest"],
        "archive_verification": {
            "all_hashes_match": all(row["matched"] for row in verification_rows),
            "rows": verification_rows,
        },
        "protocol": {
            "max_records": MAX_RECORDS,
            "max_states_per_task": MAX_STATES_PER_TASK,
            "candidate_range": [MIN_CANDIDATES, MAX_CANDIDATES],
            "minimum_raw_queries": MIN_RAW_QUERIES,
            "maximum_partition_representatives": MAX_PARTITION_REPRESENTATIVES,
            "minimum_redundancy": MIN_REDUNDANCY,
            "profile_seeds": list(PROFILE_SEEDS),
            "budget": response.BUDGET,
            "budget_ladder": list(BUDGET_LADDER),
        },
        "dataset_summaries": dataset_summaries,
        "contributing_dataset_count": sum(int(row["selected_states"] > 0) for row in dataset_summaries),
        "base_state_count": len(base_digests),
        "profiled_state_count": len(reference_rows),
        "pareto_solved_count": len(solved),
        "both_solved_count": len(both),
        "pareto_only_count": len(pareto_only),
        "plan_mismatch_count": sum(int(not row["matched_if_both"]) for row in both),
        "dominated_queries_removed": sum(
            int(row["pareto_stats"]["dominated_queries_removed"]) for row in solved
        ),
        "root_incomparable_classes": sum(
            int(row["root_pareto_certificate"]["incomparable_pareto_classes"])
            for row in reference_rows
        ),
        "expansion_ratio_median": float(np.median(ratios)) if ratios else None,
        "expansion_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "budget_ladder_summary": ladder,
        "state_input_sha256": hashlib.sha256(state_output.read_bytes()).hexdigest(),
        "rows": reference_rows,
    }
    reference["frozen_external_digest"] = hashlib.sha256(json.dumps({
        "archive_lock_digest": reference["archive_lock_digest"],
        "protocol": reference["protocol"],
        "dataset_summaries": reference["dataset_summaries"],
        "state_input_sha256": reference["state_input_sha256"],
        "state_digests": [row["state_digest"] for row in reference_rows],
    }, sort_keys=True).encode("utf-8")).hexdigest()
    reference_output.parent.mkdir(parents=True, exist_ok=True)
    reference_output.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.states, args.reference)
    print(json.dumps({
        "status": report["status"],
        "datasets": report["contributing_dataset_count"],
        "base_states": report["base_state_count"],
        "profiled_states": report["profiled_state_count"],
        "pareto_solved": report["pareto_solved_count"],
        "plain_solved": report["both_solved_count"],
        "pareto_only": report["pareto_only_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
