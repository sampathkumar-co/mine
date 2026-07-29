from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from . import conditioned_cell_frontier_v60 as conditioned
from . import external_response_cost_v58 as external
from . import response_cost_export_v57 as export_v57
from . import response_cost_lower_bound_v65 as bounded
from . import response_cost_pareto_v56 as response


MANIFEST = Path(__file__).resolve().parents[2] / "external-data" / "uci-v68" / "manifest.json"
LOCK_DIGEST = "9abc52a7e83255498c84b802d432306b5ff15dece032469968b8db3501d0a385"
REGISTRY_DIGEST = "b88fcb352c2f80af8bc89a3a7576b9cd384800b67d1b168534ad26df9985b6c1"
V66_DIGEST = "3b2bb026556ff9f6321ad6a8375854ae46931e64080329c76f86f31d12c0d643"
PROFILE_SEEDS = conditioned.PROFILE_SEEDS
BUDGET = response.BUDGET
BUDGET_LADDER = response.BUDGET_LADDER


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mini-ORIGIN-v0.68-evaluation/1"})
    with urlopen(request, timeout=300) as handle:
        return handle.read()


def member(archive: zipfile.ZipFile, basename: str) -> str:
    matches = [
        name for name in archive.namelist()
        if name.rsplit("/", 1)[-1].lower() == basename.lower()
        and not name.startswith("__MACOSX/")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {basename!r}, found {matches!r}")
    return matches[0]


def decompress_z(payload: bytes) -> bytes:
    errors = []
    for command in (("gzip", "-dc"), ("uncompress", "-c")):
        try:
            result = subprocess.run(
                command,
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            if result.stdout:
                return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            errors.append(str(error))
    raise RuntimeError(f"unable to decompress .Z payload: {errors}")


def lines(payload: bytes) -> list[str]:
    return [
        line.strip()
        for line in payload.decode("utf-8-sig", errors="strict").splitlines()
        if line.strip()
    ]


def comma_row(line: str) -> list[str]:
    return [value.strip() for value in line.split(",")]


def label_last(values: list[str]) -> tuple[tuple[str, ...], str]:
    if len(values) < 2:
        raise RuntimeError(f"bad label-last row: {values!r}")
    return tuple(values[:-1]), values[-1].strip()


def label_first(values: list[str]) -> tuple[tuple[str, ...], str]:
    if len(values) < 2:
        raise RuntimeError(f"bad label-first row: {values!r}")
    return tuple(values[1:]), values[0].strip()


def parse_ionosphere(archive: zipfile.ZipFile):
    rows = [
        label_last(comma_row(line))
        for line in lines(archive.read(member(archive, "ionosphere.data")))
    ]
    assert_shape("Ionosphere", rows, 351, 34)
    return rows


def parse_musk(archive: zipfile.ZipFile):
    payload = decompress_z(archive.read(member(archive, "clean1.data.Z")))
    rows = []
    for line in lines(payload):
        values = comma_row(line)
        if len(values) != 169:
            raise RuntimeError(f"unexpected Musk row width {len(values)}")
        rows.append((tuple(values[2:-1]), values[-1]))
    assert_shape("Musk (Version 1)", rows, 476, 166)
    return rows


def parse_spambase(archive: zipfile.ZipFile):
    rows = [
        label_last(comma_row(line))
        for line in lines(archive.read(member(archive, "spambase.data")))
    ]
    assert_shape("Spambase", rows, 4601, 57)
    return rows


def parse_sonar(archive: zipfile.ZipFile):
    rows = [
        label_last(comma_row(line))
        for line in lines(archive.read(member(archive, "sonar.all-data")))
    ]
    assert_shape("Connectionist Bench (Sonar, Mines vs. Rocks)", rows, 208, 60)
    return rows


def parse_hill_valley(archive: zipfile.ZipFile):
    basenames = (
        "Hill_Valley_with_noise_Training.data",
        "Hill_Valley_with_noise_Testing.data",
        "Hill_Valley_without_noise_Training.data",
        "Hill_Valley_without_noise_Testing.data",
    )
    rows = []
    for basename in basenames:
        source = lines(archive.read(member(archive, basename)))
        header = comma_row(source[0])
        if len(header) != 101 or header[-1].lower() != "class":
            raise RuntimeError(f"unexpected Hill-Valley header in {basename}")
        rows.extend(label_last(comma_row(line)) for line in source[1:])
    assert_shape("Hill-Valley", rows, 2424, 100)
    return rows


def parse_libras(archive: zipfile.ZipFile):
    rows = [
        label_last(comma_row(line))
        for line in lines(archive.read(member(archive, "movement_libras.data")))
    ]
    assert_shape("Libras Movement", rows, 360, 90)
    return rows


def parse_urban(archive: zipfile.ZipFile):
    rows = []
    for basename in ("training.csv", "testing.csv"):
        source = lines(archive.read(member(archive, basename)))
        header = comma_row(source[0])
        if len(header) != 148 or header[0].lower() != "class":
            raise RuntimeError(f"unexpected Urban Land Cover header in {basename}")
        rows.extend(label_first(comma_row(line)) for line in source[1:])
    assert_shape("Urban Land Cover", rows, 675, 147)
    return rows


PARSERS = {
    "Ionosphere": parse_ionosphere,
    "Musk (Version 1)": parse_musk,
    "Spambase": parse_spambase,
    "Connectionist Bench (Sonar, Mines vs. Rocks)": parse_sonar,
    "Hill-Valley": parse_hill_valley,
    "Libras Movement": parse_libras,
    "Urban Land Cover": parse_urban,
}


def assert_shape(name: str, rows, expected_rows: int, expected_features: int) -> None:
    widths = {len(features) for features, _ in rows}
    if len(rows) != expected_rows or widths != {expected_features}:
        raise RuntimeError(
            f"unexpected {name} shape: rows={len(rows)}, widths={sorted(widths)}"
        )
    if any(not label for _, label in rows):
        raise RuntimeError(f"empty label in {name}")


def parse_records(name: str, payload: bytes):
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        return PARSERS[name](archive)


def compact_state(task: object, allowed: int, remaining: int, seed: int):
    row = export_v57.compact_state(task, allowed, remaining, seed)
    base_digest = hashlib.sha256(
        f"v68:{task.name}:{allowed}:{remaining}".encode("utf-8")
    ).hexdigest()
    row["base_digest"] = base_digest
    row["digest"] = hashlib.sha256(
        f"{base_digest}:{seed}:clean-lower-bound-v68".encode("utf-8")
    ).hexdigest()
    return row


def protocol() -> dict[str, object]:
    return {
        "profile_seeds": list(PROFILE_SEEDS),
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
        "budget": BUDGET,
        "budget_ladder": list(BUDGET_LADDER),
    }


def run(states_path: Path, reference_path: Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["lock_digest"] != LOCK_DIGEST:
        raise RuntimeError("unexpected archive-lock digest")
    if manifest["repository_registry_digest"] != REGISTRY_DIGEST:
        raise RuntimeError("unexpected registry digest")
    if manifest["parent_v66_evidence_digest"] != V66_DIGEST:
        raise RuntimeError("unexpected v0.66 parent digest")

    tasks = []
    summaries = []
    verification = []
    for dataset in manifest["datasets"]:
        payload = download(str(dataset["url"]))
        actual_hash = hashlib.sha256(payload).hexdigest()
        matched = (
            actual_hash == dataset["sha256"]
            and len(payload) == int(dataset["bytes"])
        )
        verification.append({
            "name": dataset["name"],
            "uci_id": dataset["uci_id"],
            "expected_sha256": dataset["sha256"],
            "actual_sha256": actual_hash,
            "expected_bytes": dataset["bytes"],
            "actual_bytes": len(payload),
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
    rows = []
    base_states = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            base_digest = hashlib.sha256(
                f"v68:{task.name}:{allowed}:{remaining}".encode("utf-8")
            ).hexdigest()
            base_states.add(base_digest)
            for seed in PROFILE_SEEDS:
                profile = response.profile_for_task(task, seed)
                current = bounded_result = plain = None
                current_solved = bounded_solved = plain_solved = False
                try:
                    current = response.ParetoPlanner(task, profile, BUDGET).result(
                        allowed, remaining
                    )
                    current_solved = True
                except response.BudgetExceeded:
                    pass
                try:
                    bounded_result = bounded.LowerBoundParetoPlanner(
                        task, profile, BUDGET
                    ).result(allowed, remaining)
                    bounded_solved = True
                except response.BudgetExceeded:
                    pass
                if bounded_solved:
                    try:
                        plain = response.PlainPlanner(task, profile, BUDGET).result(
                            allowed, remaining
                        )
                        plain_solved = True
                    except response.BudgetExceeded:
                        pass

                compact = compact_state(task, allowed, remaining, seed)
                state_rows.append(compact)
                bounded_exp = (
                    bounded_result.stats.query_expansions
                    if bounded_result is not None else BUDGET + 1
                )
                current_exp = (
                    current.stats.query_expansions
                    if current is not None else BUDGET + 1
                )
                plain_lower = (
                    plain.stats.query_expansions
                    if plain is not None else BUDGET + 1
                )
                certificate = response.pareto_certificate(
                    task, profile, allowed, remaining
                )
                rows.append({
                    "task": task.name,
                    "profile_seed": seed,
                    "state_digest": compact["digest"],
                    "base_state_digest": base_digest,
                    "structural_partition_representatives": representatives,
                    "candidate_count": allowed.bit_count(),
                    "raw_remaining_queries": remaining.bit_count(),
                    "root_pareto_certificate": certificate,
                    "current_solved": current_solved,
                    "bounded_solved": bounded_solved,
                    "plain_solved": plain_solved,
                    "current_plan": (
                        list(bounded.exact_plan_tuple(current.plan))
                        if current is not None else None
                    ),
                    "bounded_plan": (
                        list(bounded.exact_plan_tuple(bounded_result.plan))
                        if bounded_result is not None else None
                    ),
                    "plain_plan_metrics": (
                        list(response.plan_metrics(plain.plan))
                        if plain is not None else None
                    ),
                    "current_bounded_plan_match": (
                        current_solved and bounded_solved
                        and bounded.exact_plan_tuple(current.plan)
                        == bounded.exact_plan_tuple(bounded_result.plan)
                    ),
                    "plain_bounded_objective_match": (
                        plain_solved and bounded_solved
                        and response.plan_metrics(plain.plan)
                        == response.plan_metrics(bounded_result.plan)
                    ),
                    "current_stats": (
                        asdict(current.stats) if current is not None else None
                    ),
                    "bounded_stats": (
                        asdict(bounded_result.stats)
                        if bounded_result is not None else None
                    ),
                    "plain_stats": (
                        asdict(plain.stats) if plain is not None else None
                    ),
                    "current_expansions": current_exp,
                    "bounded_expansions": bounded_exp,
                    "plain_expansions_lower_bound": plain_lower,
                    "plain_over_bounded_ratio": plain_lower / max(1, bounded_exp),
                    "current_over_bounded_ratio": current_exp / max(1, bounded_exp),
                    "bounded_expansion_regression": bounded_exp > current_exp,
                })

    state_rows.sort(key=lambda row: str(row["digest"]))
    rows.sort(key=lambda row: str(row["state_digest"]))
    export_v57.write_text(state_rows, states_path)
    bounded_rows = [row for row in rows if row["bounded_solved"]]
    both_plain = [
        row for row in bounded_rows if row["plain_solved"]
    ]
    both_current = [
        row for row in bounded_rows if row["current_solved"]
    ]
    bounded_only = [
        row for row in bounded_rows if not row["plain_solved"]
    ]
    current_sum = sum(
        int(row["current_expansions"]) for row in both_current
    )
    bounded_sum = sum(
        int(row["bounded_expansions"]) for row in both_current
    )
    ratios = [float(row["plain_over_bounded_ratio"]) for row in bounded_rows]
    ladder = {
        str(budget): {
            "bounded_solved": sum(
                int(row["bounded_solved"] and row["bounded_expansions"] <= budget)
                for row in rows
            ),
            "plain_solved": sum(
                int(row["plain_solved"] and row["plain_expansions_lower_bound"] <= budget)
                for row in rows
            ),
        }
        for budget in BUDGET_LADDER
    }
    result = {
        "status": "clean_lower_bound_python_reference_v68",
        "archive_lock_digest": LOCK_DIGEST,
        "repository_registry_digest": REGISTRY_DIGEST,
        "parent_v66_evidence_digest": V66_DIGEST,
        "archive_verification": {
            "all_hashes_match": all(row["matched"] for row in verification),
            "rows": verification,
        },
        "protocol": protocol(),
        "dataset_summaries": summaries,
        "contributing_dataset_count": sum(
            int(row["selected_states"] > 0) for row in summaries
        ),
        "base_state_count": len(base_states),
        "profiled_state_count": len(rows),
        "bounded_solved_count": len(bounded_rows),
        "both_plain_bounded_count": len(both_plain),
        "bounded_only_count": len(bounded_only),
        "plain_bounded_objective_mismatch_count": sum(
            int(not row["plain_bounded_objective_match"]) for row in both_plain
        ),
        "both_current_bounded_count": len(both_current),
        "current_bounded_plan_mismatch_count": sum(
            int(not row["current_bounded_plan_match"]) for row in both_current
        ),
        "bounded_expansion_regression_count": sum(
            int(row["bounded_expansion_regression"]) for row in both_current
        ),
        "states_with_lower_bound_pruning": sum(
            int(
                row["bounded_stats"] is not None
                and row["bounded_stats"]["bound_pruned_queries"] > 0
            )
            for row in rows
        ),
        "current_query_expansions": current_sum,
        "bounded_query_expansions": bounded_sum,
        "aggregate_bounded_reduction_fraction": (
            1.0 - bounded_sum / current_sum if current_sum else 0.0
        ),
        "dominated_queries_removed": sum(
            int(row["bounded_stats"]["dominated_queries_removed"])
            for row in bounded_rows
        ),
        "root_incomparable_classes": sum(
            int(row["root_pareto_certificate"]["incomparable_pareto_classes"])
            for row in rows
        ),
        "expansion_ratio_median": float(np.median(ratios)) if ratios else None,
        "expansion_ratio_p90": float(np.quantile(ratios, 0.9)) if ratios else None,
        "budget_ladder_summary": ladder,
        "state_input_sha256": hashlib.sha256(states_path.read_bytes()).hexdigest(),
        "rows": rows,
    }
    result["frozen_external_digest"] = hashlib.sha256(
        json.dumps({
            "archive_lock_digest": LOCK_DIGEST,
            "repository_registry_digest": REGISTRY_DIGEST,
            "parent_v66_evidence_digest": V66_DIGEST,
            "protocol": result["protocol"],
            "dataset_summaries": summaries,
            "state_input_sha256": result["state_input_sha256"],
            "state_digests": [row["state_digest"] for row in rows],
        }, sort_keys=True).encode("utf-8")
    ).hexdigest()
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
        "bounded_solved": result["bounded_solved_count"],
        "plain_solved": result["both_plain_bounded_count"],
        "bounded_only": result["bounded_only_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
