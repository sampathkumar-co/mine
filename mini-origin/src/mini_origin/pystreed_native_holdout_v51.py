from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import time

from . import pystreed_integration_v50 as v50


PINNED_COMMIT = v50.PINNED_COMMIT
DEVELOPMENT_COUNT = 4
HOLDOUT_COUNT = 8
MAX_ROWS = 256
MAX_FEATURES = 48
MAX_DEPTH = 3
MAX_NODES = 5
TIME_LIMIT = 180
RANDOM_SEED = 5151
REPETITIONS = 3


class ProtocolError(RuntimeError):
    pass


def instrument_hashed_solver(root: Path) -> None:
    path = root / "src" / "solver" / "solver.cpp"
    text = path.read_text(encoding="utf-8-sig")
    text = v50.replace_once(
        text,
        '#include "utils/utils.h"\n',
        '#include "utils/utils.h"\n#include <unordered_map>\n',
        "unordered_map include",
    )
    text = v50.replace_once(
        text,
        "\ttypename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int org_max_depth, int org_num_nodes) {\n\t\tint max_depth = org_max_depth, num_nodes = org_num_nodes;",
        "\ttypename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int org_max_depth, int org_num_nodes) {\n"
        "\t\tstats.num_solve_subtree_calls++;\n"
        "\t\tint max_depth = org_max_depth, num_nodes = org_num_nodes;",
        "SolveSubTree counter",
    )
    text = v50.replace_once(
        text,
        "\t\tfeature_selector->Initialize(data);\n\n\t\t// Loop over each feature",
        "\t\tfeature_selector->Initialize(data);\n"
        "\t\tstd::unordered_map<size_t, std::vector<std::pair<ADataViewBitSet, ADataViewBitSet>>> local_accuracy_partition_buckets;\n\n"
        "\t\t// Loop over each feature",
        "hashed partition archive",
    )
    text = v50.replace_once(
        text,
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n\t\t\t// Generate the context descriptors",
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n"
        "\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n"
        "\t\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\t\tADataViewBitSet left_partition(left_data);\n"
        "\t\t\t\tADataViewBitSet right_partition(right_data);\n"
        "\t\t\t\tconst size_t left_hash = std::hash<ADataViewBitSet>()(left_partition);\n"
        "\t\t\t\tconst size_t right_hash = std::hash<ADataViewBitSet>()(right_partition);\n"
        "\t\t\t\tconst size_t low_hash = std::min(left_hash, right_hash);\n"
        "\t\t\t\tconst size_t high_hash = std::max(left_hash, right_hash);\n"
        "\t\t\t\tconst size_t combined_hash = low_hash ^ (high_hash + 0x9e3779b97f4a7c15ULL + (low_hash << 6) + (low_hash >> 2));\n"
        "\t\t\t\tauto& bucket = local_accuracy_partition_buckets[combined_hash];\n"
        "\t\t\t\tbool equivalent_partition = false;\n"
        "\t\t\t\tfor (const auto& previous : bucket) {\n"
        "\t\t\t\t\tif ((left_partition == previous.first && right_partition == previous.second)\n"
        "\t\t\t\t\t\t|| (left_partition == previous.second && right_partition == previous.first)) {\n"
        "\t\t\t\t\t\tequivalent_partition = true;\n"
        "\t\t\t\t\t\tbreak;\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t}\n"
        "\t\t\t\tif (equivalent_partition) {\n"
        "\t\t\t\t\tstats.num_local_equivalent_features_skipped++;\n"
        "\t\t\t\t\tcontinue;\n"
        "\t\t\t\t}\n"
        "\t\t\t\tbucket.emplace_back(left_partition, right_partition);\n"
        "\t\t\t}\n"
        "\t\t\tstats.num_feature_branch_expansions++;\n\n"
        "\t\t\t// Generate the context descriptors",
        "hashed local quotient insertion",
    )
    path.write_text(text, encoding="utf-8")


def prepare_source(source: Path, destination: Path, local: bool) -> None:
    shutil.copytree(source, destination)
    v50.instrument_statistics(destination)
    if local:
        instrument_hashed_solver(destination)
    else:
        v50.instrument_solver(destination, False)


def eligible_datasets(source: Path):
    candidates = []
    for path in sorted((source / "data").rglob("*.csv")):
        parsed = v50.parse_table(path)
        if parsed is None:
            continue
        labels, features = parsed
        relative = path.relative_to(source).as_posix()
        candidates.append((
            hashlib.sha256(relative.encode("utf-8")).hexdigest(),
            relative,
            path,
            labels,
            features,
        ))
    candidates.sort(key=lambda row: (row[0], row[1]))
    return candidates


def write_native_suite(
    source: Path,
    output: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = eligible_datasets(source)
    required = DEVELOPMENT_COUNT + HOLDOUT_COUNT
    if len(candidates) < required:
        raise ProtocolError(
            f"only {len(candidates)} eligible datasets; need {required}"
        )
    rows = []
    for rank, (_, relative, path, labels, features) in enumerate(
        candidates[:required]
    ):
        labels = labels[:MAX_ROWS]
        features = features[:MAX_ROWS]
        columns = [
            list(values) for values in zip(*features)
        ][:MAX_FEATURES]
        split = "development" if rank < DEVELOPMENT_COUNT else "holdout"
        dataset_path = output / "datasets" / f"{split}-{rank}.csv"
        v50.write_dataset(dataset_path, labels, columns)
        rows.append({
            "rank": rank,
            "split": split,
            "source_path": relative,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_rows": len(v50.parse_table(path)[0]),
            "used_rows": len(labels),
            "source_features": len(features[0]),
            "used_features": len(columns),
            "label_count": len(set(labels)),
            "dataset_path": dataset_path,
        })
    return (
        [row for row in rows if row["split"] == "development"],
        [row for row in rows if row["split"] == "holdout"],
    )


def execute_once(binary: Path, dataset: Path) -> dict[str, object]:
    command = [
        str(binary),
        "-task", "accuracy",
        "-file", str(dataset),
        "-max-depth", str(MAX_DEPTH),
        "-max-num-nodes", str(MAX_NODES),
        "-max-num-features", str(MAX_FEATURES),
        "-num-instances", str(MAX_ROWS),
        "-use-terminal-solver", "0",
        "-use-similarity-lower-bound", "0",
        "-feature-ordering", "in-order",
        "-random-seed", str(RANDOM_SEED),
        "-time", str(TIME_LIMIT),
        "-verbose", "1",
    ]
    start = time.perf_counter()
    result = v50.run_command(command, timeout=TIME_LIMIT + 60)
    wall = time.perf_counter() - start
    text = result.stdout
    if "Warning: No proof of optimality" in text or "No tree found" in text:
        raise ProtocolError(f"solver did not prove optimum for {dataset}")
    solution = v50.SOLUTION_RE.search(text)
    if not solution:
        raise ProtocolError(f"could not parse solution for {dataset}")
    return {
        "depth": int(solution.group(1)),
        "nodes": int(solution.group(2)),
        "train_score": float(solution.group(3)),
        "test_score": float(solution.group(4)),
        "reported_total_seconds": v50.metric(
            text, "Total time elapsed:", False
        ),
        "solve_cpu_seconds": v50.metric(
            text, "CLOCKS FOR SOLVE:", False
        ),
        "wall_seconds": wall,
        "solve_subtree_calls": v50.metric(text, "SolveSubTree calls:"),
        "feature_branch_expansions": v50.metric(
            text, "Feature branch expansions:"
        ),
        "local_equivalent_features_skipped": v50.metric(
            text, "Local equivalent features skipped:"
        ),
        "stdout_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def execute(binary: Path, dataset: Path) -> dict[str, object]:
    repetitions = [
        execute_once(binary, dataset) for _ in range(REPETITIONS)
    ]
    structural = {
        key: repetitions[0][key]
        for key in (
            "depth", "nodes", "train_score", "test_score",
            "solve_subtree_calls", "feature_branch_expansions",
            "local_equivalent_features_skipped",
        )
    }
    for row in repetitions[1:]:
        for key, value in structural.items():
            if row[key] != value:
                raise ProtocolError(
                    f"non-deterministic structural metric {key} for {dataset}"
                )
    structural.update({
        "reported_total_seconds": statistics.median(
            row["reported_total_seconds"] for row in repetitions
        ),
        "solve_cpu_seconds": statistics.median(
            row["solve_cpu_seconds"] for row in repetitions
        ),
        "wall_seconds": statistics.median(
            row["wall_seconds"] for row in repetitions
        ),
        "repetitions": repetitions,
    })
    return structural


def optimum_matched(
    original: dict[str, object],
    patched: dict[str, object],
) -> bool:
    return (
        original["depth"] == patched["depth"]
        and original["nodes"] == patched["nodes"]
        and abs(original["train_score"] - patched["train_score"]) <= 1e-12
        and abs(original["test_score"] - patched["test_score"]) <= 1e-12
    )


def benchmark_rows(
    datasets: list[dict[str, object]],
    original_binary: Path,
    patched_binary: Path,
) -> list[dict[str, object]]:
    rows = []
    for dataset in datasets:
        original = execute(original_binary, dataset["dataset_path"])
        patched = execute(patched_binary, dataset["dataset_path"])
        rows.append({
            "rank": dataset["rank"],
            "split": dataset["split"],
            "source_path": dataset["source_path"],
            "source_sha256": dataset["source_sha256"],
            "used_rows": dataset["used_rows"],
            "used_features": dataset["used_features"],
            "optimum_matched": optimum_matched(original, patched),
            "original": original,
            "patched": patched,
            "subtree_call_reduction_fraction": (
                1.0 - patched["solve_subtree_calls"]
                / max(1, original["solve_subtree_calls"])
            ),
            "branch_expansion_reduction_fraction": (
                1.0 - patched["feature_branch_expansions"]
                / max(1, original["feature_branch_expansions"])
            ),
            "reported_speedup": (
                original["reported_total_seconds"]
                / max(1e-9, patched["reported_total_seconds"])
            ),
        })
    return rows


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    original_calls = sum(
        row["original"]["solve_subtree_calls"] for row in rows
    )
    patched_calls = sum(
        row["patched"]["solve_subtree_calls"] for row in rows
    )
    original_expansions = sum(
        row["original"]["feature_branch_expansions"] for row in rows
    )
    patched_expansions = sum(
        row["patched"]["feature_branch_expansions"] for row in rows
    )
    runtime_ratios = [
        row["patched"]["reported_total_seconds"]
        / max(1e-9, row["original"]["reported_total_seconds"])
        for row in rows
    ]
    return {
        "row_count": len(rows),
        "optimum_match_count": sum(
            int(row["optimum_matched"]) for row in rows
        ),
        "tasks_with_local_skips": sum(
            int(row["patched"]["local_equivalent_features_skipped"] > 0)
            for row in rows
        ),
        "total_local_skips": sum(
            row["patched"]["local_equivalent_features_skipped"]
            for row in rows
        ),
        "original_subtree_calls": original_calls,
        "patched_subtree_calls": patched_calls,
        "subtree_call_reduction_fraction": (
            1.0 - patched_calls / max(1, original_calls)
        ),
        "original_branch_expansions": original_expansions,
        "patched_branch_expansions": patched_expansions,
        "branch_expansion_reduction_fraction": (
            1.0 - patched_expansions / max(1, original_expansions)
        ),
        "reported_speedup_median": statistics.median(
            row["reported_speedup"] for row in rows
        ),
        "patched_over_original_runtime_median": statistics.median(
            runtime_ratios
        ),
        "patched_over_original_runtime_max": max(runtime_ratios),
        "tasks_speedup_ge_1_05": sum(
            int(row["reported_speedup"] >= 1.05) for row in rows
        ),
    }


def run(source: Path, work: Path) -> dict[str, object]:
    actual_commit = v50.run_command(
        ["git", "rev-parse", "HEAD"], cwd=source
    ).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        raise ProtocolError(
            f"source revision {actual_commit} != {PINNED_COMMIT}"
        )
    work.mkdir(parents=True, exist_ok=True)
    original_root = work / "pystreed-original"
    patched_root = work / "pystreed-hashed-local-quotient"
    prepare_source(source, original_root, False)
    prepare_source(source, patched_root, True)
    original_binary = v50.build_solver(original_root)
    patched_binary = v50.build_solver(patched_root)
    development_datasets, holdout_datasets = write_native_suite(
        source, work
    )
    development_rows = benchmark_rows(
        development_datasets, original_binary, patched_binary
    )
    development = aggregate(development_rows)
    development_gate = (
        development["row_count"] == DEVELOPMENT_COUNT
        and development["optimum_match_count"] == DEVELOPMENT_COUNT
        and development["patched_over_original_runtime_median"] <= 1.05
        and development["patched_over_original_runtime_max"] <= 1.20
    )
    holdout_rows = benchmark_rows(
        holdout_datasets, original_binary, patched_binary
    )
    holdout = aggregate(holdout_rows)
    gate = (
        development_gate
        and holdout["row_count"] == HOLDOUT_COUNT
        and holdout["optimum_match_count"] == HOLDOUT_COUNT
        and holdout["tasks_with_local_skips"] >= 6
        and holdout["subtree_call_reduction_fraction"] >= 0.03
        and holdout["branch_expansion_reduction_fraction"] >= 0.03
        and holdout["reported_speedup_median"] >= 1.0
        and holdout["tasks_speedup_ge_1_05"] >= 3
        and holdout["patched_over_original_runtime_max"] <= 1.25
    )
    return {
        "status": (
            "pystreed_native_holdout_candidate" if gate else "not_yet"
        ),
        "native_holdout_gate": gate,
        "development_gate": development_gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "scope": (
            "PySTreeD Accuracy only. The four opened v0.50 datasets are used "
            "solely to validate the hashed implementation's overhead. The next "
            "eight eligible datasets in the previously frozen path-hash order "
            "form a native, unaugmented holdout."
        ),
        "protocol": {
            "development_ranks": [0, DEVELOPMENT_COUNT - 1],
            "holdout_ranks": [
                DEVELOPMENT_COUNT,
                DEVELOPMENT_COUNT + HOLDOUT_COUNT - 1,
            ],
            "max_rows": MAX_ROWS,
            "max_features": MAX_FEATURES,
            "max_depth": MAX_DEPTH,
            "max_nodes": MAX_NODES,
            "repetitions": REPETITIONS,
            "random_seed": RANDOM_SEED,
            "terminal_solver": False,
            "similarity_lower_bound": False,
            "feature_ordering": "in-order",
        },
        "development_summary": development,
        "holdout_summary": holdout,
        "development_rows": development_rows,
        "holdout_rows": holdout_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source.resolve(), args.work.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "development_gate": report["development_gate"],
        "holdout_optimum_matches": report[
            "holdout_summary"
        ]["optimum_match_count"],
        "holdout_local_skips": report[
            "holdout_summary"
        ]["total_local_skips"],
        "holdout_branch_reduction": report[
            "holdout_summary"
        ]["branch_expansion_reduction_fraction"],
        "holdout_median_speedup": report[
            "holdout_summary"
        ]["reported_speedup_median"],
    }, indent=2))


if __name__ == "__main__":
    main()
