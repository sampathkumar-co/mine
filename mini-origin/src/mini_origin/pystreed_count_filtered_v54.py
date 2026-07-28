from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics

from . import pystreed_integration_v50 as v50
from . import pystreed_native_holdout_v51 as v51


PINNED_COMMIT = v50.PINNED_COMMIT
DATASET_RANKS = tuple(range(0, 24))
MAX_ROWS = 256
MAX_FEATURES = 48
MAX_DEPTH = 3
MAX_NODES = 5
TIME_LIMIT = 180
RANDOM_SEED = 5454
REPETITIONS = 3


class ProtocolError(RuntimeError):
    pass


def patch_count_filtered_solver(root: Path) -> None:
    path = root / "src" / "solver" / "solver.cpp"
    text = path.read_text(encoding="utf-8-sig")
    text = v50.replace_once(
        text,
        '#include "utils/utils.h"\n',
        '#include "utils/utils.h"\n#include <unordered_map>\n',
        "unordered map include",
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
        "\t\tstd::unordered_map<size_t, std::vector<int>> local_accuracy_count_buckets;\n\n"
        "\t\t// Loop over each feature",
        "count buckets",
    )
    text = v50.replace_once(
        text,
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n\t\t\t// Generate the context descriptors",
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n"
        "\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n"
        "\t\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\t\tif (branch.Depth() > 0) {\n"
        "\t\t\t\t\tsize_t left_count_hash = static_cast<size_t>(left_data.Size());\n"
        "\t\t\t\t\tsize_t right_count_hash = static_cast<size_t>(right_data.Size());\n"
        "\t\t\t\t\tfor (int label = 0; label < data.NumLabels(); label++) {\n"
        "\t\t\t\t\t\tleft_count_hash ^= static_cast<size_t>(left_data.NumInstancesForLabel(label)) + 0x9e3779b9 + (left_count_hash << 6) + (left_count_hash >> 2);\n"
        "\t\t\t\t\t\tright_count_hash ^= static_cast<size_t>(right_data.NumInstancesForLabel(label)) + 0x9e3779b9 + (right_count_hash << 6) + (right_count_hash >> 2);\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t\tconst size_t count_key = std::min(left_count_hash, right_count_hash);\n"
        "\t\t\t\t\tauto& bucket = local_accuracy_count_buckets[count_key];\n"
        "\t\t\t\t\tbool equivalent_partition = false;\n"
        "\t\t\t\t\tfor (int previous_feature : bucket) {\n"
        "\t\t\t\t\t\tADataView previous_left;\n"
        "\t\t\t\t\t\tADataView previous_right;\n"
        "\t\t\t\t\t\tdata_splitter.Split(data, branch, previous_feature, previous_left, previous_right);\n"
        "\t\t\t\t\t\tif ((left_data == previous_left && right_data == previous_right)\n"
        "\t\t\t\t\t\t\t|| (left_data == previous_right && right_data == previous_left)) {\n"
        "\t\t\t\t\t\t\tequivalent_partition = true;\n"
        "\t\t\t\t\t\t\tbreak;\n"
        "\t\t\t\t\t\t}\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t\tif (equivalent_partition) {\n"
        "\t\t\t\t\t\tstats.num_local_equivalent_features_skipped++;\n"
        "\t\t\t\t\t\tcontinue;\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t\tbucket.push_back(feature);\n"
        "\t\t\t\t}\n"
        "\t\t\t}\n"
        "\t\t\tstats.num_feature_branch_expansions++;\n\n"
        "\t\t\t// Generate the context descriptors",
        "count-filtered quotient",
    )
    path.write_text(text, encoding="utf-8")


def prepare_source(source: Path, destination: Path, patched: bool) -> None:
    shutil.copytree(source, destination)
    v50.instrument_statistics(destination)
    if patched:
        patch_count_filtered_solver(destination)
    else:
        v50.instrument_solver(destination, False)


def write_datasets(source: Path, output: Path) -> list[dict[str, object]]:
    candidates = v51.eligible_datasets(source)
    if len(candidates) <= max(DATASET_RANKS):
        raise ProtocolError(
            f"only {len(candidates)} eligible datasets; rank {max(DATASET_RANKS)} required"
        )
    rows = []
    for rank in DATASET_RANKS:
        _, relative, path, labels, features = candidates[rank]
        labels = labels[:MAX_ROWS]
        features = features[:MAX_ROWS]
        columns = [list(values) for values in zip(*features)][:MAX_FEATURES]
        dataset_path = output / "datasets" / f"dataset-{rank}.csv"
        v50.write_dataset(dataset_path, labels, columns)
        rows.append({
            "rank": rank,
            "source_path": relative,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "used_rows": len(labels),
            "used_features": len(columns),
            "dataset_path": dataset_path,
        })
    return rows


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
    result = v50.run_command(command, timeout=TIME_LIMIT + 60)
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
        "reported_total_seconds": v50.metric(text, "Total time elapsed:", False),
        "solve_cpu_seconds": v50.metric(text, "CLOCKS FOR SOLVE:", False),
        "solve_subtree_calls": v50.metric(text, "SolveSubTree calls:"),
        "feature_branch_expansions": v50.metric(text, "Feature branch expansions:"),
        "local_equivalent_features_skipped": v50.metric(
            text, "Local equivalent features skipped:"
        ),
    }


def execute(binary: Path, dataset: Path) -> dict[str, object]:
    repetitions = [execute_once(binary, dataset) for _ in range(REPETITIONS)]
    structural_keys = (
        "depth", "nodes", "train_score", "test_score",
        "solve_subtree_calls", "feature_branch_expansions",
        "local_equivalent_features_skipped",
    )
    for key in structural_keys:
        if any(row[key] != repetitions[0][key] for row in repetitions[1:]):
            raise ProtocolError(f"non-deterministic metric {key} for {dataset}")
    result = {key: repetitions[0][key] for key in structural_keys}
    result.update({
        "reported_total_seconds": statistics.median(
            row["reported_total_seconds"] for row in repetitions
        ),
        "solve_cpu_seconds": statistics.median(
            row["solve_cpu_seconds"] for row in repetitions
        ),
        "repetitions": repetitions,
    })
    return result


def optimum_matched(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        left["depth"] == right["depth"]
        and left["nodes"] == right["nodes"]
        and abs(left["train_score"] - right["train_score"]) <= 1e-12
        and abs(left["test_score"] - right["test_score"]) <= 1e-12
    )


def run(source: Path, work: Path) -> dict[str, object]:
    actual_commit = v50.run_command(
        ["git", "rev-parse", "HEAD"], cwd=source
    ).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        raise ProtocolError(f"source revision {actual_commit} != {PINNED_COMMIT}")
    work.mkdir(parents=True, exist_ok=True)
    original_root = work / "pystreed-original"
    patched_root = work / "pystreed-count-filtered"
    prepare_source(source, original_root, False)
    prepare_source(source, patched_root, True)
    original_binary = v50.build_solver(original_root)
    patched_binary = v50.build_solver(patched_root)
    datasets = write_datasets(source, work)
    rows = []
    for dataset in datasets:
        original = execute(original_binary, dataset["dataset_path"])
        patched = execute(patched_binary, dataset["dataset_path"])
        rows.append({
            "rank": dataset["rank"],
            "source_path": dataset["source_path"],
            "source_sha256": dataset["source_sha256"],
            "used_rows": dataset["used_rows"],
            "used_features": dataset["used_features"],
            "optimum_matched": optimum_matched(original, patched),
            "original": original,
            "patched": patched,
            "reported_speedup": (
                original["reported_total_seconds"]
                / max(1e-9, patched["reported_total_seconds"])
            ),
        })
    original_calls = sum(row["original"]["solve_subtree_calls"] for row in rows)
    patched_calls = sum(row["patched"]["solve_subtree_calls"] for row in rows)
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
    summary = {
        "row_count": len(rows),
        "optimum_match_count": sum(int(row["optimum_matched"]) for row in rows),
        "tasks_with_local_skips": sum(
            int(row["patched"]["local_equivalent_features_skipped"] > 0)
            for row in rows
        ),
        "total_local_skips": sum(
            row["patched"]["local_equivalent_features_skipped"] for row in rows
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
        "patched_over_original_runtime_median": statistics.median(runtime_ratios),
        "patched_over_original_runtime_max": max(runtime_ratios),
        "tasks_speedup_ge_1_03": sum(
            int(row["reported_speedup"] >= 1.03) for row in rows
        ),
    }
    gate = (
        summary["row_count"] == len(DATASET_RANKS)
        and summary["optimum_match_count"] == len(DATASET_RANKS)
        and summary["tasks_with_local_skips"] >= 18
        and summary["subtree_call_reduction_fraction"] >= 0.02
        and summary["branch_expansion_reduction_fraction"] >= 0.02
        and summary["reported_speedup_median"] >= 1.0
        and summary["tasks_speedup_ge_1_03"] >= 8
        and summary["patched_over_original_runtime_max"] <= 1.10
    )
    return {
        "status": (
            "count_filtered_quotient_candidate" if gate else "not_yet"
        ),
        "development_gate": gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "scope": (
            "PySTreeD Accuracy only. Per-label child-count hashes filter possible "
            "equivalent splits; every hash match is verified with exact ADataView "
            "equality up to swapping children. No extra instance scan, bitset, or "
            "fingerprint is introduced. Ranks 0-23 are opened development data."
        ),
        "protocol": {
            "dataset_ranks": list(DATASET_RANKS),
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
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.source.resolve(), args.work.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "optimum_matches": report["summary"]["optimum_match_count"],
        "local_skips": report["summary"]["total_local_skips"],
        "branch_reduction": report["summary"]["branch_expansion_reduction_fraction"],
        "median_speedup": report["summary"]["reported_speedup_median"],
    }, indent=2))


if __name__ == "__main__":
    main()
