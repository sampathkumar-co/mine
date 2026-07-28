from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

from . import pystreed_integration_v50 as v50
from . import pystreed_native_holdout_v51 as v51
from . import pystreed_fingerprint_holdout_v52 as v52


PINNED_COMMIT = v50.PINNED_COMMIT
TRAIN_RANKS = tuple(range(0, 12))
VALIDATION_RANKS = tuple(range(12, 24))
THRESHOLD_CANDIDATES = (0, 12, 16, 24, 32, 48, 64, 96, 128, 256)
MAX_ROWS = 256
MAX_FEATURES = 48
MAX_DEPTH = 3
MAX_NODES = 5
TIME_LIMIT = 180
RANDOM_SEED = 5353
SELECTION_REPETITIONS = 2
VALIDATION_REPETITIONS = 3


class ProtocolError(RuntimeError):
    pass


def patch_parameter(root: Path) -> None:
    path = root / "src" / "solver" / "define_parameters.cpp"
    text = path.read_text(encoding="utf-8")
    anchor = (
        '\t\tparameters.DefineBooleanParameter\n'
        '\t\t(\n'
        '\t\t\t"use-similarity-lower-bound",\n'
        '\t\t\t"Activate similarity-based lower bounding. Disabling this option may be better for some benchmarks, but on most of our tested datasets keeping this on was beneficial.",\n'
        '\t\t\ttrue,\n'
        '\t\t\t"Algorithmic Parameters"\n'
        '\t\t);\n'
    )
    insertion = anchor + (
        '\n\t\tparameters.DefineIntegerParameter\n'
        '\t\t(\n'
        '\t\t\t"local-equivalence-max-instances",\n'
        '\t\t\t"Apply descendant-local equivalent-feature pruning only when the current data view has at most this many instances; zero disables it.",\n'
        '\t\t\t0,\n'
        '\t\t\t"Algorithmic Parameters",\n'
        '\t\t\t0,\n'
        '\t\t\tINT32_MAX\n'
        '\t\t);\n'
    )
    text = v50.replace_once(text, anchor, insertion, "adaptive parameter")
    path.write_text(text, encoding="utf-8")


def patch_adaptive_solver(root: Path) -> None:
    v52.patch_data_header(root)
    v52.patch_data_source(root)
    v52.patch_splitter_header(root)
    v52.patch_splitter_source(root)
    v52.patch_solver(root)
    path = root / "src" / "solver" / "solver.cpp"
    text = path.read_text(encoding="utf-8")
    text = v50.replace_once(
        text,
        "\t\t\t\tif (branch.Depth() > 0) fingerprint_ptr = &split_fingerprint;",
        "\t\t\t\tif (use_local_accuracy_quotient) fingerprint_ptr = &split_fingerprint;",
        "adaptive fingerprint activation",
    )
    text = v50.replace_once(
        text,
        "\t\t\t\tif (branch.Depth() > 0) {",
        "\t\t\t\tif (use_local_accuracy_quotient) {",
        "adaptive quotient activation",
    )
    anchor = (
        "\t\tstd::unordered_map<size_t, std::vector<int>> local_accuracy_partition_buckets;\n\n"
        "\t\t// Loop over each feature"
    )
    insertion = (
        "\t\tstd::unordered_map<size_t, std::vector<int>> local_accuracy_partition_buckets;\n"
        "\t\tbool use_local_accuracy_quotient = false;\n"
        "\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\tconst int local_limit = int(parameters.GetIntegerParameter(\"local-equivalence-max-instances\"));\n"
        "\t\t\tuse_local_accuracy_quotient = branch.Depth() > 0 && local_limit > 0 && data.Size() <= local_limit;\n"
        "\t\t}\n\n"
        "\t\t// Loop over each feature"
    )
    text = v50.replace_once(text, anchor, insertion, "adaptive state guard")
    path.write_text(text, encoding="utf-8")


def prepare_source(source: Path, destination: Path, adaptive: bool) -> None:
    shutil.copytree(source, destination)
    v50.instrument_statistics(destination)
    if adaptive:
        patch_parameter(destination)
        patch_adaptive_solver(destination)
    else:
        v50.instrument_solver(destination, False)


def write_ranked_datasets(
    source: Path,
    output: Path,
    ranks: tuple[int, ...],
    split: str,
) -> list[dict[str, object]]:
    candidates = v51.eligible_datasets(source)
    if not ranks or len(candidates) <= max(ranks):
        raise ProtocolError(
            f"only {len(candidates)} eligible datasets; rank {max(ranks)} required"
        )
    rows = []
    for rank in ranks:
        _, relative, path, labels, features = candidates[rank]
        labels = labels[:MAX_ROWS]
        features = features[:MAX_ROWS]
        columns = [list(values) for values in zip(*features)][:MAX_FEATURES]
        dataset_path = output / "datasets" / f"{split}-{rank}.csv"
        v50.write_dataset(dataset_path, labels, columns)
        rows.append({
            "rank": rank,
            "split": split,
            "source_path": relative,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "used_rows": len(labels),
            "used_features": len(columns),
            "dataset_path": dataset_path,
        })
    return rows


def execute_once(
    binary: Path,
    dataset: Path,
    threshold: int | None,
) -> dict[str, object]:
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
    if threshold is not None:
        command += ["-local-equivalence-max-instances", str(threshold)]
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
        "reported_total_seconds": v50.metric(text, "Total time elapsed:", False),
        "solve_cpu_seconds": v50.metric(text, "CLOCKS FOR SOLVE:", False),
        "wall_seconds": wall,
        "solve_subtree_calls": v50.metric(text, "SolveSubTree calls:"),
        "feature_branch_expansions": v50.metric(text, "Feature branch expansions:"),
        "local_equivalent_features_skipped": v50.metric(
            text, "Local equivalent features skipped:"
        ),
    }


def execute(
    binary: Path,
    dataset: Path,
    threshold: int | None,
    repetitions: int,
) -> dict[str, object]:
    rows = [
        execute_once(binary, dataset, threshold)
        for _ in range(repetitions)
    ]
    structural_keys = (
        "depth", "nodes", "train_score", "test_score",
        "solve_subtree_calls", "feature_branch_expansions",
        "local_equivalent_features_skipped",
    )
    for key in structural_keys:
        if any(row[key] != rows[0][key] for row in rows[1:]):
            raise ProtocolError(f"non-deterministic metric {key} for {dataset}")
    result = {key: rows[0][key] for key in structural_keys}
    result.update({
        "reported_total_seconds": statistics.median(
            row["reported_total_seconds"] for row in rows
        ),
        "solve_cpu_seconds": statistics.median(
            row["solve_cpu_seconds"] for row in rows
        ),
        "wall_seconds": statistics.median(row["wall_seconds"] for row in rows),
        "repetitions": rows,
    })
    return result


def optimum_matched(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        left["depth"] == right["depth"]
        and left["nodes"] == right["nodes"]
        and abs(left["train_score"] - right["train_score"]) <= 1e-12
        and abs(left["test_score"] - right["test_score"]) <= 1e-12
    )


def baseline_results(
    binary: Path,
    datasets: list[dict[str, object]],
    repetitions: int,
) -> dict[int, dict[str, object]]:
    return {
        dataset["rank"]: execute(
            binary, dataset["dataset_path"], None, repetitions
        )
        for dataset in datasets
    }


def benchmark_threshold(
    binary: Path,
    datasets: list[dict[str, object]],
    originals: dict[int, dict[str, object]],
    threshold: int,
    repetitions: int,
) -> dict[str, object]:
    rows = []
    for dataset in datasets:
        original = originals[dataset["rank"]]
        patched = execute(
            binary, dataset["dataset_path"], threshold, repetitions
        )
        rows.append({
            "rank": dataset["rank"],
            "source_path": dataset["source_path"],
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
    return {
        "threshold": threshold,
        "row_count": len(rows),
        "optimum_match_count": sum(int(row["optimum_matched"]) for row in rows),
        "tasks_with_local_skips": sum(
            int(row["patched"]["local_equivalent_features_skipped"] > 0)
            for row in rows
        ),
        "total_local_skips": sum(
            row["patched"]["local_equivalent_features_skipped"] for row in rows
        ),
        "subtree_call_reduction_fraction": (
            1.0 - patched_calls / max(1, original_calls)
        ),
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
        "rows": rows,
    }


def select_threshold(candidates: list[dict[str, object]]) -> dict[str, object]:
    admissible = [
        row for row in candidates
        if row["threshold"] > 0
        and row["optimum_match_count"] == row["row_count"]
        and row["patched_over_original_runtime_max"] <= 1.15
        and row["tasks_with_local_skips"] >= 4
        and row["branch_expansion_reduction_fraction"] > 0
    ]
    if not admissible:
        return {
            "selected": False,
            "reason": "no positive threshold met the locked admissibility rule",
        }
    best = max(admissible, key=lambda row: (
        row["reported_speedup_median"],
        row["branch_expansion_reduction_fraction"],
        -row["threshold"],
    ))
    return {
        "selected": True,
        "threshold": best["threshold"],
        "training_speedup_median": best["reported_speedup_median"],
        "training_branch_reduction": best[
            "branch_expansion_reduction_fraction"
        ],
        "training_max_runtime_ratio": best[
            "patched_over_original_runtime_max"
        ],
    }


def run(source: Path, work: Path) -> dict[str, object]:
    actual_commit = v50.run_command(
        ["git", "rev-parse", "HEAD"], cwd=source
    ).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        raise ProtocolError(f"source revision {actual_commit} != {PINNED_COMMIT}")
    work.mkdir(parents=True, exist_ok=True)
    original_root = work / "pystreed-original"
    adaptive_root = work / "pystreed-selective"
    prepare_source(source, original_root, False)
    prepare_source(source, adaptive_root, True)
    original_binary = v50.build_solver(original_root)
    adaptive_binary = v50.build_solver(adaptive_root)

    training_datasets = write_ranked_datasets(
        source, work, TRAIN_RANKS, "training"
    )
    training_originals = baseline_results(
        original_binary, training_datasets, SELECTION_REPETITIONS
    )
    candidates = [
        benchmark_threshold(
            adaptive_binary,
            training_datasets,
            training_originals,
            threshold,
            SELECTION_REPETITIONS,
        )
        for threshold in THRESHOLD_CANDIDATES
    ]
    selection = select_threshold(candidates)
    if not selection["selected"]:
        return {
            "status": "not_yet",
            "development_gate": False,
            "pinned_commit": PINNED_COMMIT,
            "protocol": protocol(),
            "selection": selection,
            "training_candidates": candidates,
        }

    selected_threshold = int(selection["threshold"])
    validation_datasets = write_ranked_datasets(
        source, work, VALIDATION_RANKS, "validation"
    )
    validation_originals = baseline_results(
        original_binary, validation_datasets, VALIDATION_REPETITIONS
    )
    validation = benchmark_threshold(
        adaptive_binary,
        validation_datasets,
        validation_originals,
        selected_threshold,
        VALIDATION_REPETITIONS,
    )
    gate = (
        validation["row_count"] == len(VALIDATION_RANKS)
        and validation["optimum_match_count"] == len(VALIDATION_RANKS)
        and validation["tasks_with_local_skips"] >= 6
        and validation["branch_expansion_reduction_fraction"] >= 0.01
        and validation["subtree_call_reduction_fraction"] >= 0.01
        and validation["reported_speedup_median"] >= 1.0
        and validation["tasks_speedup_ge_1_03"] >= 3
        and validation["patched_over_original_runtime_max"] <= 1.12
    )
    frozen_digest = hashlib.sha256(json.dumps({
        "pinned_commit": PINNED_COMMIT,
        "threshold_candidates": THRESHOLD_CANDIDATES,
        "training_ranks": TRAIN_RANKS,
        "validation_ranks": VALIDATION_RANKS,
        "selection_rule": (
            "max_median_speedup_then_branch_reduction_then_smaller_threshold"
        ),
        "selected_threshold": selected_threshold,
        "protocol": protocol(),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": (
            "selective_threshold_candidate" if gate else "not_yet"
        ),
        "development_gate": gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "claim_scope": (
            "PySTreeD Accuracy only. A candidate-size threshold is selected on "
            "ranks 0-11 and evaluated on ranks 12-23. No ranks beyond 23 are "
            "benchmarked by this workflow. A separate frozen workflow is required "
            "before any external holdout claim."
        ),
        "protocol": protocol(),
        "selection": selection,
        "selected_threshold": selected_threshold,
        "training_candidates": candidates,
        "validation": validation,
        "frozen_selective_digest": frozen_digest,
    }


def protocol() -> dict[str, object]:
    return {
        "training_ranks": list(TRAIN_RANKS),
        "validation_ranks": list(VALIDATION_RANKS),
        "threshold_candidates": list(THRESHOLD_CANDIDATES),
        "max_rows": MAX_ROWS,
        "max_features": MAX_FEATURES,
        "max_depth": MAX_DEPTH,
        "max_nodes": MAX_NODES,
        "selection_repetitions": SELECTION_REPETITIONS,
        "validation_repetitions": VALIDATION_REPETITIONS,
        "random_seed": RANDOM_SEED,
        "terminal_solver": False,
        "similarity_lower_bound": False,
        "feature_ordering": "in-order",
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
        "selection": report.get("selection"),
        "validation_speedup": (
            report.get("validation", {}).get("reported_speedup_median")
        ),
        "validation_branch_reduction": (
            report.get("validation", {}).get(
                "branch_expansion_reduction_fraction"
            )
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
