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
TRAIN_RANKS = tuple(range(0, 12))
VALIDATION_RANKS = tuple(range(12, 24))
POLICY_CANDIDATES = (
    (4, 1),
    (6, 1),
    (8, 1),
    (8, 2),
    (12, 1),
    (12, 2),
    (16, 1),
    (16, 2),
)
MAX_ROWS = 256
MAX_FEATURES = 48
MAX_DEPTH = 3
MAX_NODES = 5
TIME_LIMIT = 180
RANDOM_SEED = 5555
SELECTION_REPETITIONS = 2
VALIDATION_REPETITIONS = 3


class ProtocolError(RuntimeError):
    pass


def patch_parameters(root: Path) -> None:
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
        '\t\t\t"local-equivalence-probe-features",\n'
        '\t\t\t"Number of valid descendant features inspected before deciding whether local equivalent-feature pruning remains active; zero disables it.",\n'
        '\t\t\t0,\n'
        '\t\t\t"Algorithmic Parameters",\n'
        '\t\t\t0,\n'
        '\t\t\tINT32_MAX\n'
        '\t\t);\n'
        '\n\t\tparameters.DefineIntegerParameter\n'
        '\t\t(\n'
        '\t\t\t"local-equivalence-probe-duplicates",\n'
        '\t\t\t"Minimum exact duplicate partitions required during the probe to keep local equivalent-feature pruning active.",\n'
        '\t\t\t1,\n'
        '\t\t\t"Algorithmic Parameters",\n'
        '\t\t\t1,\n'
        '\t\t\tINT32_MAX\n'
        '\t\t);\n'
    )
    text = v50.replace_once(text, anchor, insertion, "probe parameters")
    path.write_text(text, encoding="utf-8")


def patch_probe_solver(root: Path) -> None:
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
        "\t\tstd::unordered_map<size_t, std::vector<int>> local_accuracy_count_buckets;\n"
        "\t\tint local_probe_seen = 0;\n"
        "\t\tint local_probe_duplicates = 0;\n"
        "\t\tbool local_probe_decided = false;\n"
        "\t\tbool local_probe_active = false;\n"
        "\t\tint local_probe_limit = 0;\n"
        "\t\tint local_probe_required = 1;\n"
        "\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\tlocal_probe_limit = int(parameters.GetIntegerParameter(\"local-equivalence-probe-features\"));\n"
        "\t\t\tlocal_probe_required = int(parameters.GetIntegerParameter(\"local-equivalence-probe-duplicates\"));\n"
        "\t\t}\n\n"
        "\t\t// Loop over each feature",
        "probe state",
    )
    text = v50.replace_once(
        text,
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n\t\t\t// Generate the context descriptors",
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n"
        "\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n"
        "\t\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\t\tconst bool probe_enabled = branch.Depth() > 0 && local_probe_limit > 0;\n"
        "\t\t\t\tconst bool check_equivalence = probe_enabled && (!local_probe_decided || local_probe_active);\n"
        "\t\t\t\tif (check_equivalence) {\n"
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
        "\t\t\t\t\tif (!equivalent_partition) bucket.push_back(feature);\n"
        "\t\t\t\t\tif (!local_probe_decided) {\n"
        "\t\t\t\t\t\tlocal_probe_seen++;\n"
        "\t\t\t\t\t\tif (equivalent_partition) local_probe_duplicates++;\n"
        "\t\t\t\t\t\tif (local_probe_seen >= local_probe_limit) {\n"
        "\t\t\t\t\t\t\tlocal_probe_decided = true;\n"
        "\t\t\t\t\t\t\tlocal_probe_active = local_probe_duplicates >= local_probe_required;\n"
        "\t\t\t\t\t\t}\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t\tif (equivalent_partition) {\n"
        "\t\t\t\t\t\tstats.num_local_equivalent_features_skipped++;\n"
        "\t\t\t\t\t\tcontinue;\n"
        "\t\t\t\t\t}\n"
        "\t\t\t\t}\n"
        "\t\t\t}\n"
        "\t\t\tstats.num_feature_branch_expansions++;\n\n"
        "\t\t\t// Generate the context descriptors",
        "probe-activated quotient",
    )
    path.write_text(text, encoding="utf-8")


def prepare_source(source: Path, destination: Path, patched: bool) -> None:
    shutil.copytree(source, destination)
    v50.instrument_statistics(destination)
    if patched:
        patch_parameters(destination)
        patch_probe_solver(destination)
    else:
        v50.instrument_solver(destination, False)


def write_ranked_datasets(
    source: Path,
    output: Path,
    ranks: tuple[int, ...],
    split: str,
) -> list[dict[str, object]]:
    candidates = v51.eligible_datasets(source)
    if len(candidates) <= max(ranks):
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
    policy: tuple[int, int] | None,
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
    if policy is not None:
        command += [
            "-local-equivalence-probe-features", str(policy[0]),
            "-local-equivalence-probe-duplicates", str(policy[1]),
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


def execute(
    binary: Path,
    dataset: Path,
    policy: tuple[int, int] | None,
    repetitions: int,
) -> dict[str, object]:
    rows = [execute_once(binary, dataset, policy) for _ in range(repetitions)]
    structural_keys = (
        "depth", "nodes", "train_score", "test_score",
        "solve_subtree_calls", "feature_branch_expansions",
        "local_equivalent_features_skipped",
    )
    for key in structural_keys:
        if any(row[key] != rows[0][key] for row in rows[1:]):
            raise ProtocolError(f"non-deterministic metric {key} for {dataset}")
    answer = {key: rows[0][key] for key in structural_keys}
    answer.update({
        "reported_total_seconds": statistics.median(
            row["reported_total_seconds"] for row in rows
        ),
        "solve_cpu_seconds": statistics.median(
            row["solve_cpu_seconds"] for row in rows
        ),
        "repetitions": rows,
    })
    return answer


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
        row["rank"]: execute(binary, row["dataset_path"], None, repetitions)
        for row in datasets
    }


def benchmark_policy(
    binary: Path,
    datasets: list[dict[str, object]],
    originals: dict[int, dict[str, object]],
    policy: tuple[int, int],
    repetitions: int,
) -> dict[str, object]:
    rows = []
    for dataset in datasets:
        original = originals[dataset["rank"]]
        patched = execute(binary, dataset["dataset_path"], policy, repetitions)
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
        "probe_features": policy[0],
        "probe_duplicates": policy[1],
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


def select_policy(candidates: list[dict[str, object]]) -> dict[str, object]:
    admissible = [
        row for row in candidates
        if row["optimum_match_count"] == row["row_count"]
        and row["patched_over_original_runtime_max"] <= 1.10
        and row["tasks_with_local_skips"] >= 4
        and row["branch_expansion_reduction_fraction"] > 0
    ]
    if not admissible:
        return {
            "selected": False,
            "reason": "no probe policy met the locked admissibility rule",
        }
    best = max(admissible, key=lambda row: (
        row["reported_speedup_median"],
        row["branch_expansion_reduction_fraction"],
        -row["probe_features"],
        -row["probe_duplicates"],
    ))
    return {
        "selected": True,
        "probe_features": best["probe_features"],
        "probe_duplicates": best["probe_duplicates"],
        "training_speedup_median": best["reported_speedup_median"],
        "training_branch_reduction": best[
            "branch_expansion_reduction_fraction"
        ],
        "training_max_runtime_ratio": best[
            "patched_over_original_runtime_max"
        ],
    }


def protocol() -> dict[str, object]:
    return {
        "training_ranks": list(TRAIN_RANKS),
        "validation_ranks": list(VALIDATION_RANKS),
        "policy_candidates": [list(policy) for policy in POLICY_CANDIDATES],
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


def run(source: Path, work: Path) -> dict[str, object]:
    actual_commit = v50.run_command(
        ["git", "rev-parse", "HEAD"], cwd=source
    ).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        raise ProtocolError(f"source revision {actual_commit} != {PINNED_COMMIT}")
    work.mkdir(parents=True, exist_ok=True)
    original_root = work / "pystreed-original"
    patched_root = work / "pystreed-probe-activated"
    prepare_source(source, original_root, False)
    prepare_source(source, patched_root, True)
    original_binary = v50.build_solver(original_root)
    patched_binary = v50.build_solver(patched_root)

    training_datasets = write_ranked_datasets(
        source, work, TRAIN_RANKS, "training"
    )
    training_originals = baseline_results(
        original_binary, training_datasets, SELECTION_REPETITIONS
    )
    candidates = [
        benchmark_policy(
            patched_binary,
            training_datasets,
            training_originals,
            policy,
            SELECTION_REPETITIONS,
        )
        for policy in POLICY_CANDIDATES
    ]
    selection = select_policy(candidates)
    if not selection["selected"]:
        return {
            "status": "not_yet",
            "development_gate": False,
            "pinned_commit": PINNED_COMMIT,
            "protocol": protocol(),
            "selection": selection,
            "training_candidates": candidates,
        }

    policy = (
        int(selection["probe_features"]),
        int(selection["probe_duplicates"]),
    )
    validation_datasets = write_ranked_datasets(
        source, work, VALIDATION_RANKS, "validation"
    )
    validation_originals = baseline_results(
        original_binary, validation_datasets, VALIDATION_REPETITIONS
    )
    validation = benchmark_policy(
        patched_binary,
        validation_datasets,
        validation_originals,
        policy,
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
        and validation["patched_over_original_runtime_max"] <= 1.08
    )
    frozen_digest = hashlib.sha256(json.dumps({
        "pinned_commit": PINNED_COMMIT,
        "policy_candidates": POLICY_CANDIDATES,
        "training_ranks": TRAIN_RANKS,
        "validation_ranks": VALIDATION_RANKS,
        "selection_rule": (
            "max_median_speedup_then_branch_reduction_then_shorter_probe"
        ),
        "selected_policy": policy,
        "protocol": protocol(),
    }, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": (
            "probe_activated_quotient_candidate" if gate else "not_yet"
        ),
        "development_gate": gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "claim_scope": (
            "PySTreeD Accuracy only. Each descendant probes a fixed number of "
            "valid features using per-label count buckets and exact ADataView "
            "verification. Full-state checking continues only if the probe finds "
            "the selected number of exact duplicate partitions. Ranks 0-23 are "
            "opened development data; no external holdout is claimed here."
        ),
        "protocol": protocol(),
        "selection": selection,
        "selected_policy": list(policy),
        "training_candidates": candidates,
        "validation": validation,
        "frozen_probe_digest": frozen_digest,
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
        "validation_speedup": report.get("validation", {}).get(
            "reported_speedup_median"
        ),
        "validation_branch_reduction": report.get("validation", {}).get(
            "branch_expansion_reduction_fraction"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
