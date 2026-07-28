from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


PINNED_COMMIT = "9ad41626a1f26c4b7481e8360c5c8b1871e10d96"
DATASET_COUNT = 4
MAX_ROWS = 192
MAX_NATIVE_FEATURES = 28
AUGMENTED_FEATURES = 40
MAX_DEPTH = 3
MAX_NODES = 5
TIME_LIMIT = 180
RANDOM_SEED = 5050


class ProtocolError(RuntimeError):
    pass


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout: int = 1200,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ProtocolError(f"{label}: expected one replacement, found {count}")
    return text.replace(old, new, 1)


def instrument_statistics(root: Path) -> None:
    path = root / "include" / "solver" / "statistics.h"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "\t\t\tnum_cache_hit_optimality = 0;\n\t\t}",
        "\t\t\tnum_cache_hit_optimality = 0;\n"
        "\t\t\tnum_solve_subtree_calls = 0;\n"
        "\t\t\tnum_feature_branch_expansions = 0;\n"
        "\t\t\tnum_local_equivalent_features_skipped = 0;\n"
        "\t\t}",
        "statistics constructor",
    )
    text = replace_once(
        text,
        "\t\t\tstd::cout << \"\\tTerminal 3 node: \" << num_terminal_nodes_with_node_budget_three << std::endl;\n\t\t}",
        "\t\t\tstd::cout << \"\\tTerminal 3 node: \" << num_terminal_nodes_with_node_budget_three << std::endl;\n"
        "\t\t\tstd::cout << \"SolveSubTree calls: \" << num_solve_subtree_calls << std::endl;\n"
        "\t\t\tstd::cout << \"Feature branch expansions: \" << num_feature_branch_expansions << std::endl;\n"
        "\t\t\tstd::cout << \"Local equivalent features skipped: \" << num_local_equivalent_features_skipped << std::endl;\n"
        "\t\t}",
        "statistics print",
    )
    text = replace_once(
        text,
        "\t\tsize_t num_cache_hit_nonzero_bound;\n\n\n\t\tdouble total_time;",
        "\t\tsize_t num_cache_hit_nonzero_bound;\n"
        "\t\tsize_t num_solve_subtree_calls;\n"
        "\t\tsize_t num_feature_branch_expansions;\n"
        "\t\tsize_t num_local_equivalent_features_skipped;\n\n\n"
        "\t\tdouble total_time;",
        "statistics fields",
    )
    path.write_text(text, encoding="utf-8")


def instrument_solver(root: Path, enable_local_quotient: bool) -> None:
    path = root / "src" / "solver" / "solver.cpp"
    text = path.read_text(encoding="utf-8-sig")
    text = replace_once(
        text,
        "\ttypename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int org_max_depth, int org_num_nodes) {\n\t\tint max_depth = org_max_depth, num_nodes = org_num_nodes;",
        "\ttypename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int org_max_depth, int org_num_nodes) {\n"
        "\t\tstats.num_solve_subtree_calls++;\n"
        "\t\tint max_depth = org_max_depth, num_nodes = org_num_nodes;",
        "SolveSubTree counter",
    )
    if enable_local_quotient:
        text = replace_once(
            text,
            "\t\tfeature_selector->Initialize(data);\n\n\t\t// Loop over each feature",
            "\t\tfeature_selector->Initialize(data);\n"
            "\t\tstd::vector<std::pair<ADataViewBitSet, ADataViewBitSet>> local_accuracy_partitions;\n\n"
            "\t\t// Loop over each feature",
            "local partition archive",
        )
        text = replace_once(
            text,
            "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n\t\t\t// Generate the context descriptors",
            "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n"
            "\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n"
            "\t\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
            "\t\t\t\tADataViewBitSet left_partition(left_data);\n"
            "\t\t\t\tADataViewBitSet right_partition(right_data);\n"
            "\t\t\t\tbool equivalent_partition = false;\n"
            "\t\t\t\tfor (const auto& previous : local_accuracy_partitions) {\n"
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
            "\t\t\t\tlocal_accuracy_partitions.emplace_back(left_partition, right_partition);\n"
            "\t\t\t}\n"
            "\t\t\tstats.num_feature_branch_expansions++;\n\n"
            "\t\t\t// Generate the context descriptors",
            "local quotient insertion",
        )
    else:
        text = replace_once(
            text,
            "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n\t\t\t// Generate the context descriptors",
            "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n"
            "\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n"
            "\t\t\tstats.num_feature_branch_expansions++;\n\n"
            "\t\t\t// Generate the context descriptors",
            "baseline branch counter",
        )
    path.write_text(text, encoding="utf-8")


def prepare_source(source: Path, destination: Path, local: bool) -> None:
    shutil.copytree(source, destination)
    instrument_statistics(destination)
    instrument_solver(destination, local)


def build_solver(root: Path) -> Path:
    build = root / "build-v50"
    run_command([
        "cmake", "-S", str(root), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release",
    ], timeout=1200)
    run_command([
        "cmake", "--build", str(build), "--config", "Release", "-j", "2"
    ], timeout=1800)
    candidates = [
        path for path in build.rglob("*")
        if path.is_file() and path.name.lower() == "streed"
    ]
    if not candidates:
        raise ProtocolError("built STreeD executable not found")
    candidates.sort(key=lambda path: (len(path.parts), str(path)))
    return candidates[0]


def parse_table(path: Path) -> tuple[list[int], list[list[int]]] | None:
    rows = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        fields = [value for value in re.split(r"[\s,;]+", raw) if value]
        try:
            values = [int(value) for value in fields]
        except ValueError:
            return None
        rows.append(values)
    if len(rows) < 60:
        return None
    width = len(rows[0])
    if width < 11 or width > 161 or any(len(row) != width for row in rows):
        return None
    labels = [row[0] for row in rows]
    features = [row[1:] for row in rows]
    if len(set(labels)) < 2 or len(set(labels)) > 12:
        return None
    if any(value not in (0, 1) for row in features for value in row):
        return None
    label_map = {value: index for index, value in enumerate(sorted(set(labels)))}
    return [label_map[value] for value in labels], features


def gini_gain(labels: list[int], column: list[int]) -> float:
    def impurity(values: list[int]) -> float:
        if not values:
            return 0.0
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        total = len(values)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    root = impurity(labels)
    total = len(labels)
    weighted = 0.0
    for bit in (0, 1):
        child = [label for label, value in zip(labels, column) if value == bit]
        weighted += (len(child) / total) * impurity(child)
    return root - weighted


def write_dataset(path: Path, labels: list[int], columns: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, label in enumerate(labels):
            values = [label] + [column[row_index] for column in columns]
            handle.write(" ".join(map(str, values)) + "\n")


def make_augmented(
    labels: list[int], features: list[list[int]]
) -> tuple[list[list[int]], dict[str, object]]:
    columns = [list(values) for values in zip(*features)]
    ranked = sorted(
        range(len(columns)),
        key=lambda index: (-gini_gain(labels, columns[index]), index),
    )
    base_indices = ranked[: min(16, len(ranked))]
    base_columns = [columns[index] for index in base_indices]
    anchors = list(range(min(2, len(base_columns))))
    partners = list(range(2, min(8, len(base_columns))))
    augmented = list(base_columns)
    existing = {tuple(column) for column in augmented}
    existing |= {tuple(1 - value for value in column) for column in augmented}
    derived = []
    for anchor_index in anchors:
        anchor = base_columns[anchor_index]
        for partner_index in partners:
            partner = base_columns[partner_index]
            for operation in ("and", "or"):
                if operation == "and":
                    column = [left & right for left, right in zip(anchor, partner)]
                else:
                    column = [left | right for left, right in zip(anchor, partner)]
                signature = tuple(column)
                complement = tuple(1 - value for value in column)
                if len(set(column)) <= 1 or signature in existing or complement in existing:
                    continue
                existing.add(signature)
                existing.add(complement)
                augmented.append(column)
                derived.append({
                    "anchor_position": anchor_index,
                    "partner_position": partner_index,
                    "operation": operation,
                })
                if len(augmented) >= AUGMENTED_FEATURES:
                    break
            if len(augmented) >= AUGMENTED_FEATURES:
                break
        if len(augmented) >= AUGMENTED_FEATURES:
            break
    return augmented, {
        "base_feature_indices": base_indices,
        "derived_feature_count": len(derived),
        "derived_features": derived,
    }


def select_datasets(source: Path, output: Path) -> list[dict[str, object]]:
    candidates = []
    for path in sorted((source / "data").rglob("*.csv")):
        parsed = parse_table(path)
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
    if len(candidates) < DATASET_COUNT:
        raise ProtocolError(
            f"only {len(candidates)} eligible pinned datasets found"
        )
    selected = []
    for index, (_, relative, path, labels, features) in enumerate(
        candidates[:DATASET_COUNT]
    ):
        labels = labels[:MAX_ROWS]
        features = features[:MAX_ROWS]
        native_columns = [
            list(values) for values in zip(*features)
        ][:MAX_NATIVE_FEATURES]
        native_path = output / "datasets" / f"dataset-{index}-native.csv"
        write_dataset(native_path, labels, native_columns)
        augmented_columns, augmentation = make_augmented(labels, features)
        augmented_path = output / "datasets" / f"dataset-{index}-augmented.csv"
        write_dataset(augmented_path, labels, augmented_columns)
        selected.append({
            "index": index,
            "source_path": relative,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source_rows": len(parse_table(path)[0]),
            "used_rows": len(labels),
            "source_features": len(features[0]),
            "native_features": len(native_columns),
            "augmented_features": len(augmented_columns),
            "label_count": len(set(labels)),
            "native_path": native_path,
            "augmented_path": augmented_path,
            "augmentation": augmentation,
        })
    return selected


SOLUTION_RE = re.compile(
    r"Solution 0:\s+(\d+)\s+(\d+)\s+([0-9eE+\-.]+)\s+([0-9eE+\-.]+)"
)


def metric(text: str, label: str, integer: bool = True):
    match = re.search(re.escape(label) + r"\s*([0-9eE+\-.]+)", text)
    if not match:
        raise ProtocolError(f"missing metric {label!r}")
    return int(float(match.group(1))) if integer else float(match.group(1))


def execute(binary: Path, dataset: Path) -> dict[str, object]:
    command = [
        str(binary),
        "-task", "accuracy",
        "-file", str(dataset),
        "-max-depth", str(MAX_DEPTH),
        "-max-num-nodes", str(MAX_NODES),
        "-max-num-features", str(AUGMENTED_FEATURES),
        "-num-instances", str(MAX_ROWS),
        "-use-terminal-solver", "0",
        "-use-similarity-lower-bound", "0",
        "-feature-ordering", "in-order",
        "-random-seed", str(RANDOM_SEED),
        "-time", str(TIME_LIMIT),
        "-verbose", "1",
    ]
    start = time.perf_counter()
    result = run_command(command, timeout=TIME_LIMIT + 60)
    wall = time.perf_counter() - start
    text = result.stdout
    if "Warning: No proof of optimality" in text or "No tree found" in text:
        raise ProtocolError(f"solver did not prove optimum for {dataset}")
    solution = SOLUTION_RE.search(text)
    if not solution:
        raise ProtocolError(f"could not parse solution for {dataset}\n{text[-2000:]}")
    return {
        "depth": int(solution.group(1)),
        "nodes": int(solution.group(2)),
        "train_score": float(solution.group(3)),
        "test_score": float(solution.group(4)),
        "reported_total_seconds": metric(text, "Total time elapsed:", False),
        "solve_cpu_seconds": metric(text, "CLOCKS FOR SOLVE:", False),
        "wall_seconds": wall,
        "solve_subtree_calls": metric(text, "SolveSubTree calls:"),
        "feature_branch_expansions": metric(text, "Feature branch expansions:"),
        "local_equivalent_features_skipped": metric(
            text, "Local equivalent features skipped:"
        ),
        "stdout_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def same_optimum(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        left["depth"] == right["depth"]
        and left["nodes"] == right["nodes"]
        and abs(left["train_score"] - right["train_score"]) <= 1e-12
        and abs(left["test_score"] - right["test_score"]) <= 1e-12
    )


def run(source: Path, work: Path) -> dict[str, object]:
    actual_commit = run_command(
        ["git", "rev-parse", "HEAD"], cwd=source
    ).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        raise ProtocolError(
            f"source revision {actual_commit} != pinned {PINNED_COMMIT}"
        )
    work.mkdir(parents=True, exist_ok=True)
    original_root = work / "pystreed-original"
    patched_root = work / "pystreed-local-quotient"
    prepare_source(source, original_root, False)
    prepare_source(source, patched_root, True)
    original_binary = build_solver(original_root)
    patched_binary = build_solver(patched_root)
    datasets = select_datasets(source, work)
    rows = []
    for dataset in datasets:
        for variant in ("native", "augmented"):
            path = dataset[f"{variant}_path"]
            original = execute(original_binary, path)
            patched = execute(patched_binary, path)
            rows.append({
                "dataset_index": dataset["index"],
                "source_path": dataset["source_path"],
                "variant": variant,
                "features": (
                    dataset["native_features"]
                    if variant == "native"
                    else dataset["augmented_features"]
                ),
                "optimum_matched": same_optimum(original, patched),
                "original": original,
                "patched": patched,
                "subtree_call_reduction": (
                    original["solve_subtree_calls"]
                    - patched["solve_subtree_calls"]
                ),
                "branch_expansion_reduction": (
                    original["feature_branch_expansions"]
                    - patched["feature_branch_expansions"]
                ),
                "reported_speedup": (
                    original["reported_total_seconds"]
                    / max(1e-9, patched["reported_total_seconds"])
                ),
            })
    augmented = [row for row in rows if row["variant"] == "augmented"]
    native = [row for row in rows if row["variant"] == "native"]
    original_aug_calls = sum(
        row["original"]["solve_subtree_calls"] for row in augmented
    )
    patched_aug_calls = sum(
        row["patched"]["solve_subtree_calls"] for row in augmented
    )
    original_aug_expansions = sum(
        row["original"]["feature_branch_expansions"] for row in augmented
    )
    patched_aug_expansions = sum(
        row["patched"]["feature_branch_expansions"] for row in augmented
    )
    total_skipped = sum(
        row["patched"]["local_equivalent_features_skipped"]
        for row in augmented
    )
    native_runtime_ratios = [
        row["patched"]["reported_total_seconds"]
        / max(1e-9, row["original"]["reported_total_seconds"])
        for row in native
    ]
    augmented_speedups = [row["reported_speedup"] for row in augmented]
    gate = (
        actual_commit == PINNED_COMMIT
        and len(datasets) == DATASET_COUNT
        and len(rows) == DATASET_COUNT * 2
        and all(row["optimum_matched"] for row in rows)
        and all(
            row["patched"]["local_equivalent_features_skipped"] > 0
            for row in augmented
        )
        and total_skipped >= 100
        and patched_aug_calls <= 0.90 * original_aug_calls
        and patched_aug_expansions <= 0.80 * original_aug_expansions
        and statistics.median(augmented_speedups) >= 1.05
        and max(native_runtime_ratios) <= 1.75
    )
    return {
        "status": (
            "pystreed_integration_candidate" if gate else "not_yet"
        ),
        "external_integration_gate": gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "scope": (
            "PySTreeD Accuracy only: constant feature identity cost, no task "
            "constraints, binary splits. Complementary left/right partitions "
            "are treated as the same unlabeled partition. Other PySTreeD tasks "
            "are not claimed safe by this patch."
        ),
        "protocol": {
            "dataset_selection": (
                "eligible binary CSV files under pinned data tree ranked by "
                "SHA256(relative path), first four"
            ),
            "dataset_count": DATASET_COUNT,
            "max_rows": MAX_ROWS,
            "max_native_features": MAX_NATIVE_FEATURES,
            "augmented_feature_limit": AUGMENTED_FEATURES,
            "max_depth": MAX_DEPTH,
            "max_nodes": MAX_NODES,
            "terminal_solver": False,
            "similarity_lower_bound": False,
            "feature_ordering": "in-order",
            "random_seed": RANDOM_SEED,
        },
        "datasets": [
            {key: value for key, value in row.items() if not key.endswith("_path")}
            for row in datasets
        ],
        "row_count": len(rows),
        "optimum_match_count": sum(
            int(row["optimum_matched"]) for row in rows
        ),
        "augmented_total_local_skips": total_skipped,
        "augmented_original_subtree_calls": original_aug_calls,
        "augmented_patched_subtree_calls": patched_aug_calls,
        "augmented_subtree_call_reduction_fraction": (
            1.0 - patched_aug_calls / max(1, original_aug_calls)
        ),
        "augmented_original_branch_expansions": original_aug_expansions,
        "augmented_patched_branch_expansions": patched_aug_expansions,
        "augmented_branch_expansion_reduction_fraction": (
            1.0 - patched_aug_expansions / max(1, original_aug_expansions)
        ),
        "augmented_reported_speedup_median": statistics.median(
            augmented_speedups
        ),
        "native_patched_over_original_runtime_max": max(
            native_runtime_ratios
        ),
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
    args.output.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "optimum_matches": report["optimum_match_count"],
        "local_skips": report["augmented_total_local_skips"],
        "subtree_reduction": report[
            "augmented_subtree_call_reduction_fraction"
        ],
        "branch_reduction": report[
            "augmented_branch_expansion_reduction_fraction"
        ],
        "median_speedup": report[
            "augmented_reported_speedup_median"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
