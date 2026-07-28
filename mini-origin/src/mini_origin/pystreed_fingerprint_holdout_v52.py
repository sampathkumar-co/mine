from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from . import pystreed_integration_v50 as v50
from . import pystreed_native_holdout_v51 as v51


PINNED_COMMIT = v50.PINNED_COMMIT
DEVELOPMENT_COUNT = 12
HOLDOUT_COUNT = 12
HOLDOUT_START = DEVELOPMENT_COUNT


class ProtocolError(RuntimeError):
    pass


def patch_data_header(root: Path) -> None:
    path = root / "include" / "model" / "data.h"
    text = path.read_text(encoding="utf-8")
    text = v50.replace_once(
        text,
        "\tclass ADataView;\n",
        "\tstruct DataSplitFingerprint {\n"
        "\t\tsize_t left_size{0}, right_size{0};\n"
        "\t\tuint64_t left_xor{0}, right_xor{0};\n"
        "\t\tuint64_t left_sum{0}, right_sum{0};\n\n"
        "\t\tstatic uint64_t Token(int id) {\n"
        "\t\t\tconst uint64_t value = static_cast<uint64_t>(id) + 1ULL;\n"
        "\t\t\treturn value * 0x9e3779b97f4a7c15ULL;\n"
        "\t\t}\n"
        "\t\tvoid AddLeft(int id) {\n"
        "\t\t\tconst uint64_t token = Token(id);\n"
        "\t\t\tleft_size++; left_xor ^= token; left_sum += token ^ (token >> 29);\n"
        "\t\t}\n"
        "\t\tvoid AddRight(int id) {\n"
        "\t\t\tconst uint64_t token = Token(id);\n"
        "\t\t\tright_size++; right_xor ^= token; right_sum += token ^ (token >> 29);\n"
        "\t\t}\n"
        "\t\tvoid Canonicalize() {\n"
        "\t\t\tconst bool swap_sides =\n"
        "\t\t\t\tleft_size > right_size\n"
        "\t\t\t\t|| (left_size == right_size && left_xor > right_xor)\n"
        "\t\t\t\t|| (left_size == right_size && left_xor == right_xor && left_sum > right_sum);\n"
        "\t\t\tif (swap_sides) {\n"
        "\t\t\t\tstd::swap(left_size, right_size);\n"
        "\t\t\t\tstd::swap(left_xor, right_xor);\n"
        "\t\t\t\tstd::swap(left_sum, right_sum);\n"
        "\t\t\t}\n"
        "\t\t}\n"
        "\t\tsize_t Hash() const {\n"
        "\t\t\tsize_t seed = left_size;\n"
        "\t\t\tseed ^= static_cast<size_t>(left_xor) + 0x9e3779b9 + (seed << 6) + (seed >> 2);\n"
        "\t\t\tseed ^= static_cast<size_t>(left_sum) + 0x9e3779b9 + (seed << 6) + (seed >> 2);\n"
        "\t\t\tseed ^= right_size + 0x9e3779b9 + (seed << 6) + (seed >> 2);\n"
        "\t\t\tseed ^= static_cast<size_t>(right_xor) + 0x9e3779b9 + (seed << 6) + (seed >> 2);\n"
        "\t\t\tseed ^= static_cast<size_t>(right_sum) + 0x9e3779b9 + (seed << 6) + (seed >> 2);\n"
        "\t\t\treturn seed;\n"
        "\t\t}\n"
        "\t};\n\n"
        "\tclass ADataView;\n",
        "fingerprint struct",
    )
    text = v50.replace_once(
        text,
        "\t\tvoid SplitData(int feature, ADataView& left, ADataView& right) const;",
        "\t\tvoid SplitData(int feature, ADataView& left, ADataView& right, DataSplitFingerprint* fingerprint = nullptr) const;",
        "SplitData declaration",
    )
    path.write_text(text, encoding="utf-8")


def patch_data_source(root: Path) -> None:
    path = root / "src" / "model" / "data.cpp"
    text = path.read_text(encoding="utf-8")
    text = v50.replace_once(
        text,
        "\tvoid ADataView::SplitData(int feature, ADataView& left, ADataView& right) const {\n\t\tleft.data = data;",
        "\tvoid ADataView::SplitData(int feature, ADataView& left, ADataView& right, DataSplitFingerprint* fingerprint) const {\n"
        "\t\tif (fingerprint != nullptr) *fingerprint = DataSplitFingerprint();\n"
        "\t\tleft.data = data;",
        "SplitData definition",
    )
    text = v50.replace_once(
        text,
        "\t\t\t\tif (instance->IsFeaturePresent(feature)) {\n\t\t\t\t\tright.instances[label].push_back(instance);\n\t\t\t\t} else {\n\t\t\t\t\tleft.instances[label].push_back(instance);\n\t\t\t\t}",
        "\t\t\t\tif (instance->IsFeaturePresent(feature)) {\n"
        "\t\t\t\t\tright.instances[label].push_back(instance);\n"
        "\t\t\t\t\tif (fingerprint != nullptr) fingerprint->AddRight(instance->GetID());\n"
        "\t\t\t\t} else {\n"
        "\t\t\t\t\tleft.instances[label].push_back(instance);\n"
        "\t\t\t\t\tif (fingerprint != nullptr) fingerprint->AddLeft(instance->GetID());\n"
        "\t\t\t\t}",
        "fingerprint accumulation",
    )
    text = v50.replace_once(
        text,
        "\t\truntime_assert(left.size + right.size == size);\n\t}",
        "\t\truntime_assert(left.size + right.size == size);\n"
        "\t\tif (fingerprint != nullptr) fingerprint->Canonicalize();\n"
        "\t}",
        "fingerprint canonicalization",
    )
    path.write_text(text, encoding="utf-8")


def patch_splitter_header(root: Path) -> None:
    path = root / "include" / "solver" / "data_splitter.h"
    text = path.read_text(encoding="utf-8")
    text = v50.replace_once(
        text,
        "\t\tvoid Split(const ADataView& data, const Branch& branch, int feature, ADataView& left, ADataView& right, bool test=false);",
        "\t\tvoid Split(const ADataView& data, const Branch& branch, int feature, ADataView& left, ADataView& right, bool test=false, DataSplitFingerprint* fingerprint=nullptr);",
        "DataSplitter declaration",
    )
    path.write_text(text, encoding="utf-8")


def patch_splitter_source(root: Path) -> None:
    path = root / "src" / "solver" / "data_splitter.cpp"
    text = path.read_text(encoding="utf-8")
    text = v50.replace_once(
        text,
        "\tvoid DataSplitter::Split(const ADataView& data, const Branch& branch, int feature, ADataView& left, ADataView& right, bool test) {",
        "\tvoid DataSplitter::Split(const ADataView& data, const Branch& branch, int feature, ADataView& left, ADataView& right, bool test, DataSplitFingerprint* fingerprint) {",
        "DataSplitter definition",
    )
    text = v50.replace_once(
        text,
        "\t\t\tif (iter == hashmap.end()) {\n\t\t\t\tdata.SplitData(feature, left, right);",
        "\t\t\tif (iter == hashmap.end()) {\n"
        "\t\t\t\tdata.SplitData(feature, left, right, fingerprint);",
        "cache miss fingerprint",
    )
    text = v50.replace_once(
        text,
        "\t\t\tleft = iter->second.first;\n\t\t\tright = iter->second.second;\n\t\t} else {\n\t\t\tdata.SplitData(feature, left, right);",
        "\t\t\tleft = iter->second.first;\n"
        "\t\t\tright = iter->second.second;\n"
        "\t\t\tif (fingerprint != nullptr) {\n"
        "\t\t\t\t*fingerprint = DataSplitFingerprint();\n"
        "\t\t\t\tfor (int label = 0; label < left.NumLabels(); label++) {\n"
        "\t\t\t\t\tfor (auto instance : left.GetInstancesForLabel(label)) fingerprint->AddLeft(instance->GetID());\n"
        "\t\t\t\t\tfor (auto instance : right.GetInstancesForLabel(label)) fingerprint->AddRight(instance->GetID());\n"
        "\t\t\t\t}\n"
        "\t\t\t\tfingerprint->Canonicalize();\n"
        "\t\t\t}\n"
        "\t\t} else {\n"
        "\t\t\tdata.SplitData(feature, left, right, fingerprint);",
        "cache hit and disabled fingerprint",
    )
    path.write_text(text, encoding="utf-8")


def patch_solver(root: Path) -> None:
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
        "\t\tstd::unordered_map<size_t, std::vector<int>> local_accuracy_partition_buckets;\n\n"
        "\t\t// Loop over each feature",
        "fingerprint buckets",
    )
    text = v50.replace_once(
        text,
        "\t\t\tADataView left_data;\n\t\t\tADataView right_data;\n\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n\t\t\t// Generate the context descriptors",
        "\t\t\tADataView left_data;\n"
        "\t\t\tADataView right_data;\n"
        "\t\t\tDataSplitFingerprint split_fingerprint;\n"
        "\t\t\tDataSplitFingerprint* fingerprint_ptr = nullptr;\n"
        "\t\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\t\tif (branch.Depth() > 0) fingerprint_ptr = &split_fingerprint;\n"
        "\t\t\t}\n"
        "\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data, false, fingerprint_ptr);\n"
        "\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n"
        "\t\t\tif constexpr (std::is_same<OT, Accuracy>::value) {\n"
        "\t\t\t\tif (branch.Depth() > 0) {\n"
        "\t\t\t\t\tauto& bucket = local_accuracy_partition_buckets[split_fingerprint.Hash()];\n"
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
        "inline fingerprint quotient",
    )
    path.write_text(text, encoding="utf-8")


def prepare_source(source: Path, destination: Path, local: bool) -> None:
    shutil.copytree(source, destination)
    v50.instrument_statistics(destination)
    if not local:
        v50.instrument_solver(destination, False)
        return
    patch_data_header(destination)
    patch_data_source(destination)
    patch_splitter_header(destination)
    patch_splitter_source(destination)
    patch_solver(destination)


def write_suite(source: Path, output: Path):
    candidates = v51.eligible_datasets(source)
    required = DEVELOPMENT_COUNT + HOLDOUT_COUNT
    if len(candidates) < required:
        raise ProtocolError(
            f"only {len(candidates)} eligible datasets; need {required}"
        )
    rows = []
    for rank, (_, relative, path, labels, features) in enumerate(
        candidates[:required]
    ):
        labels = labels[:v51.MAX_ROWS]
        features = features[:v51.MAX_ROWS]
        columns = [list(values) for values in zip(*features)][:v51.MAX_FEATURES]
        split = "development" if rank < DEVELOPMENT_COUNT else "holdout"
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
    return (
        [row for row in rows if row["split"] == "development"],
        [row for row in rows if row["split"] == "holdout"],
    )


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
    patched_root = work / "pystreed-inline-fingerprint"
    prepare_source(source, original_root, False)
    prepare_source(source, patched_root, True)
    original_binary = v50.build_solver(original_root)
    patched_binary = v50.build_solver(patched_root)
    development_datasets, holdout_datasets = write_suite(source, work)
    development_rows = v51.benchmark_rows(
        development_datasets, original_binary, patched_binary
    )
    development = v51.aggregate(development_rows)
    development_gate = (
        development["row_count"] == DEVELOPMENT_COUNT
        and development["optimum_match_count"] == DEVELOPMENT_COUNT
        and development["patched_over_original_runtime_median"] <= 1.02
        and development["patched_over_original_runtime_max"] <= 1.15
    )
    holdout_rows = v51.benchmark_rows(
        holdout_datasets, original_binary, patched_binary
    )
    holdout = v51.aggregate(holdout_rows)
    gate = (
        development_gate
        and holdout["row_count"] == HOLDOUT_COUNT
        and holdout["optimum_match_count"] == HOLDOUT_COUNT
        and holdout["tasks_with_local_skips"] >= 9
        and holdout["subtree_call_reduction_fraction"] >= 0.03
        and holdout["branch_expansion_reduction_fraction"] >= 0.03
        and holdout["reported_speedup_median"] >= 1.0
        and holdout["tasks_speedup_ge_1_05"] >= 5
        and holdout["patched_over_original_runtime_max"] <= 1.15
    )
    return {
        "status": (
            "pystreed_inline_fingerprint_candidate" if gate else "not_yet"
        ),
        "inline_fingerprint_gate": gate,
        "development_gate": development_gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "scope": (
            "PySTreeD Accuracy only. Allocation-free fingerprints are accumulated "
            "inside the existing data split loop below the root. Fingerprint "
            "matches are verified by exact ADataView equality before pruning. "
            "Ranks 0-11 are opened development data; ranks 12-23 are a fresh "
            "native holdout under the previously fixed path-hash order."
        ),
        "protocol": {
            "development_ranks": [0, DEVELOPMENT_COUNT - 1],
            "holdout_ranks": [HOLDOUT_START, HOLDOUT_START + HOLDOUT_COUNT - 1],
            "max_rows": v51.MAX_ROWS,
            "max_features": v51.MAX_FEATURES,
            "max_depth": v51.MAX_DEPTH,
            "max_nodes": v51.MAX_NODES,
            "repetitions": v51.REPETITIONS,
            "random_seed": v51.RANDOM_SEED,
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
        "holdout_matches": report["holdout_summary"]["optimum_match_count"],
        "holdout_skips": report["holdout_summary"]["total_local_skips"],
        "holdout_branch_reduction": report["holdout_summary"]["branch_expansion_reduction_fraction"],
        "holdout_median_speedup": report["holdout_summary"]["reported_speedup_median"],
    }, indent=2))


if __name__ == "__main__":
    main()
