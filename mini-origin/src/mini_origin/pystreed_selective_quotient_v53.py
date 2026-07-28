from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from . import pystreed_integration_v50 as v50
from . import pystreed_native_holdout_v51 as v51
from . import pystreed_fingerprint_holdout_v52 as v52

PINNED_COMMIT = v50.PINNED_COMMIT
ACTIVE_INSTANCE_LIMIT = 64
DEVELOPMENT_COUNT = 24
HOLDOUT_COUNT = 12
HOLDOUT_START = DEVELOPMENT_COUNT


class ProtocolError(RuntimeError):
    pass


def patch_solver_selective(root: Path) -> None:
    v52.patch_solver(root)
    path = root / "src" / "solver" / "solver.cpp"
    text = path.read_text(encoding="utf-8")
    old = "if (branch.Depth() > 0)"
    new = f"if (branch.Depth() > 0 && data.Size() <= {ACTIVE_INSTANCE_LIMIT})"
    count = text.count(old)
    if count != 2:
        raise ProtocolError(f"expected two activation sites, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def prepare_source(source: Path, destination: Path, local: bool) -> None:
    shutil.copytree(source, destination)
    v50.instrument_statistics(destination)
    if not local:
        v50.instrument_solver(destination, False)
        return
    v52.patch_data_header(destination)
    v52.patch_data_source(destination)
    v52.patch_splitter_header(destination)
    v52.patch_splitter_source(destination)
    patch_solver_selective(destination)


def write_suite(source: Path, output: Path):
    candidates = v51.eligible_datasets(source)
    required = DEVELOPMENT_COUNT + HOLDOUT_COUNT
    if len(candidates) < required:
        raise ProtocolError(f"only {len(candidates)} eligible datasets; need {required}")
    rows = []
    for rank, (_, relative, path, labels, features) in enumerate(candidates[:required]):
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
    actual_commit = v50.run_command(["git", "rev-parse", "HEAD"], cwd=source).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        raise ProtocolError(f"source revision {actual_commit} != {PINNED_COMMIT}")
    work.mkdir(parents=True, exist_ok=True)
    original_root = work / "pystreed-original"
    patched_root = work / "pystreed-selective"
    prepare_source(source, original_root, False)
    prepare_source(source, patched_root, True)
    original_binary = v50.build_solver(original_root)
    patched_binary = v50.build_solver(patched_root)
    development_datasets, holdout_datasets = write_suite(source, work)
    development_rows = v51.benchmark_rows(development_datasets, original_binary, patched_binary)
    development = v51.aggregate(development_rows)
    development_gate = (
        development["row_count"] == DEVELOPMENT_COUNT
        and development["optimum_match_count"] == DEVELOPMENT_COUNT
        and development["patched_over_original_runtime_median"] <= 1.02
        and development["patched_over_original_runtime_max"] <= 1.12
    )
    holdout_rows = v51.benchmark_rows(holdout_datasets, original_binary, patched_binary)
    holdout = v51.aggregate(holdout_rows)
    gate = (
        development_gate
        and holdout["row_count"] == HOLDOUT_COUNT
        and holdout["optimum_match_count"] == HOLDOUT_COUNT
        and holdout["tasks_with_local_skips"] >= 6
        and holdout["subtree_call_reduction_fraction"] >= 0.01
        and holdout["branch_expansion_reduction_fraction"] >= 0.01
        and holdout["reported_speedup_median"] >= 1.0
        and holdout["tasks_speedup_ge_1_05"] >= 3
        and holdout["patched_over_original_runtime_max"] <= 1.10
    )
    return {
        "status": "pystreed_selective_quotient_candidate" if gate else "not_yet",
        "selective_quotient_gate": gate,
        "development_gate": development_gate,
        "pinned_repository": "AlgTUDelft/pystreed",
        "pinned_commit": PINNED_COMMIT,
        "activation_rule": {"minimum_depth": 1, "maximum_active_instances": ACTIVE_INSTANCE_LIMIT},
        "scope": (
            "PySTreeD Accuracy only. Exact fingerprint quotienting is activated only below the root "
            "when the current data view has at most 64 active instances. Ranks 0-23 are opened "
            "development data and ranks 24-35 are a fresh native holdout under the frozen ordering."
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
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
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
