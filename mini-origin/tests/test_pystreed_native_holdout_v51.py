from pathlib import Path

from mini_origin import pystreed_native_holdout_v51 as v51


def test_native_holdout_ranks_are_disjoint_and_frozen() -> None:
    assert v51.DEVELOPMENT_COUNT == 4
    assert v51.HOLDOUT_COUNT == 8
    development = set(range(v51.DEVELOPMENT_COUNT))
    holdout = set(range(
        v51.DEVELOPMENT_COUNT,
        v51.DEVELOPMENT_COUNT + v51.HOLDOUT_COUNT,
    ))
    assert development.isdisjoint(holdout)
    assert min(holdout) == 4
    assert max(holdout) == 11


def test_hashed_patch_contains_collision_checked_buckets(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "src" / "solver").mkdir(parents=True)
    source = (
        '#include "solver/solver.h"\n'
        '#include "utils/debug.h"\n'
        '#include "utils/utils.h"\n'
        '\ttypename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int org_max_depth, int org_num_nodes) {\n'
        '\t\tint max_depth = org_max_depth, num_nodes = org_num_nodes;\n'
        '\t\tfeature_selector->Initialize(data);\n\n'
        '\t\t// Loop over each feature\n'
        '\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n'
        '\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n'
        '\t\t\t// Generate the context descriptors\n'
    )
    path = root / "src" / "solver" / "solver.cpp"
    path.write_text(source, encoding="utf-8")
    v51.instrument_hashed_solver(root)
    patched = path.read_text(encoding="utf-8")
    assert "std::unordered_map" in patched
    assert "combined_hash" in patched
    assert "for (const auto& previous : bucket)" in patched
    assert "left_partition == previous.first" in patched
    assert "right_partition == previous.first" in patched
    assert "num_local_equivalent_features_skipped++" in patched


def test_native_protocol_uses_repeated_timing() -> None:
    assert v51.REPETITIONS == 3
    assert v51.MAX_ROWS == 256
    assert v51.MAX_FEATURES == 48
    assert v51.PINNED_COMMIT == (
        "9ad41626a1f26c4b7481e8360c5c8b1871e10d96"
    )
