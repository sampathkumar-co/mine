from pathlib import Path

from mini_origin import pystreed_count_filtered_v54 as v54


def test_count_filtered_dataset_scope_is_opened_only() -> None:
    assert v54.DATASET_RANKS == tuple(range(0, 24))
    assert v54.REPETITIONS == 3
    assert v54.MAX_ROWS == 256
    assert v54.MAX_FEATURES == 48


def test_count_filtered_patch_uses_counts_then_exact_equality(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    (root / "src" / "solver").mkdir(parents=True)
    path = root / "src" / "solver" / "solver.cpp"
    path.write_text(
        '#include "solver/solver.h"\n'
        '#include "utils/debug.h"\n'
        '#include "utils/utils.h"\n'
        '\ttypename Solver<OT>::SolContainer Solver<OT>::SolveSubTree(ADataView& data, const Solver<OT>::Context& context, typename Solver<OT>::SolContainer UB_, int org_max_depth, int org_num_nodes) {\n'
        '\t\tint max_depth = org_max_depth, num_nodes = org_num_nodes;\n'
        '\t\tfeature_selector->Initialize(data);\n\n'
        '\t\t// Loop over each feature\n'
        '\t\t\tdata_splitter.Split(data, branch, feature, left_data, right_data);\n'
        '\t\t\tif (!SatisfiesMinimumLeafNodeSize(left_data) || !SatisfiesMinimumLeafNodeSize(right_data)) continue;\n\n'
        '\t\t\t// Generate the context descriptors\n',
        encoding="utf-8",
    )
    v54.patch_count_filtered_solver(root)
    patched = path.read_text(encoding="utf-8")
    assert "NumInstancesForLabel" in patched
    assert "local_accuracy_count_buckets" in patched
    assert "left_data == previous_left" in patched
    assert "right_data == previous_left" in patched
    assert "ADataViewBitSet" not in patched
    assert "DataSplitFingerprint" not in patched


def test_external_source_revision_is_fixed() -> None:
    assert v54.PINNED_COMMIT == (
        "9ad41626a1f26c4b7481e8360c5c8b1871e10d96"
    )
