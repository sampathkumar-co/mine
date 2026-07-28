from pathlib import Path

from mini_origin import pystreed_selective_threshold_v53 as v53


def test_threshold_candidates_and_rank_splits_are_frozen() -> None:
    assert v53.THRESHOLD_CANDIDATES == (
        0, 12, 16, 24, 32, 48, 64, 96, 128, 256
    )
    assert set(v53.TRAIN_RANKS).isdisjoint(v53.VALIDATION_RANKS)
    assert v53.TRAIN_RANKS == tuple(range(0, 12))
    assert v53.VALIDATION_RANKS == tuple(range(12, 24))


def test_selection_uses_only_admissible_positive_thresholds() -> None:
    rows = [
        {
            "threshold": 0,
            "row_count": 12,
            "optimum_match_count": 12,
            "tasks_with_local_skips": 0,
            "branch_expansion_reduction_fraction": 0.0,
            "reported_speedup_median": 1.2,
            "patched_over_original_runtime_max": 1.0,
        },
        {
            "threshold": 24,
            "row_count": 12,
            "optimum_match_count": 12,
            "tasks_with_local_skips": 6,
            "branch_expansion_reduction_fraction": 0.03,
            "reported_speedup_median": 1.04,
            "patched_over_original_runtime_max": 1.1,
        },
        {
            "threshold": 48,
            "row_count": 12,
            "optimum_match_count": 12,
            "tasks_with_local_skips": 8,
            "branch_expansion_reduction_fraction": 0.05,
            "reported_speedup_median": 1.03,
            "patched_over_original_runtime_max": 1.1,
        },
    ]
    selected = v53.select_threshold(rows)
    assert selected["selected"]
    assert selected["threshold"] == 24


def test_adaptive_parameter_patch_is_integer_and_disabled_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    (root / "src" / "solver").mkdir(parents=True)
    path = root / "src" / "solver" / "define_parameters.cpp"
    anchor = (
        '\t\tparameters.DefineBooleanParameter\n'
        '\t\t(\n'
        '\t\t\t"use-similarity-lower-bound",\n'
        '\t\t\t"Activate similarity-based lower bounding. Disabling this option may be better for some benchmarks, but on most of our tested datasets keeping this on was beneficial.",\n'
        '\t\t\ttrue,\n'
        '\t\t\t"Algorithmic Parameters"\n'
        '\t\t);\n'
    )
    path.write_text(anchor, encoding="utf-8")
    v53.patch_parameter(root)
    patched = path.read_text(encoding="utf-8")
    assert '"local-equivalence-max-instances"' in patched
    assert "DefineIntegerParameter" in patched
    assert '\t\t\t0,\n\t\t\t"Algorithmic Parameters"' in patched


def test_protocol_never_names_future_holdout_ranks() -> None:
    protocol = v53.protocol()
    assert max(protocol["validation_ranks"]) == 23
    assert "holdout_ranks" not in protocol
    assert v53.SELECTION_REPETITIONS == 2
    assert v53.VALIDATION_REPETITIONS == 3
