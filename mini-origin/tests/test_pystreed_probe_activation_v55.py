from pathlib import Path

from mini_origin import pystreed_probe_activation_v55 as v55


def test_probe_policies_and_rank_splits_are_frozen() -> None:
    assert v55.POLICY_CANDIDATES == (
        (4, 1), (6, 1), (8, 1), (8, 2),
        (12, 1), (12, 2), (16, 1), (16, 2),
    )
    assert v55.TRAIN_RANKS == tuple(range(0, 12))
    assert v55.VALIDATION_RANKS == tuple(range(12, 24))
    assert set(v55.TRAIN_RANKS).isdisjoint(v55.VALIDATION_RANKS)


def test_policy_selector_prefers_observed_runtime_then_reduction() -> None:
    rows = [
        {
            "probe_features": 4,
            "probe_duplicates": 1,
            "row_count": 12,
            "optimum_match_count": 12,
            "tasks_with_local_skips": 6,
            "branch_expansion_reduction_fraction": 0.02,
            "reported_speedup_median": 1.04,
            "patched_over_original_runtime_max": 1.05,
        },
        {
            "probe_features": 8,
            "probe_duplicates": 1,
            "row_count": 12,
            "optimum_match_count": 12,
            "tasks_with_local_skips": 8,
            "branch_expansion_reduction_fraction": 0.04,
            "reported_speedup_median": 1.03,
            "patched_over_original_runtime_max": 1.05,
        },
    ]
    selected = v55.select_policy(rows)
    assert selected["selected"]
    assert selected["probe_features"] == 4
    assert selected["probe_duplicates"] == 1


def test_probe_parameters_are_disabled_by_default(tmp_path: Path) -> None:
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
    v55.patch_parameters(root)
    patched = path.read_text(encoding="utf-8")
    assert '"local-equivalence-probe-features"' in patched
    assert '"local-equivalence-probe-duplicates"' in patched
    assert patched.count("DefineIntegerParameter") == 2


def test_protocol_does_not_include_future_holdout() -> None:
    protocol = v55.protocol()
    assert max(protocol["validation_ranks"]) == 23
    assert "holdout_ranks" not in protocol
    assert v55.SELECTION_REPETITIONS == 2
    assert v55.VALIDATION_REPETITIONS == 3
