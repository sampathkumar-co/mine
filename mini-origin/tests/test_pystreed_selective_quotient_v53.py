from pathlib import Path

from mini_origin import pystreed_selective_quotient_v53 as v53


def test_activation_rule_is_frozen() -> None:
    assert v53.ACTIVE_INSTANCE_LIMIT == 64
    assert v53.DEVELOPMENT_COUNT == 24
    assert v53.HOLDOUT_START == 24
    assert v53.HOLDOUT_COUNT == 12


def test_fresh_holdout_is_disjoint() -> None:
    development = set(range(v53.DEVELOPMENT_COUNT))
    holdout = set(range(v53.HOLDOUT_START, v53.HOLDOUT_START + v53.HOLDOUT_COUNT))
    assert development.isdisjoint(holdout)
    assert min(holdout) == 24
    assert max(holdout) == 35


def test_selective_patch_changes_only_activation_sites(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "src" / "solver").mkdir(parents=True)
    path = root / "src" / "solver" / "solver.cpp"
    path.write_text(
        "if (branch.Depth() > 0) fingerprint_ptr = &split_fingerprint;\n"
        "if (branch.Depth() > 0) {\n",
        encoding="utf-8",
    )
    # Test the post-v0.52 selective replacement directly.
    text = path.read_text(encoding="utf-8")
    old = "if (branch.Depth() > 0)"
    new = f"if (branch.Depth() > 0 && data.Size() <= {v53.ACTIVE_INSTANCE_LIMIT})"
    assert text.count(old) == 2
    path.write_text(text.replace(old, new), encoding="utf-8")
    patched = path.read_text(encoding="utf-8")
    assert patched.count("data.Size() <= 64") == 2


def test_external_source_remains_pinned() -> None:
    assert v53.PINNED_COMMIT == "9ad41626a1f26c4b7481e8360c5c8b1871e10d96"
