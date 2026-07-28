from pathlib import Path

from mini_origin import pystreed_fingerprint_holdout_v52 as v52


def test_second_native_holdout_is_disjoint_and_frozen() -> None:
    development = set(range(v52.DEVELOPMENT_COUNT))
    holdout = set(range(
        v52.HOLDOUT_START,
        v52.HOLDOUT_START + v52.HOLDOUT_COUNT,
    ))
    assert v52.DEVELOPMENT_COUNT == 12
    assert v52.HOLDOUT_COUNT == 12
    assert development.isdisjoint(holdout)
    assert min(holdout) == 12
    assert max(holdout) == 23


def test_fingerprint_patch_avoids_dynamic_bitset_archive(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "include" / "model").mkdir(parents=True)
    header = root / "include" / "model" / "data.h"
    header.write_text(
        "\tclass ADataView;\n"
        "\t\tvoid SplitData(int feature, ADataView& left, ADataView& right) const;\n",
        encoding="utf-8",
    )
    v52.patch_data_header(root)
    patched = header.read_text(encoding="utf-8")
    assert "struct DataSplitFingerprint" in patched
    assert "left_xor" in patched
    assert "left_sum" in patched
    assert "DataSplitFingerprint* fingerprint" in patched
    assert "ADataViewBitSet" not in patched


def test_fingerprint_protocol_reuses_native_limits() -> None:
    assert v52.v51.MAX_ROWS == 256
    assert v52.v51.MAX_FEATURES == 48
    assert v52.v51.REPETITIONS == 3
    assert v52.PINNED_COMMIT == (
        "9ad41626a1f26c4b7481e8360c5c8b1871e10d96"
    )
