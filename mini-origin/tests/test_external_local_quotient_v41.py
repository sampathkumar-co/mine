from pathlib import Path
import json
import zipfile

from mini_origin import external_local_quotient_v41 as v41
from mini_origin import state_policy_v34 as v34


def test_manifest_is_committed_before_parsing() -> None:
    manifest = json.loads(v41.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "download_hash_only_no_record_parsing_v1"
    assert manifest["archive_count"] == 4
    assert manifest["frozen_parent_compiler_digest"] == v41.FROZEN_PARENT_DIGEST
    assert all(len(row["sha256"]) == 64 for row in manifest["rows"])


def test_frozen_gate_constants() -> None:
    assert v41.CANDIDATE_THRESHOLD == 12
    assert v41.SEEDS == (2301, 2302, 2303, 2304, 2305)
    assert v41.LARGE_DOMAINS == {"chess-kr-vs-kp", "splice-junction"}


def test_chess_parser_uses_36_features(tmp_path: Path) -> None:
    fields = ["f"] * 36 + ["won"]
    with zipfile.ZipFile(tmp_path / "chess-kr-vs-kp.zip", "w") as handle:
        handle.writestr("kr-vs-kp.data", ",".join(fields) + "\n")
    task = v41.load_chess(tmp_path)
    assert task.query_count == 36
    assert task.candidate_count == 1
    assert task.labels == ("won",)


def test_splice_parser_excludes_identifier_and_splits_sequence(tmp_path: Path) -> None:
    sequence = "ACGT" * 15
    with zipfile.ZipFile(tmp_path / "splice.zip", "w") as handle:
        handle.writestr(
            "splice.data",
            f"EI, INSTANCE-1, {sequence}\n",
        )
    task = v41.load_splice(tmp_path)
    assert task.query_count == 60
    assert task.candidate_count == 1
    assert task.rows[0] == tuple(sequence)
    assert task.labels == ("EI",)


def test_hepatitis_preserves_missing_value_category(tmp_path: Path) -> None:
    fields = ["2"] + ["1"] * 18 + ["?"]
    with zipfile.ZipFile(tmp_path / "hepatitis.zip", "w") as handle:
        handle.writestr("hepatitis.data", ",".join(fields) + "\n")
    task = v41.load_hepatitis(tmp_path)
    assert task.query_count == 19
    assert task.rows[0][-1] == "?"
    assert task.labels == ("2",)


def test_row_permutation_preserves_records_and_labels() -> None:
    task = v34.base.make_task(
        "permutation-toy",
        ("a", "b"),
        (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")),
        ("x", "y", "y", "x"),
    )
    permuted = v41.permute_task(task, 2301)
    assert sorted(permuted.rows) == sorted(task.rows)
    assert sorted(permuted.labels) == sorted(task.labels)
