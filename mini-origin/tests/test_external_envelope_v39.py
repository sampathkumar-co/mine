from pathlib import Path
import json
import zipfile

from mini_origin import external_envelope_v39 as v39
from mini_origin import state_policy_v34 as v34


def test_manifest_is_hash_committed() -> None:
    manifest = json.loads(v39.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["protocol"] == "download_hash_only_no_record_parsing_v1"
    assert manifest["archive_count"] == 4
    assert all(len(row["sha256"]) == 64 for row in manifest["rows"])
    assert all(row["bytes"] > 0 for row in manifest["rows"])


def test_frozen_external_gate_constants() -> None:
    assert v39.CANDIDATE_THRESHOLD == 12
    assert v39.FEATURE_LIMIT == 12
    assert v39.SEEDS == (2201, 2202, 2203, 2204, 2205)
    assert v39.FROZEN_DEVELOPMENT_DIGEST == (
        "8117605201d4a4dba684757b2802e4426cacb0065e878b58708196606bfa28dc"
    )


def test_audiology_parser_excludes_unique_identifier(tmp_path: Path) -> None:
    fields = ["f"] * 69 + ["p1", "class-a"]
    with zipfile.ZipFile(tmp_path / "audiology-standardized.zip", "w") as z:
        z.writestr("audiology.standardized.data", ",".join(fields) + "\n")
        fields[-2:] = ["t1", "class-b"]
        z.writestr("audiology.standardized.test", ",".join(fields) + "\n")
    task = v39.load_audiology(tmp_path)
    assert task.query_count == 69
    assert task.candidate_count == 2
    assert task.rows[0] == tuple(["f"] * 69)
    assert task.labels == ("class-a", "class-b")


def test_row_permutation_preserves_task_statistics() -> None:
    task = v34.base.make_task(
        "permutation-toy",
        ("a", "b"),
        (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")),
        ("x", "y", "y", "x"),
    )
    permuted = v39.permute_task(task, 2201)
    assert permuted.candidate_count == task.candidate_count
    assert permuted.query_count == task.query_count
    assert sorted(permuted.rows) == sorted(task.rows)
    assert sorted(permuted.labels) == sorted(task.labels)
