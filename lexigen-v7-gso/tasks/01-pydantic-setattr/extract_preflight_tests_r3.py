from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import duckdb

HF_DATASET = "gso-bench/gso"
HF_REVISION = "c2e4f1a58427cccd15e0e542f136bd204fb19284"
PARQUET_PATH = "data/test-00000-of-00001.parquet"
INSTANCE_ID = "pydantic__pydantic-addf1f9"
EXPECTED = [
    "97bc04c54bdef1b4d11443d6c806e8c665520f4235ec229dc66239b537b45e4e",
    "59ff7d0e1a796834640d25304b603c07bf77c46946c6f4fb90210f76cc1aa6dd",
    "b583ee6c5140cbc37ae8d3a32457a8d07b7458ac50ecca07ac7518fe5846b30c",
    "00b97bd5792c4e92dc810098d0e8731d742124e0cf7c1e015ef799cda353ab29",
    "4dbc59c9f72a659bf35962d5cde50f2c8bbd482b5f208e1d0f3db5c5ffdc7cbb",
    "460a7969acd2f3d1818b3eceaa607ff1766e268e61b0983029ae267c8952f935",
    "efcf30defa8294725f7c40e4e155cff640c007e5a061f07b7a3c479857fcdb2e",
    "c821fe1743b7b6bab68425f7fafcc894377691a6237ed86ba00372df34cd682e",
    "ed75147edd43ec8bb1e92e70daccee6d64ac33cc65e1e04b49b246366da6be65",
    "7012b75a327575a550e73a37810a203ac3286f3c9be75ca480fbf5030c38d866",
    "7a8096455a7054a337190415329d9f765f935897945689f197faceab8f6553cb",
    "3a2643b7c3aebdff01da8589faf800ea7449834cf09542ae3ebb90dacebb2651",
    "f327a08d75d471c49f8b85239d9559d75958fce0b1b1e8077463090d1e3c29f3",
    "db4667f52a441d1b755a83f3096d582799194470457d48c345f0bf07c8851bb6",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parquet_url = (
        f"https://huggingface.co/datasets/{HF_DATASET}/resolve/"
        f"{HF_REVISION}/{PARQUET_PATH}?download=true"
    )
    con = duckdb.connect(database=":memory:")
    con.execute("SET enable_http_metadata_cache=false")
    rows = con.execute(
        'SELECT "instance_id", "tests" FROM read_parquet(?) WHERE instance_id = ?',
        [parquet_url, INSTANCE_ID],
    ).fetchall()
    if len(rows) != 1 or str(rows[0][0]) != INSTANCE_ID:
        raise RuntimeError("could not recover exactly one frozen Task-1 test row")
    tests = list(rows[0][1] or [])
    if len(tests) != len(EXPECTED):
        raise RuntimeError(f"expected {len(EXPECTED)} tests, got {len(tests)}")
    observed = [hashlib.sha256(str(test).encode()).hexdigest() for test in tests]
    if observed != EXPECTED:
        raise RuntimeError(f"Task-1 tests do not match source-stage sealed hashes: {observed}")
    for i, test in enumerate(tests):
        (args.output / f"gso_test_{i}.py").write_text(str(test), encoding="utf-8")
    (args.output / "HASHES.txt").write_text(
        "\n".join(f"{i} {digest}" for i, digest in enumerate(observed)) + "\n",
        encoding="utf-8",
    )
    print(f"sealed_task1_tests_verified={len(tests)}")
    print("expert_columns_requested=false")


if __name__ == "__main__":
    main()
