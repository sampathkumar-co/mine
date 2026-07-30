from __future__ import annotations

import json
import tempfile
from pathlib import Path

from enumerator_v25_checkpoint import (
    ControlledCheckpointStop,
    enumerate_programs_checkpointed,
)
from enumerator_v25_recovery import enumerate_programs


BUDGETS = {
    "maximum_depth": 3,
    "maximum_unique_per_type_per_depth": 5000,
    "maximum_total_unique": 30000,
    "maximum_raw_candidates": 300000,
}


def examples():
    return [
        (
            ((0, 3, 0, 0, 3, 0),),
            ((0, 8, 8, 8, 8, 0),),
        )
    ]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def original_result():
    return enumerate_programs(examples(), **BUDGETS)


def checkpoint_result(path: Path, *, resume: bool, stop_after=None):
    return enumerate_programs_checkpointed(
        examples(),
        checkpoint_path=path,
        resume=resume,
        checkpoint_interval_processed_candidates=50,
        stop_after_processed_candidates=stop_after,
        **BUDGETS,
    )


def test_uninterrupted_checkpointed_result_matches_original():
    expected = original_result()
    with tempfile.TemporaryDirectory() as temp:
        actual = checkpoint_result(Path(temp) / "checkpoint.sqlite3", resume=False)
    assert canonical(actual) == canonical(expected)


def test_forced_interruption_and_resume_matches_original():
    expected = original_result()
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "checkpoint.sqlite3"
        interrupted = False
        try:
            checkpoint_result(
                checkpoint,
                resume=False,
                stop_after=200,
            )
        except ControlledCheckpointStop:
            interrupted = True
        assert interrupted
        assert checkpoint.exists()
        actual = checkpoint_result(checkpoint, resume=True)
    assert canonical(actual) == canonical(expected)


def test_resume_rejects_changed_input_binding():
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "checkpoint.sqlite3"
        try:
            checkpoint_result(
                checkpoint,
                resume=False,
                stop_after=200,
            )
        except ControlledCheckpointStop:
            pass
        changed = [
            (
                ((0, 3, 0, 0, 3, 0),),
                ((0, 9, 9, 9, 9, 0),),
            )
        ]
        rejected = False
        try:
            enumerate_programs_checkpointed(
                changed,
                checkpoint_path=checkpoint,
                resume=True,
                checkpoint_interval_processed_candidates=50,
                **BUDGETS,
            )
        except RuntimeError as error:
            rejected = "binding mismatch" in str(error)
        assert rejected


def main():
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
