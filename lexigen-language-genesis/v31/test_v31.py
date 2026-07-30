from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import scan_one_v31 as scanner


def motif_ast() -> dict:
    return {
        "op": "paint",
        "grid": {"op": "input_grid"},
        "points": {"op": "non_background_points"},
        "colour": {"op": "param_color", "name": "c0"},
    }


def test_seed_schedule_is_fixed() -> None:
    assert scanner.seed_for("lexigen-v31-demonstration", "bd4472b8", 0) == 157150798
    assert scanner.seed_for("lexigen-v31-fresh", "bd4472b8", 0) == 2915976672
    assert scanner.seed_for("lexigen-v31-fresh", "bd4472b8", 99) == 3018459013


def test_independent_background_tie_rule() -> None:
    grid = ((3, 1), (1, 3))
    assert scanner.independent_background(grid) == 1
    assert scanner.independent_execute(grid, 5) == ((5, 1), (1, 5))


def test_unique_demonstration_match() -> None:
    examples = [
        (((0, 2, 0), (2, 2, 0)), ((0, 5, 0), (5, 5, 0))),
        (((3, 1), (1, 3)), ((5, 1), (1, 5))),
    ]
    exact, invalid, identity = scanner.match_demonstrations(
        examples, motif_ast(), list(range(10))
    )
    assert exact == [5]
    assert invalid == 0
    assert identity == 0


def test_identity_candidates_are_rejected() -> None:
    examples = [
        (((0, 0), (0, 0)), ((0, 0), (0, 0))),
    ]
    exact, invalid, identity = scanner.match_demonstrations(
        examples, motif_ast(), list(range(10))
    )
    assert exact == []
    assert invalid == 0
    assert identity == 10


def test_scanner_does_not_import_task_modules() -> None:
    source = (HERE / "scan_one_v31.py").read_text(encoding="utf-8")
    assert "tasks.task_" not in source
    assert "importlib" not in source


def test_fresh_gate_passes_with_two_runtimes() -> None:
    original = scanner.generate_case
    pairs = {
        scanner.seed_for("lexigen-v31-fresh", "synthetic", index): {
            "input": [[0, 2, 0], [2, 0, 0]],
            "output": [[0, 5, 0], [5, 0, 0]],
        }
        for index in range(3)
    }

    def fake_generate(_root, _task_id, seed, _timeout):
        return pairs[seed], {"status": "ok", "seed": seed}

    scanner.generate_case = fake_generate
    try:
        result = scanner.run_fresh_gate(
            Path("."), "synthetic", motif_ast(), 5, 3, 5
        )
    finally:
        scanner.generate_case = original
    assert result["passed"] is True
    assert result["totals"]["passed_cases"] == 3
    assert result["totals"]["runtime_disagreements"] == 0


def test_fresh_gate_rejects_wrong_target() -> None:
    original = scanner.generate_case

    def fake_generate(_root, _task_id, seed, _timeout):
        return {
            "input": [[0, 2]],
            "output": [[0, 4]],
        }, {"status": "ok", "seed": seed}

    scanner.generate_case = fake_generate
    try:
        result = scanner.run_fresh_gate(
            Path("."), "synthetic-fail", motif_ast(), 5, 1, 5
        )
    finally:
        scanner.generate_case = original
    assert result["passed"] is False
    assert result["totals"]["target_mismatches"] == 1


def test_precommit_has_fresh_unique_identities() -> None:
    precommit = scanner.load(HERE / "V31_PRECOMMIT.json")
    ids = precommit["fresh_identity_selection"]["task_ids"]
    assert len(ids) == 64
    assert len(set(ids)) == 64
    assert "9565186b" not in ids
    assert precommit["motif"]["candidate_colors"] == list(range(10))


def main() -> None:
    tests = sorted(
        (name, value) for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
