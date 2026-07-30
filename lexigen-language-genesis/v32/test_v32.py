from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V30 = HERE.parent / "v30"
V31 = HERE.parent / "v31"
V25 = HERE.parent / "v25"
for path in (HERE, V30, V31, V25):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scan_one_v32 as scanner
from independent_runtime_v32 import IndependentRuntimeError, evaluate_independent, normalize_grid
from memoized_evaluator_v32 import concrete_candidates, evaluate_candidates_memoized
from runtime_v25 import RuntimeV25Error, eval_ast
from scan_one_v30 import evaluate_candidates as evaluate_candidates_original

EXPECTED_PRECOMMIT = "555194c0d7d35caab361e81a02bd79002fdb3f1837e4180b644ac1f31ffdbe2e"
EXPECTED_MEMO = "c59d6624e60959facc795ab466e027435f4e17466f7fdff126b788fa4ef315e4"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_precommit_and_identity_boundary() -> None:
    precommit = load(HERE / "V32_PRECOMMIT.json")
    v31 = load(V31 / "V31_PRECOMMIT.json")
    assert sha256_file(HERE / "V32_PRECOMMIT.json") == EXPECTED_PRECOMMIT
    selected = precommit["fresh_identity_selection"]["task_ids"]
    previous = set(v31["fresh_identity_selection"]["task_ids"])
    assert len(selected) == 64
    assert len(set(selected)) == 64
    assert not (set(selected) & previous)
    assert precommit["fresh_identity_selection"]["remaining_identity_count_before_selection"] == 92


def test_frozen_bindings() -> None:
    precommit = load(HERE / "V32_PRECOMMIT.json")
    v30_precommit = load(V30 / "V30_PRECOMMIT.json")
    grammar = load(V30 / "V30_GRAMMAR.json")
    manifest = load(V30 / "V30_GRAMMAR_MANIFEST.json")
    scanner.verify_bindings(
        HERE / "V32_PRECOMMIT.json",
        precommit,
        V30 / "V30_PRECOMMIT.json",
        v30_precommit,
        V30 / "V30_GRAMMAR.json",
        grammar,
        V30 / "V30_GRAMMAR_MANIFEST.json",
        manifest,
    )
    assert sha256_file(HERE / "memoized_evaluator_v32.py") == EXPECTED_MEMO
    assert len(grammar["candidates"]) == 23916


def test_scanner_does_not_import_task_modules() -> None:
    text = (HERE / "scan_one_v32.py").read_text(encoding="utf-8")
    assert "tasks.task_" not in text
    assert "importlib.import_module" not in text


def test_every_concrete_program_agrees_across_runtimes() -> None:
    grammar = load(V30 / "V30_GRAMMAR.json")
    precommit = load(V30 / "V30_PRECOMMIT.json")
    concrete, cap_reached = concrete_candidates(grammar, precommit)
    assert cap_reached is False
    assert len(concrete) == 127596
    grids = [
        ((0, 0, 0, 0, 0), (0, 2, 2, 2, 0), (0, 2, 0, 2, 0), (0, 2, 2, 2, 0), (0, 0, 0, 0, 0)),
        ((3, 3, 3, 3), (3, 1, 3, 3), (3, 3, 4, 3), (3, 3, 3, 3)),
        ((7, 7, 7), (7, 5, 7), (7, 7, 7)),
    ]
    primary_errors = (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError)
    independent_errors = (IndependentRuntimeError, ValueError, TypeError, KeyError, IndexError, OverflowError)
    for grid in grids:
        for item in concrete:
            primary_ok = independent_ok = True
            try:
                primary = eval_ast(item.candidate["ast"], grid, item.parameters)
            except primary_errors:
                primary_ok = False
                primary = None
            try:
                independent = evaluate_independent(item.candidate["ast"], grid, item.parameters)
            except independent_errors:
                independent_ok = False
                independent = None
            assert primary_ok == independent_ok, (item.structural_index, item.parameters)
            if primary_ok:
                assert primary == independent, (item.structural_index, item.parameters)


def test_memoized_evaluator_matches_original() -> None:
    grammar = load(V30 / "V30_GRAMMAR.json")
    precommit = load(V30 / "V30_PRECOMMIT.json")
    subset = {
        **grammar,
        "candidates": grammar["candidates"][:600],
    }
    examples = [
        (
            ((0, 0, 0), (0, 2, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 5, 0), (0, 0, 0)),
        ),
        (
            ((3, 3, 3), (3, 1, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 5, 3), (3, 3, 3)),
        ),
    ]
    original = evaluate_candidates_original(examples, subset, precommit)
    memoized = evaluate_candidates_memoized(examples, subset, precommit)
    assert memoized == original


def test_selected_candidate_is_frozen_minimum() -> None:
    grammar = load(V30 / "V30_GRAMMAR.json")
    precommit = load(V30 / "V30_PRECOMMIT.json")
    examples = [
        (
            ((0, 0, 0), (0, 2, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 5, 0), (0, 0, 0)),
        ),
        (
            ((3, 3, 3), (3, 1, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 5, 3), (3, 3, 3)),
        ),
    ]
    result = evaluate_candidates_memoized(examples, grammar, precommit)
    assert result["exact_candidate_count"] > 0
    assert result["selected_candidate"] == result["exact_candidates"][0]
    assert result["selected_candidate"]["structural_index"] == 4
    assert result["selected_candidate"]["parameters"] == {"c0": 5}


def recolor_pair() -> dict[str, Any]:
    source = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    target = [[0, 0, 0], [0, 5, 0], [0, 0, 0]]
    return {"input": source, "output": target}


def test_fresh_gate_passes_with_independent_runtime() -> None:
    grammar = load(V30 / "V30_GRAMMAR.json")
    ast = grammar["candidates"][4]["ast"]
    original = scanner.generate_case
    scanner.generate_case = lambda *args, **kwargs: (recolor_pair(), {"status": "ok", "seed": args[2]})
    try:
        result = scanner.run_fresh_gate(
            Path("unused"), "synthetic", ast, {"c0": 5}, 100, 5
        )
    finally:
        scanner.generate_case = original
    assert result["passed"] is True
    assert result["totals"]["generated_cases"] == 100
    assert result["totals"]["passed_cases"] == 100


def test_fresh_gate_rejects_wrong_target() -> None:
    grammar = load(V30 / "V30_GRAMMAR.json")
    ast = grammar["candidates"][4]["ast"]
    wrong = {"input": [[0, 0, 0], [0, 2, 0], [0, 0, 0]], "output": [[0, 0, 0], [0, 6, 0], [0, 0, 0]]}
    original = scanner.generate_case
    scanner.generate_case = lambda *args, **kwargs: (wrong, {"status": "ok", "seed": args[2]})
    try:
        result = scanner.run_fresh_gate(
            Path("unused"), "synthetic", ast, {"c0": 5}, 3, 5
        )
    finally:
        scanner.generate_case = original
    assert result["passed"] is False
    assert result["totals"]["target_mismatches"] == 3


def test_seed_schedule_is_fixed() -> None:
    assert scanner.seed_for("lexigen-v32-demonstration", "f8c80d96", 0) == 3484583531
    assert scanner.seed_for("lexigen-v32-demonstration", "f8c80d96", 15) == 1157587407
    assert scanner.seed_for("lexigen-v32-fresh", "f8c80d96", 0) == 1624427756
    assert scanner.seed_for("lexigen-v32-fresh", "f8c80d96", 99) == 3111077425


def main() -> None:
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
