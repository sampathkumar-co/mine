from __future__ import annotations

import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V30 = HERE.parent / "v30"
for path in (HERE, V30):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scan_one_v30 import evaluate_candidates, load
from scan_one_v30_recovery import (
    concrete_candidates,
    evaluate_candidates_memoized,
    prepare_node,
)


def sample_examples() -> list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]]:
    return [
        (((0, 2, 0), (2, 2, 0)), ((0, 5, 0), (5, 5, 0))),
        (((3, 1), (1, 3)), ((5, 1), (1, 5))),
    ]


def grammar_prefix(count: int) -> dict:
    grammar = load(V30 / "V30_GRAMMAR.json")
    result = copy.deepcopy(grammar)
    result["candidates"] = result["candidates"][:count]
    return result


def test_prepared_nodes_track_parameter_dependency() -> None:
    grammar = grammar_prefix(20)
    dependencies = [prepare_node(item["ast"]).depends_on_parameter for item in grammar["candidates"]]
    assert any(dependencies)
    assert any(not value for value in dependencies)


def test_concrete_candidate_order_matches_original_denominator() -> None:
    grammar = grammar_prefix(250)
    precommit = load(V30 / "V30_PRECOMMIT.json")
    concrete, cap_reached = concrete_candidates(grammar, precommit)
    assert not cap_reached
    assert len(concrete) > len(grammar["candidates"])
    keys = [
        (item.structural_index, tuple(sorted(item.parameters.items())))
        for item in concrete
    ]
    assert keys == sorted(keys, key=lambda item: (item[0], item[1]))


def test_memoized_result_matches_original_prefix() -> None:
    grammar = grammar_prefix(600)
    precommit = load(V30 / "V30_PRECOMMIT.json")
    examples = sample_examples()
    original = evaluate_candidates(examples, grammar, precommit)
    recovered = evaluate_candidates_memoized(examples, grammar, precommit)
    assert recovered == original


def test_exact_and_identity_accounting_matches() -> None:
    grammar = grammar_prefix(50)
    precommit = load(V30 / "V30_PRECOMMIT.json")
    examples = [(((0, 2), (0, 0)), ((0, 5), (0, 0)))]
    original = evaluate_candidates(examples, grammar, precommit)
    recovered = evaluate_candidates_memoized(examples, grammar, precommit)
    assert recovered == original
    assert recovered["task_nontrivial"] is True


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
