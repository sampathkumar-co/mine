from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from grammar_v30 import build_grammar, parameter_names, write_grammar
from scan_one_v30 import evaluate_candidates

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V30_PRECOMMIT.json"
GRAMMAR = HERE / "V30_GRAMMAR.json"
SOURCES = HERE.parent / "v29" / "V29_SOURCE_STRUCTURES.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def operations(value):
    result = set()
    if isinstance(value, dict):
        if isinstance(value.get("op"), str):
            result.add(value["op"])
        for child in value.values():
            result.update(operations(child))
    elif isinstance(value, list):
        for child in value:
            result.update(operations(child))
    return result


def test_grammar_bounds_and_order() -> None:
    precommit = load(PRECOMMIT)
    grammar = load(GRAMMAR)
    assert grammar["structural_candidate_count"] == 23916
    assert grammar["structural_cap_reached"] is False
    assert grammar["structural_candidate_count"] <= precommit["enumeration"]["maximum_structural_candidates"]
    keys = [
        (item["depth"], item["nodes"], canonical(item["ast"]))
        for item in grammar["candidates"]
    ]
    assert keys == sorted(keys)


def test_only_source_induced_operators() -> None:
    precommit = load(PRECOMMIT)
    grammar = load(GRAMMAR)
    allowed = set(precommit["source_operator_inventory"])
    observed = set()
    for candidate in grammar["candidates"]:
        observed.update(operations(candidate["ast"]))
    assert observed <= allowed
    assert observed == allowed


def test_all_source_programs_are_present() -> None:
    grammar = load(GRAMMAR)
    sources = load(SOURCES)
    candidates = {canonical(item["ast"]) for item in grammar["candidates"]}
    source_asts = {canonical(item["structure"]) for item in sources["structures"]}
    assert len(source_asts) == 3
    assert source_asts <= candidates


def test_grammar_reproduces_deterministically() -> None:
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.json"
        second = Path(directory) / "second.json"
        one = write_grammar(PRECOMMIT, first)
        two = write_grammar(PRECOMMIT, second)
        assert first.read_bytes() == second.read_bytes()
        assert one["candidate_sha256"] == two["candidate_sha256"]
        assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()


def test_synthetic_recombination_is_solved() -> None:
    grammar = load(GRAMMAR)
    precommit = load(PRECOMMIT)
    target_ast = {
        "op": "paint",
        "grid": {"op": "input_grid"},
        "points": {"op": "non_background_points"},
        "colour": {"op": "param_color", "name": "c0"},
    }
    selected = [
        item for item in grammar["candidates"]
        if canonical(item["ast"]) == canonical(target_ast)
    ]
    assert len(selected) == 1
    mini = {"candidates": selected}
    examples = [
        (((0, 2), (0, 0)), ((0, 8), (0, 0))),
        (((3, 0, 3),), ((3, 8, 3),)),
    ]
    result = evaluate_candidates(examples, mini, precommit)
    assert result["exact_candidate_count"] == 1
    assert result["selected_candidate"]["parameters"] == {"c0": 8}


def test_parameter_inventory_is_single_color() -> None:
    grammar = load(GRAMMAR)
    names = set()
    for candidate in grammar["candidates"]:
        names.update(parameter_names(candidate["ast"]))
    assert names == {"c0"}


def test_scanner_has_no_direct_validation_import() -> None:
    text = (HERE / "scan_one_v30.py").read_text(encoding="utf-8")
    assert "tasks.task_" not in text
    assert "importlib.import_module" not in text


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
