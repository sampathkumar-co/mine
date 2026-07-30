from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(V25) not in sys.path:
    sys.path.insert(0, str(V25))

from fresh_validate_v30_956 import (
    derive_verifier,
    independent_background,
    independent_execute,
    load,
    seed_for,
    validate_bindings,
    verify_relation,
)
from runtime_v25 import eval_ast


def test_seed_schedule_is_fixed() -> None:
    assert seed_for(0) == 3544805667
    assert seed_for(1) == 1501182928
    assert seed_for(999) == 2237336463
    assert len({seed_for(index) for index in range(1000)}) == 1000


def test_background_tie_rule_is_independent() -> None:
    grid = ((3, 1), (1, 3))
    assert independent_background(grid) == 1


def test_runtime_agreement_on_synthetic_grids() -> None:
    precommit = load(HERE / "V30_956_FRESH_PRECOMMIT.json")
    ast, parameters, verifier = validate_bindings(precommit)
    grids = (
        ((0, 2, 0), (2, 2, 0)),
        ((3, 1), (1, 3)),
        ((7, 7, 4), (7, 6, 7)),
    )
    for grid in grids:
        primary = eval_ast(ast, grid, parameters)
        independent = independent_execute(grid, 5)
        assert primary == independent
        assert verify_relation(grid, independent, verifier)


def test_candidate_binding_is_exact() -> None:
    precommit = load(HERE / "V30_956_FRESH_PRECOMMIT.json")
    ast, parameters, verifier = validate_bindings(precommit)
    assert ast == precommit["selected_candidate"]["ast"]
    assert parameters == {"c0": 5}
    assert verifier["foreground_color"] == 5


def test_verifier_rejects_wrong_output() -> None:
    precommit = load(HERE / "V30_956_FRESH_PRECOMMIT.json")
    ast, parameters, verifier = validate_bindings(precommit)
    source = ((0, 2), (0, 0))
    assert verify_relation(source, eval_ast(ast, source, parameters), verifier)
    assert not verify_relation(source, source, verifier)


def test_verifier_synthesis_rejects_ast_change() -> None:
    precommit = load(HERE / "V30_956_FRESH_PRECOMMIT.json")
    altered = json.loads(json.dumps(precommit["selected_candidate"]["ast"]))
    altered["points"]["op"] = "holes"
    try:
        derive_verifier(altered, {"c0": 5})
    except RuntimeError:
        return
    raise AssertionError("altered AST was accepted")


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
