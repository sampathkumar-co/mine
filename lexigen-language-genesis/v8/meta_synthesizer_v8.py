from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Sequence

from meta_runtime_v8 import Grid, MetaRuntimeError, canonical_json, execute_extension


def var(name: str) -> dict[str, Any]:
    return {"op": "var", "name": name}


def const(value: int) -> dict[str, Any]:
    return {"op": "const", "value": value}


def unary(op: str, arg: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "arg": arg}


def binary(op: str, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": op, "left": left, "right": right}


def relation_candidates() -> list[dict[str, Any]]:
    dr, dc = var("dr"), var("dc")
    adr, adc = unary("abs", dr), unary("abs", dc)
    atoms = [
        binary("eq", adr, const(1)),
        binary("eq", adc, const(1)),
        binary("eq", dr, const(0)),
        binary("eq", dc, const(0)),
        binary("eq", adr, adc),
        binary("eq", binary("add", adr, adc), const(1)),
        binary("eq", binary("add", adr, adc), const(2)),
        binary("eq", unary("abs", binary("sub", adr, adc)), const(0)),
    ]
    candidates = list(atoms)
    candidates.extend({"op": "and", "args": [left, right]} for left, right in itertools.combinations(atoms, 2))
    unique = {canonical_json(candidate): candidate for candidate in candidates}
    return [unique[key] for key in sorted(unique, key=lambda text: hashlib.sha256(text.encode()).digest())]


def component_class_candidates() -> list[dict[str, Any]]:
    dr, dc = var("dr"), var("dc")
    edge_bodies = [
        dr,
        dc,
        binary("mul", dr, dc),
        binary("add", dr, dc),
        binary("sub", dr, dc),
        unary("abs", dr),
        unary("abs", dc),
    ]
    candidates = [
        unary("sign", {"op": "fold_sum", "collection": "edges", "body": body})
        for body in edge_bodies
    ]
    unique = {canonical_json(candidate): candidate for candidate in candidates}
    return [unique[key] for key in sorted(unique, key=lambda text: hashlib.sha256(text.encode()).digest())]


def group_score_candidates() -> list[dict[str, Any]]:
    return [
        {
            "op": "fold_sum",
            "collection": "components",
            "body": {"op": "cardinality", "target": "points"},
        },
        {
            "op": "fold_sum",
            "collection": "components",
            "body": {"op": "const", "value": 1},
        },
    ]


def infer_changed_colours(examples: Sequence[tuple[Grid, Grid]]) -> tuple[int, tuple[int, int]]:
    sources: set[int] = set()
    targets: set[int] = set()
    for source, target in examples:
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            raise ValueError("v8 currently requires shape-preserving demonstrations")
        for source_row, target_row in zip(source, target):
            for before, after in zip(source_row, target_row):
                if before != after:
                    sources.add(before)
                    targets.add(after)
    if len(sources) != 1 or len(targets) != 2:
        raise ValueError("v8 requires one changed source colour and two target colours")
    return next(iter(sources)), tuple(sorted(targets))  # type: ignore[return-value]


def extension_description_length(extension: dict[str, Any]) -> int:
    return len(canonical_json(extension))


@dataclass(frozen=True)
class MetaSynthesisResult:
    extension: dict[str, Any] | None
    candidates_tested: int
    exact_candidate_count: int
    fixed_grammar_baseline_found: bool


def fixed_grammar_baseline(examples: Sequence[tuple[Grid, Grid]], source_colour: int, targets: tuple[int, int]) -> bool:
    for target_colour in targets:
        if all(
            tuple(
                tuple(target_colour if value == source_colour else value for value in row)
                for row in source
            )
            == target
            for source, target in examples
        ):
            return True
    return False


def synthesize_meta_extension(examples: Sequence[tuple[Grid, Grid]]) -> MetaSynthesisResult:
    if not examples:
        raise ValueError("at least one demonstration is required")
    source_colour, target_colours = infer_changed_colours(examples)
    baseline = fixed_grammar_baseline(examples, source_colour, target_colours)
    exact: list[dict[str, Any]] = []
    tested = 0
    for relation, component_class, group_score, winner_mode, mapping in itertools.product(
        relation_candidates(),
        component_class_candidates(),
        group_score_candidates(),
        ("max", "min"),
        (target_colours, tuple(reversed(target_colours))),
    ):
        tested += 1
        extension = {
            "schema": "lexigen-meta-grammar-extension-v1",
            "types": {
                "input": "Grid",
                "source": "Set[Point]",
                "relation": "Point×Point→Bool",
                "component_class": "Component→Int",
                "group_score": "List[Component]→Int",
                "output": "Grid",
            },
            "source": {"op": "select_cells_equal", "colour": source_colour},
            "relation": relation,
            "component_class": component_class,
            "group_score": group_score,
            "winner": {"op": "select_extreme_group", "mode": winner_mode},
            "render": {
                "op": "paint_component_class",
                "winner_colour": mapping[0],
                "other_colour": mapping[1],
            },
        }
        try:
            if all(execute_extension(extension, source) == target for source, target in examples):
                exact.append(extension)
        except (MetaRuntimeError, ValueError, IndexError):
            continue
    if not exact:
        return MetaSynthesisResult(None, tested, 0, baseline)
    chosen = min(
        exact,
        key=lambda extension: (
            extension_description_length(extension),
            hashlib.sha256(canonical_json(extension).encode()).digest(),
        ),
    )
    digest = hashlib.sha256(canonical_json(chosen).encode()).hexdigest()
    chosen = dict(chosen)
    chosen["name"] = "generated_production_" + digest[:12]
    chosen["provenance"] = {
        "method": "typed enumerative meta-grammar synthesis",
        "candidate_extensions_tested": tested,
        "exact_candidate_count": len(exact),
        "human_supplied_finished_task_operator": False,
        "human_supplied_generic_substrate": True,
    }
    return MetaSynthesisResult(chosen, tested, len(exact), baseline)
