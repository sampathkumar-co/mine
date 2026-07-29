from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any

AST = dict[str, Any]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def tree_size(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(tree_size(item) for key, item in value.items() if key != "op")
    if isinstance(value, list):
        return 1 + sum(tree_size(item) for item in value)
    return 1


def variable_count(value: Any) -> int:
    if isinstance(value, dict) and set(value) == {"$var"}:
        return 1
    if isinstance(value, dict):
        return sum(variable_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(variable_count(item) for item in value)
    return 0


def collect_subtrees(value: Any) -> list[Any]:
    result: list[Any] = []
    if isinstance(value, dict) and "op" in value:
        result.append(value)
    if isinstance(value, dict):
        for item in value.values():
            result.extend(collect_subtrees(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(collect_subtrees(item))
    return result


def _type_tag(value: Any) -> str:
    if isinstance(value, dict) and "op" in value:
        return "ast"
    if isinstance(value, dict):
        return "map"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def anti_unify(left: Any, right: Any) -> Any:
    counter = [0]
    pair_variables: dict[str, str] = {}

    def visit(a: Any, b: Any) -> Any:
        if a == b:
            return deepcopy(a)
        if isinstance(a, dict) and isinstance(b, dict):
            if a.get("op") == b.get("op") and set(a) == set(b):
                return {key: visit(a[key], b[key]) for key in sorted(a)}
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            return [visit(x, y) for x, y in zip(a, b)]
        pair_key = canonical([a, b])
        if pair_key not in pair_variables:
            pair_variables[pair_key] = f"v{counter[0]}_{_type_tag(a)}"
            counter[0] += 1
        return {"$var": pair_variables[pair_key]}

    return visit(left, right)


def match(template: Any, value: Any, bindings: dict[str, Any] | None = None) -> dict[str, Any] | None:
    bindings = {} if bindings is None else dict(bindings)
    if isinstance(template, dict) and set(template) == {"$var"}:
        name = str(template["$var"])
        if name in bindings and bindings[name] != value:
            return None
        bindings[name] = value
        return bindings
    if isinstance(template, dict) and isinstance(value, dict) and set(template) == set(value):
        for key in sorted(template):
            bindings = match(template[key], value[key], bindings)
            if bindings is None:
                return None
        return bindings
    if isinstance(template, list) and isinstance(value, list) and len(template) == len(value):
        for left, right in zip(template, value):
            bindings = match(left, right, bindings)
            if bindings is None:
                return None
        return bindings
    return bindings if template == value else None


def instantiate(template: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(template, dict) and set(template) == {"$var"}:
        return deepcopy(bindings[str(template["$var"])])
    if isinstance(template, dict):
        return {key: instantiate(value, bindings) for key, value in template.items()}
    if isinstance(template, list):
        return [instantiate(value, bindings) for value in template]
    return deepcopy(template)


@dataclass(frozen=True)
class Macro:
    name: str
    template: Any
    occurrences: int
    score: int


def _candidate_score(template: Any, occurrences: int) -> int:
    size = tree_size(template)
    variables = variable_count(template)
    return occurrences * max(0, size - variables - 1) - (size + variables)


def mine_macros(programs: list[AST], *, limit: int = 8) -> list[Macro]:
    subtrees = [subtree for program in programs for subtree in collect_subtrees(program)]
    templates: dict[str, Any] = {}
    for left, right in combinations(subtrees, 2):
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if left.get("op") != right.get("op") or left.get("op") in {"input", "background"}:
            continue
        template = anti_unify(left, right)
        if tree_size(template) < 3:
            continue
        templates[canonical(template)] = template
    candidates: list[Macro] = []
    for template in templates.values():
        occurrences = sum(match(template, subtree) is not None for subtree in subtrees)
        score = _candidate_score(template, occurrences)
        if occurrences < 2 or variable_count(template) == 0 or score <= 0:
            continue
        digest = hashlib.sha256(canonical(template).encode()).hexdigest()[:10]
        candidates.append(Macro(f"induced_{digest}", template, occurrences, score))
    candidates.sort(
        key=lambda macro: (-macro.score, -macro.occurrences, canonical(macro.template))
    )
    selected: list[Macro] = []
    seen_roots: set[str] = set()
    for macro in candidates:
        root = str(macro.template.get("op", ""))
        signature = root + ":" + str(tree_size(macro.template))
        if signature in seen_roots:
            continue
        selected.append(macro)
        seen_roots.add(signature)
        if len(selected) >= limit:
            break
    return selected


def compress(value: Any, macros: list[Macro]) -> Any:
    if isinstance(value, dict) and "op" in value:
        for macro in sorted(macros, key=lambda item: -tree_size(item.template)):
            bindings = match(macro.template, value)
            if bindings is not None:
                return {
                    "op": "macro_call",
                    "name": macro.name,
                    "args": bindings,
                }
        return {key: compress(item, macros) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: compress(item, macros) for key, item in value.items()}
    if isinstance(value, list):
        return [compress(item, macros) for item in value]
    return value


def expand(value: Any, macros: list[Macro]) -> Any:
    library = {macro.name: macro for macro in macros}
    if isinstance(value, dict) and value.get("op") == "macro_call":
        macro = library[str(value["name"])]
        return expand(instantiate(macro.template, value["args"]), macros)
    if isinstance(value, dict):
        return {key: expand(item, macros) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, macros) for item in value]
    return value


