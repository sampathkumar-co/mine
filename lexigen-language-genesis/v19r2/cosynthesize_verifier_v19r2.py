from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from runtime_v19r2 import Program, Grid, as_grid, execute, sha256_json
from mutations_v19r2 import apply_output_mutation, canonical, mutation_manifest
from verifier_grammar_v19r2 import (
    VERIFIER_GRAMMAR_SHA256,
    PREDICATE_CATALOG,
    contract_payload,
    predicate_holds,
    sha256_json,
)

MutationCase = dict[str, Any]


def build_mutation_cases(
    program: Program,
    examples: list[tuple[Grid, Grid]],
    *,
    mutation_limit: int = 64,
) -> tuple[list[MutationCase], list[dict[str, Any]]]:
    manifest = mutation_manifest(program, limit=mutation_limit)
    cases: list[MutationCase] = []
    runtime_invalid = 0
    for example_index, (source, _) in enumerate(examples):
        reference = execute(program, source)
        for mutation in manifest:
            try:
                if mutation["kind"] == "ast":
                    candidate = execute(mutation["ast"], source)
                else:
                    candidate = apply_output_mutation(reference, str(mutation["operator"]))
            except Exception:
                runtime_invalid += 1
                continue
            if candidate == reference:
                continue
            cases.append(
                {
                    "example_index": example_index,
                    "mutation_sha256": mutation["mutation_sha256"],
                    "source": as_grid(source),
                    "candidate": candidate,
                    "reference": reference,
                }
            )
    return cases, manifest, runtime_invalid


def _coverage(predicate_name: str, cases: list[MutationCase]) -> frozenset[int]:
    return frozenset(
        index
        for index, case in enumerate(cases)
        if not predicate_holds(
            predicate_name,
            case["source"],
            case["candidate"],
            case["reference"],
        )
    )


def _minimum_cover(cases: list[MutationCase]) -> list[dict[str, Any]]:
    if not cases:
        raise RuntimeError("no valid semantic mutations were generated")
    universe = frozenset(range(len(cases)))
    catalog = [dict(item) for item in PREDICATE_CATALOG if item["name"] != "exact_digest"]
    coverages = {item["name"]: _coverage(str(item["name"]), cases) for item in catalog}
    best: tuple[tuple[Any, ...], list[dict[str, Any]]] | None = None
    for size in range(1, len(catalog) + 1):
        for selected in combinations(catalog, size):
            covered: frozenset[int] = frozenset().union(
                *(coverages[str(item["name"])] for item in selected)
            )
            if covered != universe:
                continue
            names = tuple(sorted(str(item["name"]) for item in selected))
            score = (sum(int(item["cost"]) for item in selected), size, names)
            if best is None or score < best[0]:
                best = (score, [dict(item) for item in selected])
        if best is not None and best[0][1] == size:
            minimum_cost_at_size = best[0][0]
            if minimum_cost_at_size <= size:
                break
    if best is None:
        raise RuntimeError("verifier grammar cannot cover the mutation bank")
    return sorted(best[1], key=lambda item: str(item["name"]))


def count_survivors(predicates: Iterable[dict[str, Any]], cases: list[MutationCase]) -> int:
    names = [str(item["name"]) for item in predicates]
    return sum(
        all(
            predicate_holds(name, case["source"], case["candidate"], case["reference"])
            for name in names
        )
        for case in cases
    )


def synthesize_contract(
    production: dict[str, Any],
    concrete_program: Program,
    examples: list[tuple[Grid, Grid]],
    *,
    primary_runtime_sha256: str,
    portable_runtime_sha256: str,
    demonstration_sha256: str,
    additional_cases: list[MutationCase] | None = None,
    mutation_limit: int = 64,
    revision: int = 0,
) -> tuple[dict[str, Any], list[MutationCase], list[dict[str, Any]]]:
    cases, manifest, runtime_invalid = build_mutation_cases(
        concrete_program, examples, mutation_limit=mutation_limit
    )
    if additional_cases:
        seen = {
            (case["example_index"], case["mutation_sha256"], canonical(case["candidate"]))
            for case in cases
        }
        for case in additional_cases:
            key = (
                case.get("example_index", -1),
                case["mutation_sha256"],
                canonical(case["candidate"]),
            )
            if key not in seen:
                cases.append(case)
                seen.add(key)
    predicates = _minimum_cover(cases)
    if any(item["name"] == "exact_digest" for item in predicates):
        raise RuntimeError("learned screen selected exact digest")
    contract: dict[str, Any] = {
        "schema": "lexigen-v19r2-verifier-contract-v1",
        "revision": int(revision),
        "production_sha256": sha256_json(production),
        "concrete_program_sha256": sha256_json(concrete_program),
        "primary_runtime_sha256": primary_runtime_sha256,
        "portable_runtime_sha256": portable_runtime_sha256,
        "demonstration_sha256": demonstration_sha256,
        "verifier_grammar_sha256": VERIFIER_GRAMMAR_SHA256,
        "predicates": predicates,
        "predicate_cost": sum(int(item["cost"]) for item in predicates),
        "training_mutation_cases": len(cases),
        "training_runtime_invalid_mutations": runtime_invalid,
        "mutation_manifest_sha256": sha256_json([item["mutation_sha256"] for item in manifest]),
        "exact_digest_used": False,
        "soundness_anchor": {"name": "exact_digest", "mandatory": True},
        "shape_only_survivors": count_survivors([{"name": "shape", "cost": 1}], cases),
    }
    contract["contract_sha256"] = sha256_json(contract_payload(contract))
    return contract, cases, manifest
