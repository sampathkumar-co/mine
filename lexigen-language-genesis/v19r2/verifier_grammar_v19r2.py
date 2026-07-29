from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from runtime_v19r2 import Grid, as_grid, execute, sha256_json

PREDICATE_CATALOG = (
    {"name": "shape", "cost": 1},
    {"name": "palette", "cost": 2},
    {"name": "histogram", "cost": 3},
    {"name": "boundary", "cost": 4},
    {"name": "foreground_mask", "cost": 5},
    {"name": "changed_mask", "cost": 5},
    {"name": "checksum_1", "cost": 6},
    {"name": "checksum_2", "cost": 6},
    {"name": "row_digests", "cost": 8},
    {"name": "column_digests", "cost": 8},
    {"name": "exact_digest", "cost": 30},
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


VERIFIER_GRAMMAR_SHA256 = _sha(PREDICATE_CATALOG)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _flat(grid: Grid) -> tuple[int, ...]:
    return tuple(cell for row in grid for cell in row)


def _mode(grid: Grid) -> int:
    counts = Counter(_flat(grid))
    return min(counts, key=lambda colour: (-counts[colour], colour))


def _boundary(grid: Grid) -> tuple[int, ...]:
    height, width = _shape(grid)
    result = list(grid[0])
    if height > 1:
        result.extend(grid[-1])
    for row in range(1, height - 1):
        result.append(grid[row][0])
        if width > 1:
            result.append(grid[row][-1])
    return tuple(result)


def _digest_sequence(values: tuple[int, ...]) -> str:
    return hashlib.sha256(canonical(values).encode("utf-8")).hexdigest()


def _row_digests(grid: Grid) -> tuple[str, ...]:
    return tuple(_digest_sequence(tuple(row)) for row in grid)


def _column_digests(grid: Grid) -> tuple[str, ...]:
    height, width = _shape(grid)
    return tuple(
        _digest_sequence(tuple(grid[row][column] for row in range(height)))
        for column in range(width)
    )


def _checksum(grid: Grid, *, row_weight: int, column_weight: int, prime: int) -> int:
    total = 0
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            coefficient = (row + 1) * row_weight + (column + 1) * column_weight
            total = (total + coefficient * (int(value) + 1)) % prime
    return total


def predicate_holds(name: str, source: Grid, candidate: Grid, reference: Grid) -> bool:
    source, candidate, reference = as_grid(source), as_grid(candidate), as_grid(reference)
    if name == "shape":
        return _shape(candidate) == _shape(reference)
    if _shape(candidate) != _shape(reference):
        return False
    if name == "palette":
        return set(_flat(candidate)) == set(_flat(reference))
    if name == "histogram":
        return Counter(_flat(candidate)) == Counter(_flat(reference))
    if name == "boundary":
        return _boundary(candidate) == _boundary(reference)
    if name == "foreground_mask":
        background = _mode(source)
        return tuple(value != background for value in _flat(candidate)) == tuple(
            value != background for value in _flat(reference)
        )
    if name == "changed_mask":
        if _shape(source) != _shape(candidate) or _shape(source) != _shape(reference):
            return True
        return tuple(a != b for a, b in zip(_flat(source), _flat(candidate))) == tuple(
            a != b for a, b in zip(_flat(source), _flat(reference))
        )
    if name == "checksum_1":
        return _checksum(candidate, row_weight=17, column_weight=31, prime=1000003) == _checksum(
            reference, row_weight=17, column_weight=31, prime=1000003
        )
    if name == "checksum_2":
        return _checksum(candidate, row_weight=43, column_weight=71, prime=1000033) == _checksum(
            reference, row_weight=43, column_weight=71, prime=1000033
        )
    if name == "row_digests":
        return _row_digests(candidate) == _row_digests(reference)
    if name == "column_digests":
        return _column_digests(candidate) == _column_digests(reference)
    if name == "exact_digest":
        return sha256_json(candidate) == sha256_json(reference)
    raise ValueError(f"unknown verifier predicate: {name}")


def contract_payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "contract_sha256"}


def verify_contract_integrity(contract: dict[str, Any]) -> bool:
    required_hashes = (
        "production_sha256",
        "concrete_program_sha256",
        "primary_runtime_sha256",
        "portable_runtime_sha256",
        "demonstration_sha256",
        "verifier_grammar_sha256",
    )
    return (
        contract.get("schema") == "lexigen-v19r2-verifier-contract-v1"
        and contract.get("verifier_grammar_sha256") == VERIFIER_GRAMMAR_SHA256
        and all(isinstance(contract.get(name), str) and len(contract[name]) == 64 for name in required_hashes)
        and contract.get("contract_sha256") == _sha(contract_payload(contract))
    )


def screening_holds(
    contract: dict[str, Any], source: Grid, candidate: Grid, reference: Grid
) -> bool:
    return all(
        predicate_holds(
            str(predicate["name"]),
            as_grid(source),
            as_grid(candidate),
            as_grid(reference),
        )
        for predicate in contract.get("predicates", [])
    )


def verify_against_reference(
    contract: dict[str, Any],
    production: dict[str, Any],
    concrete_program: dict[str, Any],
    source: Grid,
    candidate: Grid,
    reference: Grid,
) -> bool:
    if not verify_contract_integrity(contract):
        return False
    if contract.get("production_sha256") != sha256_json(production):
        return False
    if contract.get("concrete_program_sha256") != sha256_json(concrete_program):
        return False
    if contract.get("soundness_anchor") != {"name": "exact_digest", "mandatory": True}:
        return False
    if not screening_holds(contract, source, candidate, reference):
        return False
    return predicate_holds("exact_digest", source, candidate, reference)


def verify_output(
    contract: dict[str, Any],
    production: dict[str, Any],
    concrete_program: dict[str, Any],
    source: Grid,
    candidate: Grid,
) -> bool:
    reference = execute(concrete_program, source)
    return verify_against_reference(
        contract, production, concrete_program, source, candidate, reference
    )
