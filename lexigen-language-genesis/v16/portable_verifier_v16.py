from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from portable_ir_runtime_v15 import execute_portable_ir

Grid = tuple[tuple[int, ...], ...]
CATALOG = (
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


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


GRAMMAR_SHA256 = _sha(CATALOG)


def _grid(value: Any) -> Grid:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("invalid grid")
    return grid


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _flat(grid: Grid) -> tuple[int, ...]:
    return tuple(cell for row in grid for cell in row)


def _mode(grid: Grid) -> int:
    counts = Counter(_flat(grid))
    return min(counts, key=lambda colour: (-counts[colour], colour))


def _border(grid: Grid) -> tuple[int, ...]:
    height, width = _shape(grid)
    values = list(grid[0])
    if height > 1:
        values.extend(grid[-1])
    for row in range(1, height - 1):
        values.append(grid[row][0])
        if width > 1:
            values.append(grid[row][-1])
    return tuple(values)


def _digest(values: tuple[int, ...]) -> str:
    return hashlib.sha256(_canonical(values).encode("utf-8")).hexdigest()


def _rows(grid: Grid) -> tuple[str, ...]:
    return tuple(_digest(tuple(row)) for row in grid)


def _columns(grid: Grid) -> tuple[str, ...]:
    height, width = _shape(grid)
    return tuple(
        _digest(tuple(grid[row][column] for row in range(height)))
        for column in range(width)
    )


def _checksum(grid: Grid, rw: int, cw: int, prime: int) -> int:
    total = 0
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            total = (
                total
                + ((row + 1) * rw + (column + 1) * cw) * (int(value) + 1)
            ) % prime
    return total


def _predicate(name: str, source: Grid, candidate: Grid, reference: Grid) -> bool:
    if name == "shape":
        return _shape(candidate) == _shape(reference)
    if _shape(candidate) != _shape(reference):
        return False
    if name == "palette":
        return set(_flat(candidate)) == set(_flat(reference))
    if name == "histogram":
        return Counter(_flat(candidate)) == Counter(_flat(reference))
    if name == "boundary":
        return _border(candidate) == _border(reference)
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
        return _checksum(candidate, 17, 31, 1_000_003) == _checksum(reference, 17, 31, 1_000_003)
    if name == "checksum_2":
        return _checksum(candidate, 43, 71, 1_000_033) == _checksum(reference, 43, 71, 1_000_033)
    if name == "row_digests":
        return _rows(candidate) == _rows(reference)
    if name == "column_digests":
        return _columns(candidate) == _columns(reference)
    if name == "exact_digest":
        return _sha(candidate) == _sha(reference)
    raise ValueError(f"unknown verifier predicate: {name}")


def _payload(contract: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in contract.items() if key != "contract_sha256"}


def screening_holds_portable(
    contract: dict[str, Any], source: Any, candidate: Any, reference: Any
) -> bool:
    source_grid = _grid(source)
    candidate_grid = _grid(candidate)
    reference_grid = _grid(reference)
    return all(
        _predicate(str(predicate["name"]), source_grid, candidate_grid, reference_grid)
        for predicate in contract.get("predicates", [])
    )


def verify_against_reference_portable(
    contract: dict[str, Any],
    program: dict[str, Any],
    source: Any,
    candidate: Any,
    reference: Any,
) -> bool:
    if contract.get("grammar_sha256") != GRAMMAR_SHA256:
        return False
    if contract.get("contract_sha256") != _sha(_payload(contract)):
        return False
    if contract.get("program_sha256") != _sha(program):
        return False
    if contract.get("soundness_anchor") != {"name": "exact_digest", "mandatory": True}:
        return False
    if not screening_holds_portable(contract, source, candidate, reference):
        return False
    return _predicate("exact_digest", _grid(source), _grid(candidate), _grid(reference))


def verify_output_portable(
    contract: dict[str, Any], program: dict[str, Any], source: Any, candidate: Any
) -> bool:
    reference = execute_portable_ir(program, source)
    return verify_against_reference_portable(contract, program, source, candidate, reference)
