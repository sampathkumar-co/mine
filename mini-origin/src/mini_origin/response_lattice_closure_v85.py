from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json

from mini_origin import partition_signature_coverage_v84 as parent


@dataclass(frozen=True)
class OutcomeAtom:
    query: int
    token: str
    mask: int


@dataclass(frozen=True)
class ClosureCandidate:
    candidate_queries: int
    generator_queries: int
    generator_atoms: tuple[int, ...]
    serialized: bytes
    digest: str


def outcome_atoms(task: object, allowed: int) -> tuple[OutcomeAtom, ...]:
    """Encode every compiler-emitted outcome one-vs-rest on allowed rows."""
    atoms = []
    for query, outcomes in enumerate(task.outcome_masks):
        for token, mask in outcomes:
            atoms.append(OutcomeAtom(query, token, mask & allowed))
    return tuple(atoms)


def implication_closure(atoms: tuple[OutcomeAtom, ...], generators: frozenset[int], allowed: int) -> frozenset[int]:
    """Exact finite closure: include each atom true on every generator-compatible row."""
    support = allowed
    for index in generators:
        support &= atoms[index].mask
    return frozenset(
        index for index, atom in enumerate(atoms)
        if support & ~atom.mask == 0
    )


def complete_query_projection(atoms: tuple[OutcomeAtom, ...], closed: frozenset[int]) -> int:
    """Project only complete outcome blocks; partial query blocks are forbidden."""
    blocks: dict[int, set[int]] = {}
    for index, atom in enumerate(atoms):
        blocks.setdefault(atom.query, set()).add(index)
    result = 0
    for query, block in blocks.items():
        if block and block <= closed:
            result |= 1 << query
    return result


def block_signature(atoms: tuple[OutcomeAtom, ...], query: int, allowed: int) -> tuple[int, ...]:
    """Token-independent canonical signature for one complete query block."""
    return tuple(sorted(atom.mask & allowed for atom in atoms if atom.query == query))


def canonical_candidate(
    task: object,
    allowed: int,
    candidate: int,
    generator_queries: int,
    atoms: tuple[OutcomeAtom, ...],
) -> bytes:
    """Index-free serialization used for equivariant deduplication and ordering."""
    candidate_blocks = sorted(
        block_signature(atoms, query, allowed)
        for query in range(task.query_count)
        if candidate & (1 << query)
    )
    generator_blocks = sorted(
        block_signature(atoms, query, allowed)
        for query in range(task.query_count)
        if generator_queries & (1 << query)
    )
    payload = (
        task.full_mask.bit_count(),
        allowed,
        candidate_blocks,
        generator_blocks,
    )
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def candidate_digest(serialized: bytes) -> str:
    return hashlib.sha256(serialized).hexdigest()


def _query_mask(atoms: tuple[OutcomeAtom, ...], generator_atoms: tuple[int, ...]) -> int:
    result = 0
    for index in generator_atoms:
        result |= 1 << atoms[index].query
    return result


def _candidate_structure_key(task: object, allowed: int, candidate: int, atoms: tuple[OutcomeAtom, ...]) -> bytes:
    blocks = sorted(
        block_signature(atoms, query, allowed)
        for query in range(task.query_count)
        if candidate & (1 << query)
    )
    return json.dumps((task.full_mask.bit_count(), allowed, blocks), separators=(",", ":")).encode("utf-8")


def enumerate_closure_candidates(
    task: object,
    allowed: int,
    *,
    max_generator_atoms: int = 2,
) -> tuple[ClosureCandidate, ...]:
    """Enumerate deterministic minimal-generator closure candidates.

    This pure helper is synthetic-only. It considers the empty generator and
    atom generator sets up to the preregistered bounded order, projects only
    complete query blocks, and keeps the smallest canonical generator for each
    index-free candidate structure.
    """
    if max_generator_atoms < 0:
        raise ValueError("max_generator_atoms must be non-negative")
    atoms = outcome_atoms(task, allowed)
    best: dict[bytes, tuple[int, bytes, tuple[int, ...], int]] = {}
    for size in range(min(max_generator_atoms, len(atoms)) + 1):
        for generator_atoms in itertools.combinations(range(len(atoms)), size):
            closed = implication_closure(atoms, frozenset(generator_atoms), allowed)
            candidate = complete_query_projection(atoms, closed)
            if candidate == 0:
                continue
            generator_queries = _query_mask(atoms, generator_atoms)
            serialized = canonical_candidate(task, allowed, candidate, generator_queries, atoms)
            structure = _candidate_structure_key(task, allowed, candidate, atoms)
            rank = (size, serialized, generator_atoms, generator_queries)
            previous = best.get(structure)
            if previous is None or rank < previous:
                best[structure] = rank

    result = []
    for structure in sorted(best):
        _, serialized, generator_atoms, generator_queries = best[structure]
        closed = implication_closure(atoms, frozenset(generator_atoms), allowed)
        candidate = complete_query_projection(atoms, closed)
        result.append(
            ClosureCandidate(
                candidate_queries=candidate,
                generator_queries=generator_queries,
                generator_atoms=generator_atoms,
                serialized=serialized,
                digest=candidate_digest(serialized),
            )
        )
    return tuple(sorted(result, key=lambda item: item.serialized))


def select_states(task: object):
    """Inactive adapter: preserve every nonempty v0.84 result exactly.

    Response-lattice fallback generation is intentionally not connected yet.
    Empty parent results remain empty until the remaining synthetic gates pass.
    """
    states, summary = parent.select_states(task)
    preserved = dict(summary)
    preserved["response_lattice_fallback"] = False
    preserved["response_lattice_integration"] = "synthetic-preservation-only"
    return states, preserved
