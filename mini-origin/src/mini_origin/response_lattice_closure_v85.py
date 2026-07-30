from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class OutcomeAtom:
    query: int
    token: str
    mask: int


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
