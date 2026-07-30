from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

from mini_origin import partition_signature_coverage_v84 as parent
from mini_origin import conditioned_cell_frontier_v60 as conditioned
from mini_origin import label_free_selector_certificate_v71 as label_free_selector


PREREGISTRATION = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v85-response-lattice-closure.json"
)
IMPLEMENTATION_AMENDMENT = (
    Path(__file__).resolve().parents[2]
    / "campaigns"
    / "v85-response-lattice-implementation-amendment.json"
)
compact_state = parent.compact_state


@dataclass(frozen=True)
class OutcomeAtom:
    query: int
    token: str
    mask: int


@dataclass(frozen=True)
class ClosureCandidate:
    candidate_queries: int
    generator_queries: int
    representatives: int
    serialized: bytes
    digest: str


def outcome_atoms(task: object, allowed: int) -> tuple[OutcomeAtom, ...]:
    """Encode every compiler-emitted outcome one-vs-rest on allowed rows."""
    atoms = []
    for query, outcomes in enumerate(task.outcome_masks):
        for token, mask in outcomes:
            atoms.append(OutcomeAtom(query, token, mask & allowed))
    return tuple(atoms)


def implication_closure(
    atoms: tuple[OutcomeAtom, ...],
    generators: frozenset[int],
    allowed: int,
) -> frozenset[int]:
    """Local atom-conjunction closure used by the encoding conformance tests."""
    support = allowed
    for index in generators:
        support &= atoms[index].mask
    return frozenset(
        index for index, atom in enumerate(atoms)
        if support & ~atom.mask == 0
    )


def complete_query_projection(
    atoms: tuple[OutcomeAtom, ...],
    closed: frozenset[int],
) -> int:
    """Project only complete outcome blocks; partial query blocks are forbidden."""
    blocks: dict[int, set[int]] = {}
    for index, atom in enumerate(atoms):
        blocks.setdefault(atom.query, set()).add(index)
    result = 0
    for query, block in blocks.items():
        if block and block <= closed:
            result |= 1 << query
    return result


def block_signature(
    atoms: tuple[OutcomeAtom, ...],
    query: int,
    allowed: int,
) -> tuple[int, ...]:
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


def _query_indices(mask: int) -> tuple[int, ...]:
    result = []
    pending = mask
    while pending:
        bit = pending & -pending
        result.append(bit.bit_length() - 1)
        pending ^= bit
    return tuple(result)


def query_partition(task: object, allowed: int, query: int) -> tuple[int, ...]:
    """Return the token-independent nonempty response partition for one query."""
    cells = []
    seen = 0
    for _, mask in task.outcome_masks[query]:
        cell = mask & allowed
        if not cell:
            continue
        if seen & cell:
            raise ValueError("query outcomes overlap on allowed rows")
        seen |= cell
        cells.append(cell)
    if seen != allowed:
        raise ValueError("query outcomes do not cover allowed rows")
    return tuple(sorted(cells))


def joint_partition(
    task: object,
    allowed: int,
    generator_queries: int,
) -> tuple[int, ...]:
    """Common refinement induced by complete generator-query blocks."""
    cells = (allowed,)
    for query in _query_indices(generator_queries):
        outcomes = query_partition(task, allowed, query)
        refined = {
            cell & outcome
            for cell in cells
            for outcome in outcomes
            if cell & outcome
        }
        cells = tuple(sorted(refined))
    return cells


def partition_refines(
    fine: tuple[int, ...],
    coarse: tuple[int, ...],
) -> bool:
    """Whether every fine cell is contained in one coarse response cell."""
    return all(
        any(cell & ~block == 0 for block in coarse)
        for cell in fine
    )


def query_closure(
    task: object,
    allowed: int,
    generator_queries: int,
    *,
    available_queries: int | None = None,
) -> int:
    """Functional-dependency closure of complete query blocks.

    A query belongs to the closure exactly when the joint response partition of
    the generators refines that query's response partition. This captures
    compositional response relations without selecting a privileged outcome.
    """
    if allowed & ~task.full_mask:
        raise ValueError("allowed rows exceed task full mask")
    all_queries = (1 << task.query_count) - 1
    available = all_queries if available_queries is None else available_queries
    if available & ~all_queries:
        raise ValueError("available queries exceed task query count")
    if generator_queries & ~available:
        raise ValueError("generator queries must be available")
    fine = joint_partition(task, allowed, generator_queries)
    result = 0
    for query in _query_indices(available):
        if partition_refines(fine, query_partition(task, allowed, query)):
            result |= 1 << query
    return result


def _candidate_structure_key(
    task: object,
    allowed: int,
    candidate: int,
    atoms: tuple[OutcomeAtom, ...],
) -> bytes:
    blocks = sorted(
        block_signature(atoms, query, allowed)
        for query in range(task.query_count)
        if candidate & (1 << query)
     )
    return json.dumps(
        (task.full_mask.bit_count(), allowed, blocks),
        separators=(",", ":"),
    ).encode("utf-8")


def enumerate_closure_candidates(
    task: object,
    allowed: int,
    *,
    available_queries: int | None = None,
) -> tuple[ClosureCandidate, ...]:
    """Enumerate exact minimal generators of query-level response closures.

    Exhaustive enumeration is bounded by the unchanged maximum partition-class
    threshold. Wider available sets are left untouched rather than introducing
    a new heuristic or index-dependent truncation rule.
    """
    all_queries = (1 << task.query_count) - 1
    available = all_queries if available_queries is None else available_queries
    if available & ~all_queries:
        raise ValueError("available queries exceed task query count")
    query_ids = _query_indices(available)
    if len(query_ids) > conditioned.MAX_PARTITION_CLASSES:
        return ()

    atoms = outcome_atoms(task, allowed)
    best: dict[bytes, tuple[int, bytes, int, int]] = {}
    for size in range(len(query_ids) + 1):
        for selected in itertools.combinations(query_ids, size):
            generator_queries = sum(1 << query for query in selected)
            candidate = query_closure(
                task,
                allowed,
                generator_queries,
                available_queries=available,
            )
            if candidate == 0:
                continue
            serialized = canonical_candidate(
                task,
                allowed,
                candidate,
                generator_queries,
                atoms,
            )
            structure = _candidate_structure_key(
                task,
                allowed,
                candidate,
                atoms,
            )
            rank = (size, serialized, generator_queries, candidate)
            previous = best.get(structure)
            if previous is None or rank[:2] < previous[:2]:
                best[structure] = rank

    result = []
    for structure in sorted(best):
        representatives, serialized, generator_queries, candidate = best[structure]
        result.append(
            ClosureCandidate(
                candidate_queries=candidate,
                generator_queries=generator_queries,
                representatives=representatives,
                serialized=serialized,
                digest=candidate_digest(serialized),
            )
        )
    return tuple(sorted(result, key=lambda item: item.serialized))


def eligible_closure_candidates(
    task: object,
    allowed: int,
    available_queries: int,
) -> tuple[ClosureCandidate, ...]:
    """Apply only the inherited raw, redundancy, and partition-class gates."""
    minimum_raw, minimum_redundancy = parent.parent.parent.effective_limits(task)
    result = []
    for item in enumerate_closure_candidates(
        task,
        allowed,
        available_queries=available_queries,
    ):
        raw = item.candidate_queries.bit_count()
        if (
            conditioned.MIN_PARTITION_CLASSES
            <= item.representatives
            <= conditioned.MAX_PARTITION_CLASSES
            and minimum_raw <= raw <= conditioned.MAX_RAW_QUERIES
            and raw - item.representatives >= minimum_redundancy
         ):
            result.append(item)
    return tuple(result)


def _allowed_variants(task: object, cell: int, path: str) -> tuple[int, ...]:
    cell_size = cell.bit_count()
    variants = []
    if 8 <= cell_size <= 24:
        variants.append(cell)
    coverage = parent.parent.parent
    for size in coverage.SMALL_QUERY_SAMPLE_SIZES:
        if cell_size < size:
            continue
        for seed in conditioned.PATH_SEEDS[: coverage.SMALL_QUERY_ALLOWED_SEEDS]:
            variants.append(
                label_free_selector.label_free_sample_allowed(
                    task,
                    cell,
                    size,
                    f"{path}:{seed}:{size}",
                )
            )
    return tuple(sorted(set(variants)))


def _closure_fallback_states(task: object):
    """Synthetic-preflight constructor; not connected to select_states yet."""
    cells = conditioned.conditioned_cells(task)
    candidates: dict[tuple[int, int], tuple[int, bytes]] = {}
    for cell, path_remaining, path in cells:
        for allowed in _allowed_variants(task, cell, path):
            for item in eligible_closure_candidates(
                task,
                allowed,
                path_remaining,
            ):
                key = (allowed, item.candidate_queries)
                rank = (item.representatives, item.serialized)
                previous = candidates.get(key)
                if previous is None or rank < previous:
                    candidates[key] = rank

    ranked = [
        (allowed, remaining, representatives, serialized)
        for (allowed, remaining), (representatives, serialized)
        in candidates.items()
    ]
    ranked.sort(
        key=lambda row: (
            -row[0].bit_count(),
            -(row[1].bit_count() - row[2]),
            -row[1].bit_count(),
            row[3],
        )
    )
    rows = [
        (allowed, remaining, representatives)
        for allowed, remaining, representatives, _
        in ranked[: conditioned.MAX_STATES_PER_TASK]
    ]
    minimum_raw, minimum_redundancy = parent.parent.parent.effective_limits(task)
    return rows, {
        "conditioned_cells": len(cells),
        "structural_candidates": len(candidates),
        "selected_states": len(rows),
        "response_lattice_fallback": True,
        "response_lattice_integration": "zero-parent-functional-dependency-closure",
        "effective_min_raw_queries": minimum_raw,
        "effective_min_redundancy": minimum_redundancy,
        "selected_state_set_digest": parent.parent.state_set_digest(task, rows),
        "selector_revision": "response-lattice-closure-v85",
    }


def select_states(task: object):
    """Preserve nonempty v0.84 results; repair only exact zero-state cases."""
    states, summary = parent.select_states(task)
    if states:
        preserved = dict(summary)
        preserved["response_lattice_fallback"] = False
        preserved["response_lattice_integration"] = "parent-state-set-preserved"
        return states, preserved
    return _closure_fallback_states(task)


def protocol() -> dict[str, object]:
    result = dict(parent.protocol())
    result["state_selector"] = (
        "v0.84 selector unchanged for contributing tasks; complete-query "
        "functional-dependency closure is used only when v0.84 yields zero states"
    )
    result["response_lattice_fallback"] = "zero-state-only"
    result["response_lattice_closure"] = (
        "joint generator partition refines each determined query partition"
    )
    result["development_data_status"] = "opened-but-not-accessed-by-v0.85-yet"
    return result


def install_v85_components() -> None:
    parent.configure_module()
    parent.frontier.protocol = protocol
    conditioned.select_states = select_states


def configure_module() -> None:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    amendment = json.loads(IMPLEMENTATION_AMENDMENT.read_text(encoding="utf-8"))
    if preregistration["status"] != "preregistered_before_implementation_or_dataset_evaluation":
        raise RuntimeError("v0.85 preregistration status changed")
    if amendment["status"] != "implementation_amendment_before_selector_integration_or_opened_data_evaluation":
        raise RuntimeError("v0.85 implementation amendment status changed")
    install_v85_components()
    parent.frontier.configure_module = install_v85_components
