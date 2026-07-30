from types import SimpleNamespace

from mini_origin import numeric_threshold_frontier_v70 as core
from mini_origin import response_lattice_closure_v85 as lattice


def records(label_shift=0):
    rows = []
    for index in range(96):
        categorical = ("alpha", "beta", "missing")[index % 3]
        features = (str(index % 12), categorical, "?" if index % 7 == 0 else str(index % 4))
        rows.append((features, str((index + label_shift) % 5)))
    return rows


def test_outcome_encoding_is_label_independent_and_keeps_missing():
    first, _ = core.compile_task("v85-encoding", records(0))
    second, _ = core.compile_task("v85-encoding", records(3))
    a = lattice.outcome_atoms(first, first.full_mask)
    b = lattice.outcome_atoms(second, second.full_mask)
    assert [(x.query, x.token, x.mask) for x in a] == [(x.query, x.token, x.mask) for x in b]
    assert any(atom.token == "missing" for atom in a)


def test_implication_closure_axioms_exhaustive_small_generator_sets():
    task, _ = core.compile_task("v85-axioms", records())
    atoms = lattice.outcome_atoms(task, task.full_mask)
    choices = [frozenset(), *(frozenset((i,)) for i in range(min(8, len(atoms))))]
    for generators in choices:
        closed = lattice.implication_closure(atoms, generators, task.full_mask)
        assert generators <= closed
        assert lattice.implication_closure(atoms, closed, task.full_mask) == closed
    for left in choices:
        for right in choices:
            if left <= right:
                assert lattice.implication_closure(atoms, left, task.full_mask) <= lattice.implication_closure(atoms, right, task.full_mask)


def test_projection_never_emits_partial_query_blocks():
    task, _ = core.compile_task("v85-blocks", records())
    atoms = lattice.outcome_atoms(task, task.full_mask)
    for index in range(len(atoms)):
        closed = lattice.implication_closure(atoms, frozenset((index,)), task.full_mask)
        projected = lattice.complete_query_projection(atoms, closed)
        for query in range(task.query_count):
            block = {i for i, atom in enumerate(atoms) if atom.query == query}
            assert not (projected & (1 << query)) or block <= closed


def test_canonical_candidate_ignores_outcome_token_spelling():
    task, _ = core.compile_task("v85-token-renaming", records())
    original = lattice.outcome_atoms(task, task.full_mask)
    renamed = tuple(
        lattice.OutcomeAtom(atom.query, f"renamed-outcome-{index}", atom.mask)
        for index, atom in enumerate(original)
    )
    mask = (1 << task.query_count) - 1
    assert lattice.canonical_candidate(task, task.full_mask, mask, 0, original) == lattice.canonical_candidate(task, task.full_mask, mask, 0, renamed)


def test_canonical_candidate_is_equivariant_to_query_permutation():
    task, _ = core.compile_task("v85-query-permutation", records())
    atoms = lattice.outcome_atoms(task, task.full_mask)
    permutation = tuple(reversed(range(task.query_count)))
    remapped_atoms = tuple(
        lattice.OutcomeAtom(permutation[atom.query], atom.token, atom.mask)
        for atom in reversed(atoms)
    )
    remapped_task = SimpleNamespace(query_count=task.query_count, full_mask=task.full_mask)

    def remap(mask):
        result = 0
        for query, replacement in enumerate(permutation):
            if mask & (1 << query):
                result |= 1 << replacement
        return result

    masks = [
        0,
        (1 << task.query_count) - 1,
        sum(1 << query for query in range(task.query_count) if query % 2 == 0),
        sum(1 << query for query in range(task.query_count) if query % 3 == 1),
    ]
    for candidate in masks:
        for generators in masks:
            original = lattice.canonical_candidate(task, task.full_mask, candidate, generators, atoms)
            permuted = lattice.canonical_candidate(
                remapped_task,
                task.full_mask,
                remap(candidate),
                remap(generators),
                remapped_atoms,
            )
            assert original == permuted


def test_candidate_enumeration_is_deterministic_minimal_and_deduplicated():
    task, _ = core.compile_task("v85-enumeration", records())
    first = lattice.enumerate_closure_candidates(task, task.full_mask)
    second = lattice.enumerate_closure_candidates(task, task.full_mask)
    assert first == second
    assert tuple(item.serialized for item in first) == tuple(sorted(item.serialized for item in first))
    assert len({item.digest for item in first}) == len(first)
    assert all(item.candidate_queries for item in first)
    assert all(len(item.generator_atoms) <= 2 for item in first)

    atoms = lattice.outcome_atoms(task, task.full_mask)
    structures = []
    for item in first:
        closed = lattice.implication_closure(atoms, frozenset(item.generator_atoms), task.full_mask)
        assert lattice.complete_query_projection(atoms, closed) == item.candidate_queries
        structures.append(lattice._candidate_structure_key(task, task.full_mask, item.candidate_queries, atoms))
    assert len(structures) == len(set(structures))


def test_candidate_enumeration_is_label_independent():
    first_task, _ = core.compile_task("v85-enumeration-labels", records(0))
    second_task, _ = core.compile_task("v85-enumeration-labels", records(4))
    first = lattice.enumerate_closure_candidates(first_task, first_task.full_mask)
    second = lattice.enumerate_closure_candidates(second_task, second_task.full_mask)
    assert tuple(item.serialized for item in first) == tuple(item.serialized for item in second)
    assert tuple(item.digest for item in first) == tuple(item.digest for item in second)


def test_candidate_enumeration_is_equivariant_to_query_and_atom_permutation():
    task, _ = core.compile_task("v85-enumeration-permutation", records())
    original = lattice.enumerate_closure_candidates(task, task.full_mask)

    remapped_outcomes = tuple(
        tuple(reversed(task.outcome_masks[query]))
        for query in reversed(range(task.query_count))
    )
    remapped_task = SimpleNamespace(
        query_count=task.query_count,
        full_mask=task.full_mask,
        outcome_masks=remapped_outcomes,
    )
    permuted = lattice.enumerate_closure_candidates(remapped_task, remapped_task.full_mask)

    assert len(original) == len(permuted)
    assert tuple(item.serialized for item in original) == tuple(item.serialized for item in permuted)
    assert tuple(item.digest for item in original) == tuple(item.digest for item in permuted)


def test_nonempty_v84_state_set_and_digest_are_exactly_preserved(monkeypatch):
    task, _ = core.compile_task("v85-preserve-contributor", records())
    expected = [
        (task.full_mask, (1 << task.query_count) - 1, 6),
        (task.full_mask ^ 1, sum(1 << q for q in range(task.query_count) if q % 2), 3),
    ]
    digest = lattice.parent.parent.state_set_digest(task, expected)
    parent_summary = {
        "selected_states": len(expected),
        "selected_state_set_digest": digest,
        "selector_revision": "partition-signature-coverage-v84",
        "partition_signature_fallback": False,
    }
    monkeypatch.setattr(lattice.parent, "select_states", lambda _task: (expected, parent_summary))

    actual, summary = lattice.select_states(task)
    assert actual is expected
    assert summary["selected_state_set_digest"] == digest
    assert summary["selected_states"] == len(expected)
    assert summary["selector_revision"] == parent_summary["selector_revision"]
    assert summary["partition_signature_fallback"] is False
    assert summary["response_lattice_fallback"] is False
    assert summary["response_lattice_integration"] == "synthetic-preservation-only"
