from types import SimpleNamespace

from mini_origin import numeric_threshold_frontier_v70 as core
from mini_origin import response_lattice_closure_v85 as lattice


def records(label_shift=0):
    rows = []
    for index in range(96):
        categorical = ("alpha", "beta", "missing")[index % 3]
        features = (
            str(index % 12),
            categorical,
            "?" if index % 7 == 0 else str(index % 4),
        )
        rows.append((features, str((index + label_shift) % 5)))
    return rows


def dependency_task(label_shift=0):
    """Twelve queries in six exact functional-dependency pairs."""
    row_count = 12
    full_mask = (1 << row_count) - 1
    outcome_masks = []
    for query in range(12):
        base = query % 6
        one = 1 << (6 + base)
        outcome_masks.append((("zero", full_mask ^ one), ("one", one)))
    return SimpleNamespace(
        name="v85-synthetic-zero-parent",
        query_count=12,
        full_mask=full_mask,
        outcome_masks=tuple(outcome_masks),
        rows=tuple((str(index),) for index in range(row_count)),
        labels=tuple(str((index + label_shift) % 3) for index in range(row_count)),
    )


def remap_query_mask(mask, permutation):
    result = 0
    for query, replacement in enumerate(permutation):
        if mask & (1 << query):
            result |= 1 << replacement
    return result


def test_outcome_encoding_is_label_independent_and_keeps_missing():
    first, _ = core.compile_task("v85-encoding", records(0))
    second, _ = core.compile_task("v85-encoding", records(3))
    a = lattice.outcome_atoms(first, first.full_mask)
    b = lattice.outcome_atoms(second, second.full_mask)
    assert [(x.query, x.token, x.mask) for x in a] == [
        (x.query, x.token, x.mask) for x in b
    ]
    assert any(atom.token == "missing" for atom in a)


def test_local_atom_closure_axioms_and_projection_conformance():
    task, _ = core.compile_task("v85-atom-conformance", records())
    atoms = lattice.outcome_atoms(task, task.full_mask)
    choices = [
        frozenset(),
        *(frozenset((i,)) for i in range(min(8, len(atoms)))),
    ]
    for generators in choices:
        closed = lattice.implication_closure(atoms, generators, task.full_mask)
        assert generators <= closed
        assert (
            lattice.implication_closure(atoms, closed, task.full_mask)
            == closed
        )
        projected = lattice.complete_query_projection(atoms, closed)
        for query in range(task.query_count):
            block = {
                i for i, atom in enumerate(atoms)
                if atom.query == query
            }
            assert not (projected & (1 << query)) or block <= closed
    for left in choices:
        for right in choices:
            if left <= right:
                assert lattice.implication_closure(
                    atoms, left, task.full_mask
                ) <= lattice.implication_closure(
                    atoms, right, task.full_mask
                )


def test_query_closure_axioms_exhaustive_on_dependency_lattice():
    task = dependency_task()
    all_queries = (1 << task.query_count) - 1
    closures = {}
    for generators in range(1 << task.query_count):
        closed = lattice.query_closure(task, task.full_mask, generators)
        closures[generators] = closed
        assert generators & ~closed == 0
        assert lattice.query_closure(task, task.full_mask, closed) == closed

    for generators, closed in closures.items():
        for query in range(task.query_count):
            if generators & (1 << query):
                continue
            larger = generators | (1 << query)
            assert closed & ~closures[larger] == 0

    first_six = (1 << 6) - 1
    assert lattice.query_closure(
        task, task.full_mask, first_six
    ) == all_queries


def test_canonical_candidate_ignores_outcome_token_spelling():
    task, _ = core.compile_task("v85-token-renaming", records())
    original = lattice.outcome_atoms(task, task.full_mask)
    renamed = tuple(
        lattice.OutcomeAtom(
            atom.query,
            f"renamed-outcome-{index}",
            atom.mask,
        )
        for index, atom in enumerate(original)
    )
    mask = (1 << task.query_count) - 1
    assert lattice.canonical_candidate(
        task, task.full_mask, mask, 0, original
    ) == lattice.canonical_candidate(
        task, task.full_mask, mask, 0, renamed
    )


def test_canonical_candidate_is_equivariant_to_query_permutation():
    task, _ = core.compile_task("v85-query-permutation", records())
    atoms = lattice.outcome_atoms(task, task.full_mask)
    permutation = tuple(reversed(range(task.query_count)))
    remapped_atoms = tuple(
        lattice.OutcomeAtom(
            permutation[atom.query],
            atom.token,
            atom.mask,
        )
        for atom in reversed(atoms)
    )
    remapped_task = SimpleNamespace(
        query_count=task.query_count,
        full_mask=task.full_mask,
    )
    masks = [
        0,
        (1 << task.query_count) - 1,
        sum(
            1 << query
            for query in range(task.query_count)
            if query % 2 == 0
        ),
        sum(
            1 << query
            for query in range(task.query_count)
            if query % 3 == 1
        ),
    ]
    for candidate in masks:
        for generators in masks:
            original = lattice.canonical_candidate(
                task,
                task.full_mask,
                candidate,
                generators,
                atoms,
            )
            permuted = lattice.canonical_candidate(
                remapped_task,
                task.full_mask,
                remap_query_mask(candidate, permutation),
                remap_query_mask(generators, permutation),
                remapped_atoms,
            )
            assert original == permuted


def test_query_candidate_enumeration_is_exact_minimal_and_deduplicated():
    task = dependency_task()
    all_queries = (1 << task.query_count) - 1
    first = lattice.enumerate_closure_candidates(
        task,
        task.full_mask,
        available_queries=all_queries,
    )
    second = lattice.enumerate_closure_candidates(
        task,
        task.full_mask,
        available_queries=all_queries,
    )
    assert first == second
    assert tuple(item.serialized for item in first) == tuple(
        sorted(item.serialized for item in first)
    )
    assert len({item.digest for item in first}) == len(first)
    assert all(item.candidate_queries for item in first)

    complete = [
        item for item in first
        if item.candidate_queries == all_queries
    ]
    assert len(complete) == 1
    assert complete[0].representatives == 6
    assert complete[0].generator_queries.bit_count() == 6
    assert lattice.query_closure(
        task,
        task.full_mask,
        complete[0].generator_queries,
        available_queries=all_queries,
    ) == all_queries


def test_query_candidate_enumeration_is_label_independent():
    first_task = dependency_task(0)
    second_task = dependency_task(2)
    first = lattice.enumerate_closure_candidates(
        first_task,
        first_task.full_mask,
    )
    second = lattice.enumerate_closure_candidates(
        second_task,
        second_task.full_mask,
    )
    assert tuple(item.serialized for item in first) == tuple(
        item.serialized for item in second
    )
    assert tuple(item.digest for item in first) == tuple(
        item.digest for item in second
    )


def test_query_candidate_enumeration_is_equivariant_to_query_permutation():
    task = dependency_task()
    original = lattice.enumerate_closure_candidates(
        task,
        task.full_mask,
    )
    permutation = tuple(reversed(range(task.query_count)))
    remapped_outcomes = [None] * task.query_count
    for query, replacement in enumerate(permutation):
        remapped_outcomes[replacement] = tuple(
            reversed(task.outcome_masks[query])
        )
    remapped_task = SimpleNamespace(
        name=task.name,
        query_count=task.query_count,
        full_mask=task.full_mask,
        outcome_masks=tuple(remapped_outcomes),
        rows=task.rows,
        labels=task.labels,
    )
    permuted = lattice.enumerate_closure_candidates(
        remapped_task,
        remapped_task.full_mask,
    )
    assert len(original) == len(permuted)
    assert tuple(item.serialized for item in original) == tuple(
        item.serialized for item in permuted
    )
    assert tuple(item.digest for item in original) == tuple(
        item.digest for item in permuted
    )


def test_eligible_candidates_use_only_inherited_thresholds():
    task = dependency_task()
    all_queries = (1 << task.query_count) - 1
    eligible = lattice.eligible_closure_candidates(
        task,
        task.full_mask,
        all_queries,
    )
    assert len(eligible) == 1
    item = eligible[0]
    assert item.candidate_queries == all_queries
    assert item.representatives == lattice.conditioned.MIN_PARTITION_CLASSES
    minimum_raw, minimum_redundancy = (
        lattice.parent.parent.parent.effective_limits(task)
    )
    assert item.candidate_queries.bit_count() >= minimum_raw
    assert (
        item.candidate_queries.bit_count() - item.representatives
        >= minimum_redundancy
    )


def test_synthetic_zero_parent_fallback_constructs_label_free_state(monkeypatch):
    first = dependency_task(0)
    second = dependency_task(2)
    all_queries = (1 << first.query_count) - 1

    def one_cell(task):
        return [(task.full_mask, all_queries, "root")]

    monkeypatch.setattr(
        lattice.conditioned,
        "conditioned_cells",
        one_cell,
    )
    first_rows, first_summary = lattice._closure_fallback_states(first)
    second_rows, second_summary = lattice._closure_fallback_states(second)

    assert first_rows == second_rows
    assert first_rows
    assert first_rows[0] == (first.full_mask, all_queries, 6)
    assert all(remaining == all_queries for _, remaining, _ in first_rows)
    assert all(representatives == 6 for _, _, representatives in first_rows)
    assert first_summary == second_summary
    assert first_summary["conditioned_cells"] == 1
    assert first_summary["structural_candidates"] == len(first_rows)
    assert first_summary["selected_states"] == len(first_rows)
    assert first_summary["response_lattice_fallback"] is True
    assert (
        first_summary["response_lattice_integration"]
        == "zero-parent-functional-dependency-closure"
    )


def test_nonempty_v84_state_set_and_digest_are_exactly_preserved(monkeypatch):
    task, _ = core.compile_task("v85-preserve-contributor", records())
    expected = [
        (task.full_mask, (1 << task.query_count) - 1, 6),
        (
            task.full_mask ^ 1,
            sum(
                1 << query
                for query in range(task.query_count)
                if query % 2
            ),
            3,
        ),
    ]
    digest = lattice.parent.parent.state_set_digest(task, expected)
    parent_summary = {
        "selected_states": len(expected),
        "selected_state_set_digest": digest,
        "selector_revision": "partition-signature-coverage-v84",
        "partition_signature_fallback": False,
    }
    monkeypatch.setattr(
        lattice.parent,
        "select_states",
        lambda _task: (expected, parent_summary),
    )

    actual, summary = lattice.select_states(task)
    assert actual is expected
    assert summary["selected_state_set_digest"] == digest
    assert summary["selected_states"] == len(expected)
    assert summary["selector_revision"] == parent_summary["selector_revision"]
    assert summary["partition_signature_fallback"] is False
    assert summary["response_lattice_fallback"] is False
    assert (
        summary["response_lattice_integration"]
        == "parent-state-set-preserved"
    )


def test_empty_parent_result_activates_only_preregistered_fallback(monkeypatch):
    task = dependency_task()
    all_queries = (1 << task.query_count) - 1
    parent_summary = {
        "selected_states": 0,
        "selected_state_set_digest": (
            lattice.parent.parent.state_set_digest(task, [])
        ),
        "selector_revision": "partition-signature-coverage-v84",
        "partition_signature_fallback": True,
    }
    monkeypatch.setattr(
        lattice.parent,
        "select_states",
        lambda _task: ([], parent_summary),
    )
    monkeypatch.setattr(
        lattice.conditioned,
        "conditioned_cells",
        lambda current: [(current.full_mask, all_queries, "root")],
    )
    rows, summary = lattice.select_states(task)
    assert rows
    assert rows[0] == (task.full_mask, all_queries, 6)
    assert summary["response_lattice_fallback"] is True
    assert (
        summary["response_lattice_integration"]
        == "zero-parent-functional-dependency-closure"
    )
    assert summary["selector_revision"] == "response-lattice-closure-v85"
