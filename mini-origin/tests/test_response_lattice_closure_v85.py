from mini_origin import numeric_threshold_frontier_v70 as core
from mini_origin import response_lattice_closure_v85 as lattice


def records(label_shift=0, rename=False):
    rows = []
    for index in range(96):
        categorical = ("alpha", "beta", "missing")[index % 3]
        if rename:
            categorical = {"alpha": "x", "beta": "y", "missing": "z"}[categorical]
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
    first, _ = core.compile_task("v85-token-a", records(rename=False))
    second, _ = core.compile_task("v85-token-b", records(rename=True))
    a = lattice.outcome_atoms(first, first.full_mask)
    b = lattice.outcome_atoms(second, second.full_mask)
    mask_a = (1 << first.query_count) - 1
    mask_b = (1 << second.query_count) - 1
    assert lattice.canonical_candidate(first, first.full_mask, mask_a, 0, a) == lattice.canonical_candidate(second, second.full_mask, mask_b, 0, b)
