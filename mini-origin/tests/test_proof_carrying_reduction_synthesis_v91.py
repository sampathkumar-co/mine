from __future__ import annotations

import json

from mini_origin import proof_carrying_reduction_synthesis_v91 as v91


def _safe_spec() -> v91.RelationSpec:
    return v91.RelationSpec(
        "true",
        "forall_right_exists_left",
        "le",
        "recursive_left_le_right",
    )


def test_preregistration_and_grammar_are_frozen() -> None:
    payload = json.loads(v91.PREREGISTRATION.read_text(encoding="utf-8"))
    assert payload["status"] == "preregistered_before_v91_implementation_or_evaluation"
    assert payload["frozen_relation_grammar"]["candidate_count"] == 72
    grammar = v91.relation_grammar()
    assert len(grammar) == 72
    assert len(set(grammar)) == 72


def test_recursive_simulation_can_certify_stronger_successor() -> None:
    problem = v91.Problem(
        name="handcrafted",
        initial_state=3,
        terminal_state=0,
        actions_by_state={
            1: (v91.Action(0, 3, 0),),
            2: (v91.Action(1, 5, 0),),
            3: (
                v91.Action(2, 1, 1),
                v91.Action(3, 1, 2),
            ),
        },
    )
    spec = _safe_spec()
    relation = v91.RelationEvaluator(problem, spec)
    assert relation.relates(1, 2) is True
    assert v91.exact_values(problem)[1] <= v91.exact_values(problem)[2]
    raw = v91.solve_with_spec(problem, None)
    reduced = v91.solve_with_spec(problem, spec)
    assert raw.objective == reduced.objective == 4
    assert reduced.action_expansions < raw.action_expansions
    assert reduced.actions_pruned >= 1


def test_equal_cost_mutual_relation_uses_action_id_tie_break() -> None:
    problem = v91.Problem(
        name="tie",
        initial_state=1,
        terminal_state=0,
        actions_by_state={
            1: (
                v91.Action(9, 2, 0),
                v91.Action(4, 2, 0),
            )
        },
    )
    relation = v91.RelationEvaluator(problem, _safe_spec())
    kept, pruned = v91.representative_actions(relation, problem.actions(1))
    assert pruned == 1
    assert [action.action_id for action in kept] == [4]


def test_known_unsafe_formula_has_semantic_counterexample() -> None:
    # Existential matching with ignored immediate costs can claim that an
    # expensive left state is no worse merely because one transition shape
    # can be paired. The audit must be capable of rejecting such relations.
    problem = v91.Problem(
        name="unsafe-counterexample",
        initial_state=2,
        terminal_state=0,
        actions_by_state={
            1: (v91.Action(0, 100, 0),),
            2: (v91.Action(1, 1, 0),),
        },
    )
    unsafe = v91.RelationSpec(
        "true",
        "exists_right_exists_left",
        "ignore",
        "recursive_left_le_right",
    )
    relation = v91.RelationEvaluator(problem, unsafe)
    assert relation.relates(1, 2) is True
    values = v91.exact_values(problem)
    assert values[1] > values[2]
    assert v91.semantic_violations(problem, unsafe) >= 1


def test_small_selection_freezes_before_withheld_construction(monkeypatch) -> None:
    # Shrink only the test-time corpus; selection mechanics and ordering stay
    # unchanged. This catches accidental dependence on withheld construction.
    monkeypatch.setattr(v91, "AUDIT_INSTANCES", 5)
    monkeypatch.setattr(v91, "TRAIN_INSTANCES_PER_FAMILY", 4)
    selected, synthesis = v91.select_rule()
    assert synthesis["grammar_candidates"] == 72
    assert synthesis["eligible_candidates"] >= 1
    assert synthesis["selected_spec"] == selected.payload()
    assert len(synthesis["freeze_digest"]) == 64

    monkeypatch.setattr(v91, "WITHHELD_INSTANCES", 4)
    withheld = v91.withheld_evaluation(selected, str(synthesis["freeze_digest"]))
    assert withheld["objective_mismatches"] == 0
    assert withheld["local_certificate_failures"] == 0
