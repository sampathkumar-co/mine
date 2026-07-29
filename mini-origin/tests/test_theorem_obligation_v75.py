from __future__ import annotations

from mini_origin import response_cost_pareto_v56 as response
from mini_origin import theorem_obligation_audit_v75 as v75


def test_tie_scope_boundary_is_explicit() -> None:
    witness = v75.tie_scope_witness()
    assert witness["passed"]
    assert witness["fresh_descendant_mask"] != witness["carried_descendant_mask"]
    assert witness["descendant_vectors"][0] == witness["descendant_vectors"][1]


def test_profile_and_partitions_conform() -> None:
    task, profile = response.random_task_and_profile(75_999)
    assert v75.profile_conforms(task, profile)
    assert all(
        v75.partition_conforms(task, allowed, query)
        for allowed in response.descendants(task.full_mask)
        for query in range(task.query_count)
    )
    assert response.hereditary_pareto_theorem(task, profile)["passed"]
