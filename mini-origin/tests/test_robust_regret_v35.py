from mini_origin import robust_regret_v35 as v35
from mini_origin import state_policy_v34 as v34


def test_cached_baselines_equal_direct_controls() -> None:
    tasks = v35.opened_domain_pool()[:3]
    baselines = v35.control_baselines(tasks)
    controls = v34.constant_programs()
    for task in tasks:
        rows = [
            v34.evaluate(task, program)
            for program in controls.values()
        ]
        best_diagnosed = max(
            row.diagnosed_fraction for row in rows
        )
        eligible = [
            row
            for row in rows
            if row.diagnosed_fraction >= best_diagnosed - 1e-12
        ]
        actual = baselines[task.name]
        assert actual["best_diagnosed"] == best_diagnosed
        assert actual["best_worst"] == min(
            row.worst_queries for row in eligible
        )
        assert actual["best_mean"] == min(
            row.mean_queries for row in eligible
        )


def test_cached_regret_matches_direct_metrics() -> None:
    task = v35.opened_domain_pool()[0]
    program = list(v34.grammar())[89]
    baseline = v35.control_baselines([task])[task.name]
    row = v35.task_regret(task, program, baseline)
    candidate = v34.evaluate(task, program)
    assert row["candidate_diagnosed"] == candidate.diagnosed_fraction
    assert row["candidate_worst"] == candidate.worst_queries
    assert row["candidate_mean"] == candidate.mean_queries
