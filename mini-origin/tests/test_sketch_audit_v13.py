from mini_origin.sketch_audit_v13 import methods, pareto_frontier


def test_audit_contains_single_shard_and_v12_candidates() -> None:
    names = {method.name for method in methods()}
    assert "single_random_shard" in names
    assert "v12_seed121_iid" in names
    assert "v12_seed123_antithetic" in names
    assert "dense_100pct" in names


def test_pareto_frontier_rejects_dominated_method() -> None:
    rows = [
        {"name": "cheap_good", "max_write_fraction": 0.10, "strict_post_damage": 0.90},
        {"name": "expensive_same", "max_write_fraction": 0.20, "strict_post_damage": 0.90},
        {"name": "expensive_better", "max_write_fraction": 0.20, "strict_post_damage": 0.95},
    ]
    frontier = pareto_frontier(rows)
    assert "expensive_same" not in frontier
    assert "cheap_good" in frontier
    assert "expensive_better" in frontier
