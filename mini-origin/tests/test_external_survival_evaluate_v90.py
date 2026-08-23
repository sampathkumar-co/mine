from __future__ import annotations

import gzip

from mini_origin import external_survival_evaluate_v90 as v90


def test_set_cover_identical_successor_quotient() -> None:
    payload = b"""3 4
5 2 1 2
7 2 1 2
5 2 1 2
2 1 3
"""
    instance = v90.parse_rail_set_cover("synthetic", payload)
    kept, stats = v90.quotient_set_cover(instance)
    assert len(kept) == 2
    assert stats == {
        "raw_columns": 4,
        "distinct_coverage_signatures": 2,
        "duplicate_coverage_classes": 1,
        "columns_removed_by_frozen_quotient": 2,
        "equal_cost_tie_removals": 1,
        "strict_cost_dominance_removals": 1,
        "local_certificate_failures": 0,
    }
    assert [column.index for column in kept] == [1, 4]


def test_dimacs_parallel_arc_quotient_preserves_distances() -> None:
    text = """c synthetic
p sp 4 6
a 1 2 3
a 1 2 7
a 1 3 10
a 2 3 2
a 2 4 10
a 3 4 1
""".encode("ascii")
    raw = v90.parse_dimacs_graph("synthetic", gzip.compress(text))
    quotient, stats = v90.quotient_shortest_path(raw)
    assert stats["raw_arcs"] == 6
    assert stats["quotient_arcs"] == 5
    assert stats["arcs_removed_by_frozen_quotient"] == 1
    assert stats["strict_cost_dominance_removals"] == 1
    assert stats["local_certificate_failures"] == 0
    certificate = v90.shortest_path_distance_certificate(raw, quotient)
    assert certificate["passed"] is True


def test_equal_parallel_arc_uses_lower_source_order() -> None:
    text = """p sp 2 2
a 1 2 4
a 1 2 4
""".encode("ascii")
    raw = v90.parse_dimacs_graph("synthetic", gzip.compress(text))
    quotient, stats = v90.quotient_shortest_path(raw)
    assert len(quotient.arcs) == 1
    assert quotient.arcs[0].index == 0
    assert stats["equal_cost_tie_removals"] == 1
    assert stats["local_certificate_failures"] == 0


def test_evaluation_refuses_to_run_before_committed_lock_manifest(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(v90, "LOCK_MANIFEST", tmp_path / "missing.json")
    try:
        v90._load_lock_manifest()
    except RuntimeError as exc:
        assert "evaluation is locked" in str(exc)
    else:
        raise AssertionError("evaluation ran without committed lock manifest")
