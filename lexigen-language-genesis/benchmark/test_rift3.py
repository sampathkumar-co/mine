from __future__ import annotations

from rift3 import (
    adapt_family,
    build_cases,
    fits_family,
    raw_fits,
    raw_program_search,
    run,
)


def test_family_recombines_and_extends_semantics() -> None:
    predecessor = build_cases("cycle_predecessor", [5], replicas=1)
    entry = build_cases("cycle_entry", [5], replicas=1)

    pred_instance, pred_tested, pred_extended = adapt_family(predecessor)
    entry_instance, entry_tested, entry_extended = adapt_family(entry)

    assert pred_instance["stop"] == "repeat"
    assert pred_instance["finalize"] == "current"
    assert pred_extended is False
    assert entry_instance["stop"] == "repeat"
    assert entry_instance["finalize"] == "next"
    assert entry_extended is True
    assert pred_tested == entry_tested == 8


def test_family_adaptation_beats_raw_program_search() -> None:
    for query in ("cycle_predecessor", "cycle_entry"):
        demonstrations = build_cases(query, [5], replicas=1)
        transfer = build_cases(query, range(8, 11), replicas=1)
        instance, family_tested, _ = adapt_family(demonstrations)
        artifact, raw_tested = raw_program_search(demonstrations)
        assert fits_family(instance, transfer)
        assert raw_fits(artifact, transfer)
        assert raw_tested > family_tested


def test_rift3_report(tmp_path) -> None:
    report = run(tmp_path)
    assert set(report["episodes"]) == {"cycle_predecessor", "cycle_entry"}
    assert all(result["transfer_accuracy"] == 1.0 for result in report["episodes"].values())
    assert report["episodes"]["cycle_entry"]["vocabulary_extended"] is True
