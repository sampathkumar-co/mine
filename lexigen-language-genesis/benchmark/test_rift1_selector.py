from __future__ import annotations

from rift1 import build_cases
from rift1_selector import MECHANISMS, build_library, run, select_artifact


def test_selector_uses_demonstrations_without_mechanism_labels() -> None:
    library = build_library()
    for hidden in MECHANISMS:
        demonstrations = build_cases(hidden, [5, 6], replicas=1)
        selected, _, matches, scores = select_artifact(library, demonstrations)
        assert selected == hidden
        assert matches[selected] is True
        assert selected in scores


def test_closure_ambiguity_is_resolved_by_mdl() -> None:
    library = build_library()
    demonstrations = build_cases("closure", [5, 6], replicas=1)
    selected, _, matches, scores = select_artifact(library, demonstrations)
    assert sum(int(value) for value in matches.values()) == 3
    assert selected == "closure"
    assert scores["closure"] < scores["two_cycle_canonical"] < scores["trajectory_union"]


def test_unlabeled_selector_transfers_exactly(tmp_path) -> None:
    report = run(tmp_path)
    assert len(report["episodes"]) == 3
    assert all(episode["transfer_accuracy"] == 1.0 for episode in report["episodes"])
    assert "uniqueness selection failed" in report["negative_result_preserved"]
