from __future__ import annotations

from rift1 import build_cases
from rift2 import (
    MECHANISMS,
    execute_trajectory_operator,
    induce_instance,
    induce_operator_family,
    run,
)


def test_instances_are_induced_without_mechanism_labels() -> None:
    instances = []
    for mechanism in MECHANISMS:
        demonstrations = build_cases(mechanism, range(4, 7), replicas=1)
        transfer = build_cases(mechanism, range(7, 10), replicas=1)
        instance, _ = induce_instance(demonstrations)
        instances.append(instance)
        for case in transfer:
            assert execute_trajectory_operator(instance, case.step, case.seed) == case.independently_verified_target()
    assert len({(i["stop"], i["finalize"]) for i in instances}) == 3


def test_family_anti_unifies_both_parameters() -> None:
    instances = [
        induce_instance(build_cases(mechanism, range(4, 7), replicas=1))[0]
        for mechanism in MECHANISMS
    ]
    family = induce_operator_family(instances)
    assert set(family["operator_schema"]["varying_parameters"]) == {"stop", "finalize"}
    assert len(family["instances"]) == 3


def test_rift2_report_transfers_exactly(tmp_path) -> None:
    report = run(tmp_path)
    assert all(value["transfer_accuracy"] == 1.0 for value in report["episodes"].values())
    assert report["status"].startswith("induced higher-order executable language")
    assert report["description_length"]["compression_ratio"] > 1.0
