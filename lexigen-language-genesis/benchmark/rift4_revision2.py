from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from portable_family_runtime import run_portable_instance
from rift4 import (
    MaxCycleWorld,
    SURFACES,
    adapt_family,
    execute_family,
    exhaustive_fixed_language_search,
    key,
)


def prefix(surface: str, replica: int) -> str:
    return {
        "graph": f"v{replica}_",
        "rules": f"atom{replica}_",
        "grid": f"sq{replica}_",
    }[surface]


def make_world(depth: int, surface: str, replica: int) -> MaxCycleWorld:
    p = prefix(surface, replica)
    transient = [f"{p}t{i}" for i in range(depth)]

    # Alternate the phase at which repetition is detected. In half the cases the
    # canonical maximum is `next`; in the other half it is `current`. A fixed
    # RETURN_CURRENT or RETURN_NEXT program therefore cannot solve all surfaces.
    parity = (SURFACES.index(surface) + replica + depth) % 2
    if parity == 0:
        cycle = [f"{p}cycle_b", f"{p}cycle_z", f"{p}cycle_a"]
    else:
        cycle = [f"{p}cycle_b", f"{p}cycle_a", f"{p}cycle_z"]

    transitions: list[tuple[str, str]] = []
    transitions.extend((transient[i], transient[i + 1]) for i in range(len(transient) - 1))
    transitions.append((transient[-1], cycle[0]))
    transitions.extend(((cycle[0], cycle[1]), (cycle[1], cycle[2]), (cycle[2], cycle[0])))
    return MaxCycleWorld(
        name=f"cycle-max-r2-{surface}-d{depth}-i{replica}",
        surface=surface,
        transitions=tuple(transitions),
        seed=frozenset({transient[0]}),
    )


def build_cases(depths: Iterable[int], replicas: int = 1) -> list[MaxCycleWorld]:
    return [
        make_world(depth, surface, replica)
        for depth in depths
        for replica in range(replicas)
        for surface in SURFACES
    ]


def run(output_dir: Path) -> dict[str, Any]:
    demonstrations = build_cases([5, 6], replicas=2)
    transfer = build_cases(range(8, 16), replicas=3)

    instance, family_candidates = adapt_family(demonstrations)
    fixed_solved, fixed_candidates = exhaustive_fixed_language_search(demonstrations)

    family_correct = all(
        execute_family(instance, case) == case.independently_verified_target()
        for case in transfer
    )
    portable_correct = all(
        run_portable_instance(instance, case.step, case.seed)
        == case.independently_verified_target()
        for case in transfer
    )

    ablated = dict(instance)
    ablated["finalize"] = "next"
    ablation_accuracy = sum(
        int(execute_family(ablated, case) == case.independently_verified_target())
        for case in transfer
    ) / len(transfer)

    report = {
        "benchmark": "RIFT-4 revision 2",
        "status": "internal L5 mechanism gate candidate; not an external world-breakthrough claim",
        "invented_instance": instance,
        "family_candidates_tested": family_candidates,
        "fixed_language_candidates_exhausted": fixed_candidates,
        "fixed_language_found_solution": fixed_solved,
        "transfer_case_count": len(transfer),
        "family_transfer_accuracy": 1.0 if family_correct else 0.0,
        "portable_interpreter_accuracy": 1.0 if portable_correct else 0.0,
        "ablation_accuracy": ablation_accuracy,
        "gate": {
            "new_executable_semantics": instance["finalize"] == "canonical_max",
            "hidden_surface_transfer": family_correct,
            "portability": portable_correct,
            "ablation_dependency": ablation_accuracy < 1.0,
            "fixed_language_inexpressibility_in_frozen_inventory": not fixed_solved,
        },
        "preserved_negative_result": (
            "RIFT-4 revision 1 was invalid because canonical maximum always coincided with `next`; "
            "the frozen language solved it. Revision 2 alternates cycle phase across cases."
        ),
        "claim_boundary": (
            "The project-internal L5 gate is crossed only within this synthetic protocol. Canonical maximum is a known "
            "operation and the substrate exposes canonical ordering, so this is not yet a world-level AI breakthrough."
        ),
    }
    if not all(report["gate"].values()):
        raise AssertionError(f"RIFT-4 revision 2 gate failed: {report['gate']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rift4-r2-report.json"
    artifact_path = output_dir / "rift4-r2-invented-instance.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_path.write_text(json.dumps(instance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "artifact": instance["name"],
        "family_candidates": family_candidates,
        "fixed_candidates_exhausted": fixed_candidates,
        "fixed_found_solution": fixed_solved,
        "transfer_accuracy": report["family_transfer_accuracy"],
        "portable_accuracy": report["portable_interpreter_accuracy"],
        "ablation_accuracy": ablation_accuracy,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift4"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
