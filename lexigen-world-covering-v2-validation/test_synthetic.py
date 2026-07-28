from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "lexigen-world-covering-v2"
sys.path.insert(0, str(V2))

import solver_engine  # noqa: E402
from common import Target, build_incidence, select_targets, verify_design  # noqa: E402


def target(name: str, v: int, k: int, t: int, upper: int, lower: int) -> Target:
    return Target(
        name=name,
        v=v,
        k=k,
        t=t,
        upper=upper,
        lower=lower,
        last_update="synthetic",
        gap=upper - lower,
        candidate_blocks=math.comb(v, k),
        t_subsets=math.comb(v, t),
        incidence_edges=math.comb(v, k) * math.comb(k, t),
        opportunity_score=0.0,
        tie_break="synthetic",
    )


def verify_frozen_hashes() -> None:
    lock = json.loads((V2 / "LOCK.json").read_text(encoding="utf-8"))
    for relative_name, expected in lock["git_blob_sha1"].items():
        path = V2 / relative_name
        data = path.read_bytes()
        actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert actual == expected, (relative_name, actual, expected)


def verify_selector_determinism() -> None:
    names = [
        "C(10,4,3)",
        "C(11,4,3)",
        "C(12,5,3)",
        "C(13,5,3)",
        "C(14,6,3)",
        "C(15,6,3)",
    ]
    coverdata = {
        name: {
            "size": 30 + index,
            "low_bd": 25 + index,
            "imps": [["", "", "", "2020-01-01T00:00:00+00:00"]],
        }
        for index, name in enumerate(names)
    }
    first_reserved, first_v2 = select_targets(coverdata)
    second_reserved, second_v2 = select_targets(coverdata)
    assert [row.name for row in first_reserved] == [row.name for row in second_reserved]
    assert [row.name for row in first_v2] == [row.name for row in second_v2]
    assert len(first_reserved) == 3 and len(first_v2) == 3
    assert set(row.name for row in first_reserved).isdisjoint(row.name for row in first_v2)


def verify_known_design_and_solver() -> dict[str, object]:
    fano_target = target("SYNTHETIC_C(7,3,2)", 7, 3, 2, upper=8, lower=7)
    fano_blocks = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]
    valid, reason = verify_design(fano_target, fano_blocks)
    assert valid, reason
    invalid, _ = verify_design(fano_target, fano_blocks[:-1])
    assert not invalid

    incidence = build_incidence(fano_target)
    assert len(incidence.blocks) == 35
    assert len(incidence.tsets) == 21

    solver_engine.LOCAL_SECONDS = 4.0
    solver_engine.LOCAL_RESTARTS = 2
    solver_engine.CP_SAT_SECONDS = 20.0
    solver_engine.CP_SAT_WORKERS = 2
    with tempfile.TemporaryDirectory() as temporary:
        success = solver_engine.solve_target(fano_target, Path(temporary))
    assert success["record_candidate"], success
    assert success["valid"], success
    assert int(success["result_blocks"]) <= 7, success

    impossible_target = target("SYNTHETIC_IMPOSSIBLE_C(7,3,2)", 7, 3, 2, upper=7, lower=7)
    with tempfile.TemporaryDirectory() as temporary:
        failure = solver_engine.solve_target(impossible_target, Path(temporary))
    assert not failure["record_candidate"], failure
    assert not failure["valid"], failure

    return {
        "known_design_verification": "passed",
        "solver_success_method": success["method"],
        "solver_success_blocks": success["result_blocks"],
        "impossible_case_status": failure["cp_sat_status"],
        "impossible_case_record_candidate": failure["record_candidate"],
    }


def main() -> None:
    verify_frozen_hashes()
    verify_selector_determinism()
    solver_report = verify_known_design_and_solver()
    report = {
        "protocol": "LEXIGEN World Covering Record v2 synthetic validation",
        "snapshot_accessed": False,
        "real_target_identity_accessed": False,
        "frozen_hashes": "passed",
        "selector_determinism": "passed",
        **solver_report,
    }
    output = ROOT / "lexigen-world-covering-v2-validation" / "VALIDATION_REPORT.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
