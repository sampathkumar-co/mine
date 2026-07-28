from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "v3-source" / "lexigen-world-covering-v3"
sys.path.insert(0, str(V3))

import solver_engine  # noqa: E402
from common import Target, build_incidence, select_target_lineage, verify_design  # noqa: E402

V1_SEED = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V1"
)
V2_SEED = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V2"
)
REFERENCE_DATE = datetime(2026, 4, 24, tzinfo=timezone.utc)


def git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_hashes() -> None:
    lock = json.loads((V3 / "LOCK.json").read_text(encoding="utf-8"))
    assert lock["lineage_rule"] == "reproduce_exact_v1_then_corrected_v2_then_exclude_all_six"
    for relative, expected in lock["git_blob_sha1"].items():
        path = V3 / relative
        assert git_blob(path.read_bytes()) == expected, relative


def parse_date(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def base_rows() -> dict[str, object]:
    rows = [
        ("C(10,4,3)", 30, 25, "2012-01-01T00:00:00+00:00"),
        ("C(11,4,3)", 31, 27, "2014-01-01T00:00:00+00:00"),
        ("C(12,5,3)", 34, 29, "2010-01-01T00:00:00+00:00"),
        ("C(13,5,3)", 37, 32, "2016-01-01T00:00:00+00:00"),
        ("C(14,6,3)", 43, 37, "2008-01-01T00:00:00+00:00"),
        ("C(15,6,3)", 47, 42, "2018-01-01T00:00:00+00:00"),
        ("C(16,7,3)", 55, 49, "2007-01-01T00:00:00+00:00"),
        ("C(17,7,3)", 60, 54, "2019-01-01T00:00:00+00:00"),
        ("C(17,8,4)", 70, 64, "2006-01-01T00:00:00+00:00"),
        ("C(18,8,4)", 76, 69, "2005-01-01T00:00:00+00:00"),
        ("C(19,9,4)", 90, 82, "2004-01-01T00:00:00+00:00"),
        ("C(20,9,4)", 98, 90, "2020-01-01T00:00:00+00:00"),
    ]
    return {
        name: {"size": upper, "low_bd": lower, "imps": [["", "", "", date]]}
        for name, upper, lower, date in rows
    }


def reference_select(
    coverdata: dict[str, object],
    phase: str,
    excluded: set[str],
) -> list[str]:
    items = []
    for name, raw in coverdata.items():
        if name in excluded or not isinstance(raw, dict):
            continue
        v, k, t = map(int, name[2:-1].split(","))
        upper = int(raw["size"])
        lower = int(raw["low_bd"])
        gap = upper - lower
        if not (
            10 <= v <= 22
            and 4 <= k <= min(10, v - 2)
            and 3 <= t <= min(5, k - 1)
            and gap >= 2
            and upper - 1 >= lower
        ):
            continue
        candidates = math.comb(v, k)
        tsets = math.comb(v, t)
        edges = candidates * math.comb(k, t)
        date = str(raw["imps"][0][3])
        age = max(0.0, (REFERENCE_DATE - parse_date(date)).days / 365.25)
        if phase == "v1":
            if upper > 100 or candidates > 50_000 or tsets > 5_000 or edges > 3_000_000:
                continue
            score = gap**1.35 * (1 + min(age, 25) / 18) * (100 / upper) ** 0.15 / edges**0.35
            seed = V1_SEED
        elif phase == "v2":
            if upper > 100 or candidates > 60_000 or tsets > 5_000 or edges > 3_500_000:
                continue
            score = gap**1.45 * (1 + min(age, 25) / 16) * (100 / upper) ** 0.12 / edges**0.32
            seed = V2_SEED
        else:
            raise AssertionError(phase)
        tie = hashlib.sha256(f"{seed}|{name}".encode()).hexdigest()
        items.append((score, tie, name, k, t))
    items.sort(key=lambda row: (-row[0], row[1], row[2]))
    chosen = []
    counts: dict[tuple[int, int], int] = {}
    for _, _, name, k, t in items:
        pair = (k, t)
        if counts.get(pair, 0) >= 2:
            continue
        chosen.append(name)
        counts[pair] = counts.get(pair, 0) + 1
        if len(chosen) == 3:
            return chosen
    raise AssertionError(f"insufficient {phase} targets")


def verify_lineage() -> dict[str, object]:
    data = base_rows()
    expected_v1 = reference_select(data, "v1", set())
    expected_v2 = reference_select(data, "v2", set(expected_v1))
    v1, v2, v3 = select_target_lineage(data)
    actual_v1 = [x.name for x in v1]
    actual_v2 = [x.name for x in v2]
    actual_v3 = [x.name for x in v3]
    assert actual_v1 == expected_v1
    assert actual_v2 == expected_v2
    assert len(actual_v3) == 3
    assert set(actual_v1 + actual_v2).isdisjoint(actual_v3)
    again = select_target_lineage(data)
    assert actual_v3 == [x.name for x in again[2]]
    return {"v1": actual_v1, "v2": actual_v2, "v3": actual_v3}


def target(name: str, upper: int) -> Target:
    v, k, t = 7, 3, 2
    return Target(
        name=name,
        v=v,
        k=k,
        t=t,
        upper=upper,
        lower=7,
        last_update="synthetic",
        gap=upper - 7,
        candidate_blocks=math.comb(v, k),
        t_subsets=math.comb(v, t),
        incidence_edges=math.comb(v, k) * math.comb(k, t),
        opportunity_score=0.0,
        tie_break="synthetic",
    )


def verify_solver() -> dict[str, object]:
    feasible = target("SYNTHETIC_FANO", 8)
    incidence = build_incidence(feasible)
    fano = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6), (1, 3, 5),
        (1, 4, 6), (2, 3, 6), (2, 4, 5),
    ]
    assert verify_design(feasible, fano)[0]
    assert not verify_design(feasible, fano[:-1])[0]
    hint = [incidence.blocks.index(block) for block in fano]
    pool = solver_engine.candidate_pool(incidence, hint)
    restricted, restricted_status, _ = solver_engine.cp_search(
        feasible, incidence, pool, hint, 20.0, 123
    )
    assert restricted_status in {"OPTIMAL", "FEASIBLE"}, restricted_status
    assert verify_design(feasible, [incidence.blocks[i] for i in restricted])[0]

    impossible = target("SYNTHETIC_IMPOSSIBLE", 7)
    impossible_incidence = build_incidence(impossible)
    none, impossible_status, _ = solver_engine.cp_search(
        impossible, impossible_incidence, None, [], 20.0, 456
    )
    assert not none and impossible_status == "INFEASIBLE", impossible_status

    solver_engine.REPAIR_SECONDS = 4.0
    solver_engine.REPAIR_RESTARTS = 2
    solver_engine.RESTRICTED_CP_SECONDS = 20.0
    solver_engine.FULL_CP_SECONDS = 20.0
    solver_engine.CP_WORKERS = 2
    with tempfile.TemporaryDirectory() as temporary:
        result = solver_engine.solve_target(feasible, Path(temporary))
    assert result["record_candidate"] and result["valid"], result
    return {
        "restricted_status": restricted_status,
        "impossible_status": impossible_status,
        "solve_method": result["method"],
        "solve_blocks": result["result_blocks"],
    }


def main() -> None:
    verify_hashes()
    lineage = verify_lineage()
    solver = verify_solver()
    report = {
        "protocol": "LEXIGEN World Covering Record v3 snapshot-free validation",
        "validated_head": "7ca690ba471da0aadea2b34b8f7563da7fb59024",
        "snapshot_accessed": False,
        "real_target_identity_accessed": False,
        "locked_hashes": "passed",
        "exact_lineage": "passed",
        "no_overlap": "passed",
        "solver_validation": "passed",
        "lineage": lineage,
        **solver,
    }
    output = ROOT / "lexigen-world-covering-v3-validation" / "VALIDATION_REPORT.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
