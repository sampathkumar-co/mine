from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "lexigen-world-covering-v2"
sys.path.insert(0, str(V2))

import normalize_evidence  # noqa: E402
import solver_engine  # noqa: E402
from common import (  # noqa: E402
    Target,
    build_incidence,
    select_targets,
    verify_design,
)

V1_SEED = (
    "32c897005c91865319f1b7da264b6162fc1ff4de|"
    "b2c626b07f216aac830d344eff5ad523|LEXIGEN_WORLD_COVERING_V1"
)
REFERENCE_DATE = datetime(2026, 4, 24, tzinfo=timezone.utc)


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_frozen_hashes() -> None:
    lock = json.loads((V2 / "LOCK.json").read_text(encoding="utf-8"))
    assert lock["reservation_rule"] == "exact_frozen_v1_selector_then_exclude_names"
    for relative_name, expected in lock["git_blob_sha1"].items():
        path = V2 / relative_name
        actual = git_blob_sha1(path.read_bytes())
        assert actual == expected, (relative_name, actual, expected)


def parse_date(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime(1996, 1, 1, tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def reference_v1_selection(coverdata: dict[str, object]) -> list[str]:
    eligible: list[tuple[float, str, str, int, int]] = []
    for name, raw in coverdata.items():
        if not isinstance(raw, dict) or not name.startswith("C("):
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
            and upper <= 100
            and upper - 1 >= lower
        ):
            continue
        candidate_blocks = math.comb(v, k)
        t_subsets = math.comb(v, t)
        incidence_edges = candidate_blocks * math.comb(k, t)
        if candidate_blocks > 50_000 or t_subsets > 5_000 or incidence_edges > 3_000_000:
            continue
        improvements = raw.get("imps") or []
        last_update = ""
        if isinstance(improvements, list) and improvements:
            row = improvements[0]
            if isinstance(row, list) and len(row) >= 4:
                last_update = str(row[3])
        age_years = max(0.0, (REFERENCE_DATE - parse_date(last_update)).days / 365.25)
        score = (
            (float(gap) ** 1.35)
            * (1.0 + min(age_years, 25.0) / 18.0)
            * ((100.0 / float(upper)) ** 0.15)
            / (float(incidence_edges) ** 0.35)
        )
        tie = hashlib.sha256(f"{V1_SEED}|{name}".encode()).hexdigest()
        eligible.append((score, tie, name, k, t))

    eligible.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected: list[str] = []
    pair_counts: dict[tuple[int, int], int] = {}
    for _, _, name, k, t in eligible:
        pair = (k, t)
        if pair_counts.get(pair, 0) >= 2:
            continue
        selected.append(name)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        if len(selected) == 3:
            return selected
    raise AssertionError("reference selector did not find three targets")


def synthetic_coverdata() -> dict[str, object]:
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
    ]
    return {
        name: {"size": upper, "low_bd": lower, "imps": [["", "", "", date]]}
        for name, upper, lower, date in rows
    }


def verify_exact_reservation() -> dict[str, object]:
    coverdata = synthetic_coverdata()
    expected = reference_v1_selection(coverdata)
    reserved, selected = select_targets(coverdata)
    actual = [target.name for target in reserved]
    v2_names = [target.name for target in selected]
    assert actual == expected, (actual, expected)
    assert set(actual).isdisjoint(v2_names)
    assert len(v2_names) == 3
    reserved_again, selected_again = select_targets(coverdata)
    assert actual == [target.name for target in reserved_again]
    assert v2_names == [target.name for target in selected_again]
    return {"reference_v1": expected, "v2_selected": v2_names}


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


def verify_solver_and_attribution() -> dict[str, object]:
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
    assert not verify_design(fano_target, fano_blocks[:-1])[0]
    incidence = build_incidence(fano_target)
    assert len(incidence.blocks) == 35 and len(incidence.tsets) == 21

    solver_engine.LOCAL_SECONDS = 4.0
    solver_engine.LOCAL_RESTARTS = 2
    solver_engine.CP_SAT_SECONDS = 20.0
    solver_engine.CP_SAT_WORKERS = 2
    with tempfile.TemporaryDirectory() as temporary:
        success = solver_engine.solve_target(fano_target, Path(temporary))
    assert success["record_candidate"] and success["valid"], success
    assert int(success["result_blocks"]) <= 7, success

    impossible_target = target("SYNTHETIC_IMPOSSIBLE_C(7,3,2)", 7, 3, 2, upper=7, lower=7)
    with tempfile.TemporaryDirectory() as temporary:
        failure = solver_engine.solve_target(impossible_target, Path(temporary))
    assert not failure["record_candidate"] and not failure["valid"], failure

    mislabeled = {
        "method": "generic_greedy",
        "goal_blocks": 7,
        "greedy_best_blocks": 8,
        "local_runs": [{"valid": True}],
    }
    changed = normalize_evidence.normalize_result(mislabeled)
    assert changed and mislabeled["method"] == "stochastic_fixed_budget"

    untouched = {
        "method": "generic_greedy",
        "goal_blocks": 7,
        "greedy_best_blocks": 7,
        "local_runs": [],
    }
    assert not normalize_evidence.normalize_result(untouched)
    assert untouched["method"] == "generic_greedy"

    return {
        "solver_success_method_before_normalization": success["method"],
        "solver_success_blocks": success["result_blocks"],
        "impossible_case_status": failure["cp_sat_status"],
        "normalizer_test": "passed",
    }


def main() -> None:
    verify_frozen_hashes()
    selector_report = verify_exact_reservation()
    solver_report = verify_solver_and_attribution()
    report = {
        "protocol": "LEXIGEN World Covering Record v2 corrected synthetic validation",
        "validated_v2_head": "d652f4be60fc498d07f478a663fd9ca9c4a70428",
        "snapshot_accessed": False,
        "real_target_identity_accessed": False,
        "frozen_hashes": "passed",
        "exact_v1_reservation": "passed",
        "no_v1_v2_overlap": "passed",
        **selector_report,
        **solver_report,
    }
    output = ROOT / "lexigen-world-covering-v2-validation-r2" / "VALIDATION_REPORT.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
