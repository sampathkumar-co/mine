from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path

from .rotating_sketch_v12 import (
    SketchEvaluation,
    SketchProgram,
    SketchScenario,
    evaluate_program,
)


@dataclass(frozen=True)
class AuditedMethod:
    name: str
    program: SketchProgram


def methods() -> list[AuditedMethod]:
    return [
        AuditedMethod(
            "single_random_shard",
            SketchProgram(0.001, "iid", 1, 1, 1, 2.5e-5, 310_001),
        ),
        AuditedMethod(
            "iid_05pct",
            SketchProgram(0.05, "iid", 1, 1, 1, 2.5e-5, 310_005),
        ),
        AuditedMethod(
            "iid_10pct",
            SketchProgram(0.10, "iid", 1, 1, 1, 2.5e-5, 310_010),
        ),
        AuditedMethod(
            "iid_15pct",
            SketchProgram(0.15, "iid", 1, 1, 1, 2.5e-5, 310_015),
        ),
        AuditedMethod(
            "iid_20pct",
            SketchProgram(0.20, "iid", 1, 1, 1, 2.5e-5, 310_020),
        ),
        AuditedMethod(
            "antithetic_10pct",
            SketchProgram(0.10, "antithetic", 5, 3, 7, 5.0e-5, 320_010),
        ),
        AuditedMethod(
            "antithetic_20pct",
            SketchProgram(0.20, "antithetic", 5, 3, 7, 5.0e-5, 320_020),
        ),
        AuditedMethod(
            "balanced_40pct",
            SketchProgram(0.40, "balanced", 1, 3, 5, 1.0e-5, 7_777),
        ),
        AuditedMethod(
            "static_40pct",
            SketchProgram(0.40, "cyclic", 1, 0, 0, 1.0e-5, 1),
        ),
        AuditedMethod(
            "dense_100pct",
            SketchProgram(1.00, "balanced", 1, 1, 1, 1.0e-5, 1),
        ),
        AuditedMethod(
            "v12_seed121_iid",
            SketchProgram(0.215, "iid", 65, 85, 73, 2.5e-5, 1_792_977),
        ),
        AuditedMethod(
            "v12_seed123_antithetic",
            SketchProgram(0.231, "antithetic", 83, 81, 83, 5.6e-5, 1_927_613),
        ),
    ]


def scenarios(seed: int) -> list[SketchScenario]:
    return [
        SketchScenario(seed + 501, 9, 12, 3.2, 68, 42.0, 0.055, 0.59),
        SketchScenario(seed + 502, 11, 14, 3.3, 76, 55.0, 0.065, 0.61),
        SketchScenario(seed + 503, 14, 17, 3.4, 86, 70.0, 0.075, 0.63),
        SketchScenario(seed + 504, 16, 20, 3.5, 96, 85.0, 0.085, 0.65),
    ]


def _serialize(value: SketchEvaluation) -> dict[str, float | int]:
    return {
        "score": value.score,
        "pre_damage": value.pre_damage,
        "post_damage": value.post_damage,
        "retention": value.retention,
        "write_fraction": value.write_fraction,
        "coverage_min": value.coverage_min,
        "coverage_spread": value.coverage_spread,
    }


def pareto_frontier(rows: list[dict[str, object]]) -> list[str]:
    frontier: list[str] = []
    for candidate in rows:
        candidate_write = float(candidate["max_write_fraction"])
        candidate_post = float(candidate["strict_post_damage"])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            other_write = float(other["max_write_fraction"])
            other_post = float(other["strict_post_damage"])
            if (
                other_write <= candidate_write + 1e-12
                and other_post >= candidate_post - 1e-12
                and (
                    other_write < candidate_write - 1e-12
                    or other_post > candidate_post + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["name"]))
    return frontier


def run_audit(seed: int = 131) -> dict[str, object]:
    hidden = scenarios(seed * 10_000)
    rows: list[dict[str, object]] = []
    for method in methods():
        evaluations = {
            scenario.label(): evaluate_program(method.program, scenario)
            for scenario in hidden
        }
        row = {
            "name": method.name,
            "program": method.program.text(),
            "strict_post_damage": min(value.post_damage for value in evaluations.values()),
            "median_post_damage": sorted(value.post_damage for value in evaluations.values())[len(evaluations) // 2],
            "min_retention": min(value.retention for value in evaluations.values()),
            "max_write_fraction": max(value.write_fraction for value in evaluations.values()),
            "evaluations": {key: _serialize(value) for key, value in evaluations.items()},
        }
        rows.append(row)

    frontier = pareto_frontier(rows)
    candidates = [
        row for row in rows
        if row["name"] in {"v12_seed121_iid", "v12_seed123_antithetic"}
    ]
    standard = [
        row for row in rows
        if not str(row["name"]).startswith("v12_")
    ]
    candidate_survives = False
    reasons: list[str] = []
    for candidate in candidates:
        write = float(candidate["max_write_fraction"])
        post = float(candidate["strict_post_damage"])
        retention = float(candidate["min_retention"])
        cheaper_match = [
            row for row in standard
            if float(row["max_write_fraction"]) <= write + 1e-12
            and float(row["strict_post_damage"]) >= post - 0.02
        ]
        survives = (
            post >= 0.90
            and retention >= 0.95
            and not cheaper_match
            and str(candidate["name"]) in frontier
        )
        candidate_survives = candidate_survives or survives
        if cheaper_match:
            reasons.append(
                f"{candidate['name']} matched within 0.02 by "
                + ",".join(str(row["name"]) for row in cheaper_match)
            )
        if post < 0.90:
            reasons.append(f"{candidate['name']} strict recovery below 0.90")
        if retention < 0.95:
            reasons.append(f"{candidate['name']} retention below 0.95")
        if str(candidate["name"]) not in frontier:
            reasons.append(f"{candidate['name']} is Pareto dominated")

    return {
        "verdict": "v12_candidate_survives_pareto_audit" if candidate_survives else "v12_candidate_rejected",
        "claim_scope": (
            "fresh-seed Pareto audit against single-shard and fixed-density random write baselines; "
            "this audit tests whether v0.12 offers a mechanism beyond ordinary random sharding"
        ),
        "seed": seed,
        "pareto_frontier": frontier,
        "reasons": reasons,
        "methods": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_audit(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    compact = {
        "verdict": payload["verdict"],
        "pareto_frontier": payload["pareto_frontier"],
        "reasons": payload["reasons"],
        "methods": [
            {
                "name": row["name"],
                "strict_post_damage": row["strict_post_damage"],
                "min_retention": row["min_retention"],
                "max_write_fraction": row["max_write_fraction"],
            }
            for row in payload["methods"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
