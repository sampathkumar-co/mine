from __future__ import annotations

import argparse
import json
from pathlib import Path

from .claim_contract_v18 import Bundle, Contract, Manifest, Run, verify


def parse_contract(value: dict[str, object]) -> Contract:
    return Contract(
        claim_id=str(value["claim_id"]),
        candidate_hash=str(value["candidate_hash"]),
        required_runs=int(value["required_runs"]),
        min_successes=int(value["min_successes"]),
        score_threshold=float(value["score_threshold"]),
        median_threshold=float(value["median_threshold"]),
        min_control_gap=float(value["min_control_gap"]),
        median_control_gap=float(value["median_control_gap"]),
        min_ablation_gap=float(value["min_ablation_gap"]),
        oracle_ceiling=float(value["oracle_ceiling"]),
        operation_budget=float(value["operation_budget"]),
        required_checks=tuple(str(item) for item in value["required_checks"]),
    )


def parse_manifest(value: dict[str, object]) -> Manifest:
    return Manifest(
        contract_digest=str(value["contract_digest"]),
        candidate_hash=str(value["candidate_hash"]),
        seeds=tuple(int(item) for item in value["seeds"]),
        commitment=str(value["commitment"]),
        issued_after_freeze=bool(value["issued_after_freeze"]),
    )


def parse_run(value: dict[str, object]) -> Run:
    checks = tuple(
        (str(item[0]), bool(item[1]))
        for item in value["checks"]
    )
    return Run(
        seed=int(value["seed"]),
        score=float(value["score"]),
        control=float(value["control"]),
        ablation=float(value["ablation"]),
        candidate_budget=float(value["candidate_budget"]),
        control_budget=float(value["control_budget"]),
        threshold_used=float(value["threshold_used"]),
        contract_digest=str(value["contract_digest"]),
        candidate_hash=str(value["candidate_hash"]),
        manifest_digest=str(value["manifest_digest"]),
        holdout_candidates=int(value["holdout_candidates"]),
        selected_after_holdout=bool(value["selected_after_holdout"]),
        holdout_policy_violations=int(value["holdout_policy_violations"]),
        checks=checks,
    )


def parse_bundle(value: dict[str, object]) -> Bundle:
    return Bundle(
        claim_id=str(value["claim_id"]),
        claimed_breakthrough=bool(value["claimed_breakthrough"]),
        runs=tuple(parse_run(item) for item in value["runs"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = []
    for case in payload["cases"]:
        verdict = verify(
            parse_contract(case["contract"]),
            parse_manifest(case["manifest"]),
            parse_bundle(case["bundle"]),
        )
        results.append(
            {
                "name": case["name"],
                "expected_accept": bool(case["expected_accept"]),
                "observed_accept": bool(verdict["accepted"]),
                "correct": bool(verdict["accepted"]) == bool(case["expected_accept"]),
                "problems": verdict["problems"],
            }
        )
    report = {
        "case_count": len(results),
        "correct_count": sum(item["correct"] for item in results),
        "accuracy": sum(item["correct"] for item in results) / max(1, len(results)),
        "all_correct": all(item["correct"] for item in results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
