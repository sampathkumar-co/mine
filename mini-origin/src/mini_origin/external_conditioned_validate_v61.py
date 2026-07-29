from __future__ import annotations

import argparse
import json
from pathlib import Path


STAT_FIELDS=("query_expansions","calls","memo_entries","memo_hits","raw_queries_considered","representative_queries_considered","dominated_queries_removed")


def validate(reference_path: Path,rust_path: Path,output: Path) -> dict[str,object]:
    reference=json.loads(reference_path.read_text(encoding="utf-8"))
    rust=json.loads(rust_path.read_text(encoding="utf-8"))
    rust_by_digest={row["digest"]:row for row in rust["rows"]}
    mismatches=[]
    exact_matches=0
    for expected in reference["rows"]:
        digest=expected["state_digest"]
        actual=rust_by_digest.get(digest)
        if actual is None:
            mismatches.append({"digest":digest,"kind":"missing-rust-row"})
            continue
        if bool(actual["solved"])!=bool(expected["pareto_solved"]):
            mismatches.append({"digest":digest,"kind":"solved-status","python":expected["pareto_solved"],"rust":actual["solved"]})
            continue
        if not expected["pareto_solved"]:
            exact_matches+=1
            continue
        if actual["plan"]!=list(expected["pareto_plan"]):
            mismatches.append({"digest":digest,"kind":"plan","python":expected["pareto_plan"],"rust":actual["plan"]})
            continue
        bad={field:{"python":expected["pareto_stats"][field],"rust":actual[field]} for field in STAT_FIELDS if int(expected["pareto_stats"][field])!=int(actual[field])}
        if bad:
            mismatches.append({"digest":digest,"kind":"counters","fields":bad})
            continue
        exact_matches+=1
    expected_digests={row["state_digest"] for row in reference["rows"]}
    unexpected=sorted(set(rust_by_digest)-expected_digests)
    if unexpected:
        mismatches.append({"kind":"unexpected-rust-rows","digests":unexpected[:20]})

    profiled=int(reference["profiled_state_count"])
    pareto=int(reference["pareto_solved_count"])
    both=int(reference["both_solved_count"])
    pareto_only=int(reference["pareto_only_count"])
    ladder=reference["budget_ladder_summary"]
    median=reference["expansion_ratio_median"]
    p90=reference["expansion_ratio_p90"]
    gate=(
        reference["archive_verification"]["all_hashes_match"]
        and int(reference["contributing_dataset_count"])>=5
        and int(reference["base_state_count"])>=50
        and profiled>=150
        and pareto>=int(0.9*profiled)
        and both>=40
        and int(reference["plan_mismatch_count"])==0
        and pareto_only>=25
        and not mismatches
        and exact_matches==profiled
        and int(reference["dominated_queries_removed"])>=1000
        and int(reference["root_incomparable_classes"])>0
        and median is not None and float(median)>=10.0
        and p90 is not None and float(p90)>=30.0
        and int(ladder["50000"]["pareto_solved"])>=int(ladder["50000"]["plain_solved"])+20
    )
    result={
        "status":"external_conditioned_blind_pass" if gate else "external_conditioned_blind_rejected",
        "development_gate":gate,
        "claim_scope":"Fresh official UCI archives, the frozen v0.60 solver-independent generator, matched exact Python solvers and an independently implemented Rust replay. A pass is strong preregistered external-data algorithmic evidence, not independent peer review, publication novelty or a world-first/world-class claim.",
        "archive_lock_digest":reference["archive_lock_digest"],
        "parent_v60_digest":reference["parent_v60_digest"],
        "frozen_external_digest":reference["frozen_external_digest"],
        "contributing_dataset_count":reference["contributing_dataset_count"],
        "base_state_count":reference["base_state_count"],
        "profiled_state_count":profiled,
        "pareto_solved_count":pareto,
        "both_solved_count":both,
        "pareto_only_count":pareto_only,
        "plan_mismatch_count":reference["plan_mismatch_count"],
        "dominated_queries_removed":reference["dominated_queries_removed"],
        "root_incomparable_classes":reference["root_incomparable_classes"],
        "expansion_ratio_median":median,
        "expansion_ratio_p90":p90,
        "budget_ladder_summary":ladder,
        "rust_total_milliseconds":rust.get("total_milliseconds"),
        "rust_exact_match_count":exact_matches,
        "rust_mismatch_count":len(mismatches),
        "rust_mismatches":mismatches,
        "dataset_summaries":reference["dataset_summaries"],
        "archive_verification":reference["archive_verification"],
        "protocol":reference["protocol"],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--rust",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=validate(args.reference,args.rust,args.output)
    print(json.dumps({"status":result["status"],"gate":result["development_gate"],"datasets":result["contributing_dataset_count"],"base_states":result["base_state_count"],"profiled_states":result["profiled_state_count"],"pareto_solved":result["pareto_solved_count"],"plain_solved":result["both_solved_count"],"pareto_only":result["pareto_only_count"],"rust_mismatches":result["rust_mismatch_count"]},indent=2))


if __name__=="__main__":
    main()
