from __future__ import annotations

import argparse
import json
from pathlib import Path

from lexigen_omega.development import DevelopmentArchive, DevelopmentObservation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive = DevelopmentArchive()
    tasks: list[dict[str, object]] = []
    for path in sorted(args.results_root.glob("*/BLIND_R1_RESULT.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("campaign") != "LEXIGEN v6 Applicability-Conditioned Causal Transfer Replication":
            raise SystemExit(f"unexpected campaign in {path}")
        if payload.get("stage") != "official_blind_r1_parallel7":
            raise SystemExit(f"unexpected stage in {path}")
        if not bool(payload.get("blind_run_complete", False)):
            continue

        full = payload.get("by_arm", {}).get("v6_full", {})
        transfer_ids = [str(item) for item in full.get("transfer_ids", [])]
        scientifically_valid = (
            int(full.get("valid", 0)) == int(payload.get("blind_records", -1))
            and int(full.get("invalid_output_retries", -1)) == 0
        )
        causal_win = bool(payload.get("baseline_qualified_causal_transfer_win", False))
        for transfer_id in transfer_ids:
            archive.add(
                DevelopmentObservation(
                    task_id=str(payload["task"]),
                    family=str(payload["family"]),
                    mechanism_key=f"v6-transfer:{transfer_id}",
                    reward=1.0 if causal_win else 0.0,
                    source_campaign="LEXIGEN v6 frozen blind causal verdict",
                    scientific=scientifically_valid,
                )
            )
        tasks.append(
            {
                "task": payload["task"],
                "family": payload["family"],
                "transfer_ids": transfer_ids,
                "scientifically_valid": scientifically_valid,
                "baseline_qualified_causal_transfer_win": causal_win,
                "failure_conditions": [
                    key
                    for key, value in payload.get("causal_conditions", {}).items()
                    if value is False
                ],
            }
        )

    ranking = archive.rank_mechanisms(exploration=0.0)
    report = {
        "experiment": "LEXIGEN OMEGA import of completed V6 exposed blind evidence",
        "status": "development_only_never_final_claim_evidence",
        "source_rule": "use V6 frozen transfer IDs and V6's own baseline_qualified_causal_transfer_win verdict; no Omega remapping or new threshold",
        "task_count": len(tasks),
        "scientific_observation_count": len([item for item in archive.observations if item.scientific]),
        "tasks": tasks,
        "mechanism_ranking": [
            {
                "mechanism_key": item.mechanism_key,
                "task_count": item.task_count,
                "family_count": item.family_count,
                "mean_reward": item.mean_reward,
            }
            for item in ranking
        ],
        "claim_boundary": {
            "final_claim_eligible": False,
            "causal_memory_admission": False,
            "purpose": "development search deprioritization/prioritization only"
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "task_count": report["task_count"],
        "scientific_observation_count": report["scientific_observation_count"],
        "mechanism_ranking": report["mechanism_ranking"],
    }, indent=2))


if __name__ == "__main__":
    main()
