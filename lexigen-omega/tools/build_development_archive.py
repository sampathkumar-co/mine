from __future__ import annotations

import argparse
import json
from pathlib import Path

from lexigen_omega.development import DevelopmentArchive, DevelopmentObservation


def seq_key(value: object) -> str:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("primitive sequence must be a list of strings")
    return "seq:" + (">".join(value) if value else "EMPTY")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hindsight", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive = DevelopmentArchive()
    source_reports: list[dict[str, object]] = []
    for path in args.hindsight:
        payload = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(payload.get("source_task", "")).strip()
        if not task_id:
            raise SystemExit(f"{path}: missing source_task")
        preferences = payload.get("mechanism_preferences")
        if not isinstance(preferences, list):
            raise SystemExit(f"{path}: missing mechanism_preferences")
        family = str(payload.get("source_family") or task_id.split("__", 1)[0])
        for row in preferences:
            if not isinstance(row, dict):
                raise SystemExit(f"{path}: malformed preference row")
            archive.add(
                DevelopmentObservation(
                    task_id=task_id,
                    family=family,
                    mechanism_key=seq_key(row["preferred_primitive_sequence"]),
                    reward=1.0,
                    source_campaign=str(payload.get("experiment", "omega-exposed-development")),
                )
            )
            archive.add(
                DevelopmentObservation(
                    task_id=task_id,
                    family=family,
                    mechanism_key=seq_key(row["rejected_primitive_sequence"]),
                    reward=0.0,
                    source_campaign=str(payload.get("experiment", "omega-exposed-development")),
                )
            )
        source_reports.append(
            {
                "path": str(path),
                "task_id": task_id,
                "family": family,
                "preference_count": len(preferences),
            }
        )

    ranking = archive.rank_mechanisms()
    report = {
        "experiment": "LEXIGEN OMEGA exposed development treadmill archive",
        "status": "development_only_never_final_claim_evidence",
        "source_reports": source_reports,
        "observation_count": len(archive.observations),
        "task_count": len({item.task_id for item in archive.observations}),
        "family_count": len({item.family for item in archive.observations}),
        "mechanism_ranking": [
            {
                "mechanism_key": item.mechanism_key,
                "task_count": item.task_count,
                "family_count": item.family_count,
                "mean_reward": item.mean_reward,
                "exploration_bonus": item.exploration_bonus,
                "priority": item.priority,
            }
            for item in ranking
        ],
        "claim_boundary": {
            "final_claim_eligible": False,
            "causal_memory_admission": False,
            "purpose": "development search prioritization only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"task_count": report["task_count"], "mechanisms": len(ranking)}, indent=2))


if __name__ == "__main__":
    main()
