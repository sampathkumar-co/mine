from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from . import external_local_quotient_v41 as v41


def run(seed: int) -> dict[str, object]:
    if seed not in v41.SEEDS:
        raise ValueError("seed is outside the frozen v0.41 schedule")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        verification = v41.download_and_verify(root)
        tasks = v41.load_tasks(root)
        report = v41.run_seed(tasks, seed)
    return {
        "seed_report": report,
        "archive_verification": verification,
        "frozen_configuration": {
            "candidate_threshold": v41.CANDIDATE_THRESHOLD,
            "parent_compiler_digest": v41.FROZEN_PARENT_DIGEST,
            "seeds": list(v41.SEEDS),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "seed": args.seed,
        "candidate_gate": report["seed_report"]["candidate_gate"],
        "strict_wins": report["seed_report"]["strict_wins"],
        "large_domain_strict_wins": report["seed_report"][
            "large_domain_strict_wins"
        ],
        "aggregate_total_query_saving": report["seed_report"][
            "aggregate_total_query_saving"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
