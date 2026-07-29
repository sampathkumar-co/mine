from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from . import clean_external_conditioned_v64 as clean_v64
from . import conditioned_cell_frontier_v60 as conditioned
from . import response_cost_export_v57 as export_v57
from . import response_cost_lower_bound_v65 as bounded
from . import response_cost_pareto_v56 as response


PARENT_EVIDENCE = Path(__file__).resolve().parents[3] / "research-evidence" / "mini-origin-v65-pareto-lower-bound-pass.json"
PARENT_DIGEST = "3a1ef2c86c07600a591a900caf920456acecacb158b8faeb8bdb4c6d97d0550a"


def run(
    states_path: Path,
    manifest_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    parent = json.loads(PARENT_EVIDENCE.read_text(encoding="utf-8"))
    if not parent["development_gate"] or parent["evidence_digest"] != PARENT_DIGEST:
        raise RuntimeError("unexpected v0.65 parent evidence")

    tasks, summaries = bounded.load_v64_tasks()
    state_rows = []
    reference_rows = []
    base_states: set[str] = set()
    for task, selected in tasks:
        for allowed, remaining, representatives in selected:
            for seed in conditioned.PROFILE_SEEDS:
                compact = clean_v64.compact_state(
                    task, allowed, remaining, seed
                )
                profile = response.profile_for_task(task, seed)
                result = bounded.LowerBoundParetoPlanner(
                    task, profile, response.BUDGET
                ).result(allowed, remaining)
                state_rows.append(compact)
                base_states.add(str(compact["base_digest"]))
                reference_rows.append({
                    "digest": compact["digest"],
                    "task": task.name,
                    "profile_seed": seed,
                    "structural_partition_representatives": representatives,
                    "plan": list(bounded.exact_plan_tuple(result.plan)),
                    **asdict(result.stats),
                })

    state_rows.sort(key=lambda row: str(row["digest"]))
    reference_rows.sort(key=lambda row: str(row["digest"]))
    export_v57.write_text(state_rows, states_path)
    state_input_sha256 = hashlib.sha256(states_path.read_bytes()).hexdigest()
    manifest = {
        "status": "lower_bound_compiled_input_frozen_v66",
        "format": "mini-origin-response-cost-state-v1",
        "parent_v65_evidence_digest": PARENT_DIGEST,
        "archive_lock_digest": clean_v64.LOCK_DIGEST,
        "repository_registry_digest": clean_v64.REGISTRY_DIGEST,
        "state_count": len(state_rows),
        "base_state_count": len(base_states),
        "profile_seeds": list(conditioned.PROFILE_SEEDS),
        "input_sha256": state_input_sha256,
        "state_digests": [row["digest"] for row in state_rows],
        "dataset_summaries": summaries,
    }
    manifest["manifest_digest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    reference = {
        "status": "lower_bound_python_reference_v66",
        "parent_v65_evidence_digest": PARENT_DIGEST,
        "manifest_digest": manifest["manifest_digest"],
        "state_input_sha256": state_input_sha256,
        "state_count": len(reference_rows),
        "rows": reference_rows,
    }
    reference["reference_digest"] = hashlib.sha256(
        json.dumps(reference, sort_keys=True).encode("utf-8")
    ).hexdigest()
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "reference": reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.states, args.manifest, args.reference)
    print(json.dumps({
        "status": result["reference"]["status"],
        "states": result["reference"]["state_count"],
        "manifest_digest": result["manifest"]["manifest_digest"],
        "reference_digest": result["reference"]["reference_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
