from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from latent_runtime_v13 import as_grid, execute_program
from latent_synthesizer_v13 import synthesize_latent

GATES = (
    "v12-campaign-01",
    "v12-campaign-02",
    "v12-campaign-03",
    "v12-campaign-06",
    "v12-campaign-09",
    "v12-campaign-10",
    "v12-campaign-11",
    "v12-campaign-12",
    "v12-campaign-13",
    "v12-campaign-14",
    "v12-campaign-15",
    "v12-campaign-17",
    "v12-campaign-19",
    "v12-campaign-20",
)


def load_examples(root: Path, gate: str):
    payload = json.loads((root / gate / "redacted-task.json").read_text(encoding="utf-8"))
    examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in payload["train"]]
    return payload, examples


def run(evidence_root: Path, arcgen_root: Path, holdout_count: int) -> None:
    sys.path.insert(0, str(arcgen_root))
    import task_list  # type: ignore

    registry = task_list.task_list()
    rows = []
    for gate_index, gate in enumerate(GATES):
        payload, examples = load_examples(evidence_root, gate)
        result = synthesize_latent(examples)
        program = result.program
        training_exact = bool(program and all(execute_program(program, source) == target for source, target in examples))
        failures = []
        if program:
            generator, _ = registry[payload["selected_task_id"]]
            for offset in range(holdout_count):
                seed = 230_000 + gate_index * 20_000 + offset
                random.seed(seed)
                item = generator()
                source, target = as_grid(item["input"]), as_grid(item["output"])
                if execute_program(program, source) != target:
                    failures.append(seed)
                    if len(failures) >= 20:
                        break
        row = {
            "gate": gate,
            "task": payload["selected_task_id"],
            "program": program and program["name"],
            "operator": program and program["operator"],
            "candidates_tested": result.candidates_tested,
            "exact_candidates": result.exact_candidate_count,
            "training_exact": training_exact,
            "holdout_count": holdout_count,
            "holdout_failures": failures,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    gate = {
        "all_fourteen_programs_found": all(row["program"] for row in rows),
        "all_training_exact": all(row["training_exact"] for row in rows),
        "all_fresh_holdouts_exact": all(not row["holdout_failures"] for row in rows),
        "distinct_operator_families": len({row["operator"] for row in rows}) >= 10,
    }
    print(json.dumps({"gate": gate, "rows": rows}, indent=2, sort_keys=True))
    if not all(gate.values()):
        raise AssertionError(f"v13 multifamily gate failed: {gate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--holdout-count", type=int, default=100)
    args = parser.parse_args()
    run(args.evidence_root, args.arcgen_root, args.holdout_count)


if __name__ == "__main__":
    main()
