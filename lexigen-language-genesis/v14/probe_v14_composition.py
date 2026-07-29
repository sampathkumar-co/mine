from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V13 = HERE.parent / "v13"
for path in (HERE, V13):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compositional_synthesizer_v14 import synthesize_composition
from latent_runtime_v13 import as_grid

EVIDENCE = Path(r"C:\Users\SAMPATH\AppData\Local\Temp\lexigen-v13-campaign-engine\lexigen-language-genesis\external\evidence\v13-campaign")


def load_examples(gate: int):
    path = EVIDENCE / f"v13-campaign-{gate:02d}" / "redacted-task.json"
    package = json.loads(path.read_text(encoding="utf-8"))
    examples = [(as_grid(pair["input"]), as_grid(pair["output"])) for pair in package["train"]]
    return package["selected_task_id"], examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gates", nargs="+", type=int)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--budget", type=int, default=100_000)
    args = parser.parse_args()
    for gate in args.gates:
        if gate == 34:
            continue
        task, examples = load_examples(gate)
        started = time.perf_counter()
        result = synthesize_composition(examples, max_depth=args.depth, candidate_budget=args.budget)
        elapsed = time.perf_counter() - started
        payload = {
            "gate": gate,
            "task": task,
            "found": result.pipeline is not None,
            "pipeline": result.pipeline,
            "candidates_tested": result.candidates_tested,
            "signatures_seen": result.signatures_seen,
            "inventory_states": result.inventory_states,
            "exact_pipeline_count": result.exact_pipeline_count,
            "seconds": round(elapsed, 4),
        }
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
