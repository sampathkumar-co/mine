from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from scene_runtime_v14 import as_grid
from scene_synthesizer_v14 import synthesize_scene

EVIDENCE = Path(r"C:\Users\SAMPATH\AppData\Local\Temp\lexigen-v13-campaign-engine\lexigen-language-genesis\external\evidence\v13-campaign")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gates", nargs="+", type=int)
    parser.add_argument("--depth", type=int, default=2)
    args = parser.parse_args()
    for gate in args.gates:
        path = EVIDENCE / f"v13-campaign-{gate:02d}" / "redacted-task.json"
        if not path.exists():
            continue
        package = json.loads(path.read_text())
        examples = [(as_grid(pair["input"]), as_grid(pair["output"])) for pair in package["train"]]
        started = time.perf_counter()
        result = synthesize_scene(examples, max_depth=args.depth)
        print(json.dumps({"gate": gate, "task": package["selected_task_id"], "found": result.pipeline is not None,
                          "pipeline": result.pipeline, "tested": result.candidates_tested,
                          "signatures": result.signatures_seen, "inventory": result.inventory_size,
                          "seconds": round(time.perf_counter()-started, 4)}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
