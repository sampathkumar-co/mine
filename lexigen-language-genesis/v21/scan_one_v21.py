from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V19R5 = HERE.parent / "v19r5"
V19R3 = HERE.parent / "v19r3"
for folder in (V19R5, V19R3):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from scan_discovery_v19r5 import (
    ACTIONS, BASE_MODES, SET_MODES, _prepare, _render,
    factorized_production, generate_examples, structure_descriptor,
)
from runtime_v19r2 import canonical, sha256_json


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precommit = json.loads((HERE / "V21_PRECOMMIT.json").read_text())
    if args.task_id not in precommit["selected_discovery_task_ids"]:
        raise RuntimeError("task is outside frozen v21 pool")
    examples, attempts, rejections, rejection_types = generate_examples(
        args.task_id, args.arcgen_root,
        int(precommit["examples_per_task"]),
        int(precommit["maximum_generator_attempts_per_task"]),
    )
    if len(examples) != int(precommit["examples_per_task"]):
        write(args.output, {
            "schema": "lexigen-v21-task-scan-v1",
            "task_id": args.task_id,
            "status": "generator_invalid",
            "accepted_examples": len(examples),
            "generator_attempts": attempts,
            "generator_rejections": rejections,
            "rejection_types": rejection_types,
            "exact_complete_programs": 0,
            "productions": [],
        })
        return

    prepared_by_marker = {
        marker: [_prepare(source, target, marker) for source, target in examples]
        for marker in range(10)
    }
    structures = {}
    exact_complete = 0
    for marker in range(10):
        prepared = prepared_by_marker[marker]
        if any(item is None for item in prepared):
            continue
        for background in range(10):
            for left in ACTIONS:
                for right in ACTIONS:
                    for top in ACTIONS:
                        for bottom in ACTIONS:
                            actions = (left, right, top, bottom)
                            indexes = tuple(ACTIONS.index(name) for name in actions)
                            for set_mode in SET_MODES:
                                for base_mode in BASE_MODES:
                                    if all(
                                        _render(item, indexes[item["side"]], set_mode,
                                                base_mode, background) == item["target"]
                                        for item in prepared if item is not None
                                    ):
                                        exact_complete += 1
                                        structure = structure_descriptor(actions, set_mode, base_mode)
                                        key = canonical(structure)
                                        entry = structures.setdefault(key, {
                                            "production": factorized_production(structure),
                                            "arguments": [],
                                        })
                                        entry["arguments"].append({
                                            "marker_colour": marker,
                                            "output_background": background,
                                        })
    productions = []
    for key in sorted(structures):
        entry = structures[key]
        production = entry["production"]
        productions.append({
            "production_sha256": sha256_json(production),
            "production": production,
            "arguments": sorted(entry["arguments"], key=canonical),
        })
    write(args.output, {
        "schema": "lexigen-v21-task-scan-v1",
        "task_id": args.task_id,
        "status": "completed",
        "accepted_examples": len(examples),
        "generator_attempts": attempts,
        "generator_rejections": rejections,
        "rejection_types": rejection_types,
        "exact_complete_programs": exact_complete,
        "exact_production_structures": len(productions),
        "productions": productions,
    })
    print(json.dumps({
        "task_id": args.task_id,
        "exact_complete_programs": exact_complete,
        "exact_production_structures": len(productions),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
