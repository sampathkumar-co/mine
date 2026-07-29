from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from runtime_v24 import RuntimeV24Error, as_grid, canonical, execute, sha256_json

BASE_MODES = ("input", "background_canvas")
COMPONENT_FILTERS = ("all", "largest", "smallest", "singletons", "non_singletons")
TRANSFORMS = (
    "identity",
    "grid_flip_h",
    "grid_flip_v",
    "grid_rotate180",
    "bbox_reflect_left",
    "bbox_reflect_right",
    "bbox_reflect_top",
    "bbox_reflect_bottom",
)
REGION_MODES = ("points", "bbox_fill", "bbox_border", "row_span", "col_span", "connect_aligned")
COMBINE_MODES = ("mapped_only", "source_union_mapped")
PAINT_MODES = ("source_colour", "literal_colour")
AXES = ("rows", "columns")
DIRECTIONS = ("start", "end")
EXPECTED_CANDIDATES = 272_008


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def seed_for(split: str, task_id: str, attempt: int) -> int:
    text = f"lexigen-v24:{split}:{task_id}:{attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def generate_examples(task_id: str, split: str, arcgen_root: Path, count: int, attempts_limit: int, timeout: int):
    examples = []
    attempts = 0
    timeouts = 0
    failures = 0
    failure_examples = []
    worker = HERE / "generate_case_v24.py"
    while len(examples) < count and attempts < attempts_limit:
        seed = seed_for(split, task_id, attempts)
        attempts += 1
        command = [
            sys.executable,
            str(worker),
            "--arcgen-root",
            str(arcgen_root),
            "--task-id",
            task_id,
            "--seed",
            str(seed),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            timeouts += 1
            if len(failure_examples) < 5:
                failure_examples.append({"seed": seed, "type": "timeout"})
            continue
        if completed.returncode != 0:
            failures += 1
            if len(failure_examples) < 5:
                failure_examples.append({"seed": seed, "type": "subprocess_error", "stderr": completed.stderr[-500:]})
            continue
        try:
            pair = json.loads(completed.stdout)
            examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
        except Exception as error:
            failures += 1
            if len(failure_examples) < 5:
                failure_examples.append({"seed": seed, "type": type(error).__name__, "message": str(error)})
    return examples, {
        "attempts": attempts,
        "timeouts": timeouts,
        "failures": failures,
        "failure_examples": failure_examples,
    }


def paint_structures() -> Iterable[dict[str, Any]]:
    for base_mode in BASE_MODES:
        for component_filter in COMPONENT_FILTERS:
            for transform in TRANSFORMS:
                for region_mode in REGION_MODES:
                    for combine_mode in COMBINE_MODES:
                        for paint_mode in PAINT_MODES:
                            yield {
                                "op": "paint_edit",
                                "base_mode": base_mode,
                                "component_filter": component_filter,
                                "transform": transform,
                                "region_mode": region_mode,
                                "combine_mode": combine_mode,
                                "paint_mode": paint_mode,
                            }


def classifier_structures() -> Iterable[dict[str, Any]]:
    for component_filter in COMPONENT_FILTERS:
        for relation in TRANSFORMS:
            for base_mode in BASE_MODES:
                yield {
                    "op": "relational_classify",
                    "base_mode": base_mode,
                    "component_filter": component_filter,
                    "relation": relation,
                }


def gravity_structures() -> Iterable[dict[str, Any]]:
    for axis in AXES:
        for direction in DIRECTIONS:
            for base_mode in BASE_MODES:
                yield {
                    "op": "gravity_pack",
                    "axis": axis,
                    "direction": direction,
                    "base_mode": base_mode,
                }


def exact(program: dict[str, Any], examples) -> tuple[bool, bool]:
    try:
        return all(execute(program, source) == target for source, target in examples), False
    except (RuntimeV24Error, ValueError, IndexError, KeyError, TypeError):
        return False, True


def add_survivor(index: dict[str, dict[str, Any]], structure: dict[str, Any], arguments: dict[str, int]) -> None:
    key = sha256_json(structure)
    entry = index.setdefault(key, {"structure_sha256": key, "structure": structure, "arguments": []})
    entry["arguments"].append(arguments)


def scan(examples):
    tested = 0
    invalid = 0
    exact_programs = 0
    survivors: dict[str, dict[str, Any]] = {}
    for structure in paint_structures():
        for source_colour in range(10):
            for paint_colour in range(10):
                tested += 1
                program = dict(structure, source_colour=source_colour, paint_colour=paint_colour)
                is_exact, is_invalid = exact(program, examples)
                invalid += int(is_invalid)
                if is_exact:
                    exact_programs += 1
                    add_survivor(survivors, structure, {"source_colour": source_colour, "paint_colour": paint_colour})
    for structure in classifier_structures():
        for source_colour in range(10):
            for equal_colour in range(10):
                for unequal_colour in range(10):
                    tested += 1
                    program = dict(
                        structure,
                        source_colour=source_colour,
                        equal_colour=equal_colour,
                        unequal_colour=unequal_colour,
                    )
                    is_exact, is_invalid = exact(program, examples)
                    invalid += int(is_invalid)
                    if is_exact:
                        exact_programs += 1
                        add_survivor(
                            survivors,
                            structure,
                            {
                                "source_colour": source_colour,
                                "equal_colour": equal_colour,
                                "unequal_colour": unequal_colour,
                            },
                        )
    for structure in gravity_structures():
        tested += 1
        is_exact, is_invalid = exact(structure, examples)
        invalid += int(is_invalid)
        if is_exact:
            exact_programs += 1
            add_survivor(survivors, structure, {})
    if tested != EXPECTED_CANDIDATES:
        raise RuntimeError(f"candidate denominator changed: {tested}")
    for entry in survivors.values():
        entry["arguments"] = sorted(entry["arguments"], key=canonical)
    return tested, invalid, exact_programs, [survivors[key] for key in sorted(survivors)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split", choices=("discovery", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precommit = load(HERE / "V24_PRECOMMIT.json")
    allowed = precommit[f"{args.split}_task_ids"]
    if args.task_id not in allowed:
        raise RuntimeError("task identity is not in the frozen split")
    examples, generation = generate_examples(
        args.task_id,
        args.split,
        args.arcgen_root,
        int(precommit["examples_per_task"]),
        int(precommit["generator_attempts_per_task"]),
        int(precommit["per_generation_timeout_seconds"]),
    )
    if len(examples) != int(precommit["examples_per_task"]):
        report = {
            "schema": "lexigen-v24-task-scan-v1",
            "task_id": args.task_id,
            "split": args.split,
            "status": "generator_invalid",
            "accepted_examples": len(examples),
            "generation": generation,
            "candidate_programs_tested": 0,
            "runtime_invalid_candidates": 0,
            "exact_complete_programs": 0,
            "exact_structure_count": 0,
            "exact_structures": [],
        }
    else:
        tested, invalid, exact_programs, structures = scan(examples)
        report = {
            "schema": "lexigen-v24-task-scan-v1",
            "task_id": args.task_id,
            "split": args.split,
            "status": "completed",
            "accepted_examples": len(examples),
            "generation": generation,
            "candidate_programs_tested": tested,
            "runtime_invalid_candidates": invalid,
            "exact_complete_programs": exact_programs,
            "exact_structure_count": len(structures),
            "exact_structures": structures,
        }
    write(args.output, report)
    print(json.dumps({
        "task_id": args.task_id,
        "status": report["status"],
        "exact_complete_programs": report["exact_complete_programs"],
        "exact_structure_count": report["exact_structure_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
