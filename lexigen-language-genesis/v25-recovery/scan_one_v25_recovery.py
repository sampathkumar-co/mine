from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / 'v25'
for folder in (HERE, V25):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from enumerator_v25_recovery import enumerate_programs
from runtime_v25 import as_grid, canonical


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def seed_for(split: str, task_id: str, attempt: int) -> int:
    text = f"lexigen-v25:{split}:{task_id}:{attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def generate_examples(
    task_id: str,
    split: str,
    arcgen_root: Path,
    count: int,
    attempts_limit: int,
    timeout_seconds: int,
):
    examples = []
    attempts = timeouts = failures = 0
    failure_examples = []
    worker = V25 / 'generate_case_v25.py'
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
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timeouts += 1
            if len(failure_examples) < 5:
                failure_examples.append({"seed": seed, "type": "timeout"})
            continue
        if completed.returncode != 0:
            failures += 1
            if len(failure_examples) < 5:
                failure_examples.append(
                    {
                        "seed": seed,
                        "type": "subprocess_error",
                        "stderr": completed.stderr[-500:],
                    }
                )
            continue
        try:
            pair = json.loads(completed.stdout)
            examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
        except Exception as error:
            failures += 1
            if len(failure_examples) < 5:
                failure_examples.append(
                    {
                        "seed": seed,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
    return examples, {
        "attempts": attempts,
        "timeouts": timeouts,
        "failures": failures,
        "failure_examples": failure_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split", choices=("discovery", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = V25 / 'V25_PRECOMMIT.json'
    precommit = load(precommit_path)
    allowed = list(precommit[f"{args.split}_task_ids"])
    if args.task_id not in allowed:
        raise RuntimeError("task identity is outside the frozen split")

    examples, generation = generate_examples(
        args.task_id,
        args.split,
        args.arcgen_root,
        int(precommit["examples_per_task"]),
        int(precommit["generator_attempts_per_task"]),
        int(precommit["per_generation_timeout_seconds"]),
    )
    demonstration_document = [
        {"input": source, "output": target}
        for source, target in examples
    ]
    if len(examples) != int(precommit["examples_per_task"]):
        report = {
            "schema": "lexigen-v25-task-scan-v1",
            "task_id": args.task_id,
            "split": args.split,
            "status": "generator_invalid",
            "accepted_examples": len(examples),
            "generation": generation,
            "demonstration_sha256": sha256_json(demonstration_document),
            "enumeration": None,
        }
    else:
        enumeration = enumerate_programs(
            examples,
            maximum_depth=int(precommit["enumeration"]["maximum_depth"]),
            maximum_unique_per_type_per_depth=int(
                precommit["enumeration"]["maximum_unique_expressions_per_type_per_depth"]
            ),
            maximum_total_unique=int(
                precommit["enumeration"]["maximum_total_unique_expressions_per_task"]
            ),
            maximum_raw_candidates=int(
                precommit["enumeration"]["maximum_raw_candidate_evaluations_per_task"]
            ),
        )
        report = {
            "schema": "lexigen-v25-task-scan-v1",
            "task_id": args.task_id,
            "split": args.split,
            "status": "completed",
            "accepted_examples": len(examples),
            "generation": generation,
            "demonstration_sha256": sha256_json(demonstration_document),
            "enumeration": enumeration,
        }
    write(args.output, report)
    summary = {
        "task_id": args.task_id,
        "status": report["status"],
        "accepted_examples": report["accepted_examples"],
    }
    if report["enumeration"] is not None:
        summary.update(
            {
                "enumeration_complete": report["enumeration"]["enumeration_complete"],
                "exhausted_reason": report["enumeration"]["exhausted_reason"],
                "raw_candidates": report["enumeration"]["raw_candidate_evaluations"],
                "unique_expressions": report["enumeration"]["total_unique_expressions"],
                "exact_programs": report["enumeration"]["exact_concrete_programs"],
                "exact_structures": report["enumeration"]["exact_abstract_structures"],
            }
        )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
