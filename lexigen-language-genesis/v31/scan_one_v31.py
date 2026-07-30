from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
if str(V25) not in sys.path:
    sys.path.insert(0, str(V25))

from runtime_v25 import RuntimeV25Error, eval_ast


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_grid(value: Any) -> tuple[tuple[int, ...], ...]:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("invalid grid")
    return grid


def seed_for(namespace: str, task_id: str, index: int) -> int:
    text = f"{namespace}:{task_id}:{index}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def independent_background(grid: tuple[tuple[int, ...], ...]) -> int:
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return min(counts, key=lambda color: (-counts[color], color))


def independent_execute(
    grid: tuple[tuple[int, ...], ...],
    color: int,
) -> tuple[tuple[int, ...], ...]:
    bg = independent_background(grid)
    return tuple(
        tuple(cell if cell == bg else int(color) for cell in row)
        for row in grid
    )


def verify_relation(
    source: tuple[tuple[int, ...], ...],
    output: tuple[tuple[int, ...], ...],
    color: int,
) -> bool:
    if len(source) != len(output) or any(len(a) != len(b) for a, b in zip(source, output)):
        return False
    return output == independent_execute(source, color)


def generate_case(
    arcgen_root: Path,
    task_id: str,
    seed: int,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    worker = V25 / "generate_case_v25.py"
    command = [
        sys.executable,
        str(worker),
        "--arcgen-root", str(arcgen_root),
        "--task-id", task_id,
        "--seed", str(seed),
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
        return None, {"status": "timeout", "seed": seed}
    if completed.returncode != 0:
        return None, {
            "status": "subprocess_error",
            "seed": seed,
            "stderr": completed.stderr[-500:],
        }
    try:
        return json.loads(completed.stdout), {"status": "ok", "seed": seed}
    except Exception as error:
        return None, {
            "status": "decode_error",
            "seed": seed,
            "message": str(error),
        }


def generate_demonstrations(
    arcgen_root: Path,
    task_id: str,
    count: int,
    attempts_limit: int,
    timeout_seconds: int,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    examples: list[tuple[Any, Any]] = []
    failures: list[dict[str, Any]] = []
    attempts = 0
    while len(examples) < count and attempts < attempts_limit:
        seed = seed_for("lexigen-v31-demonstration", task_id, attempts)
        pair, status = generate_case(arcgen_root, task_id, seed, timeout_seconds)
        attempts += 1
        if pair is None:
            failures.append(status)
            continue
        try:
            examples.append((normalize_grid(pair["input"]), normalize_grid(pair["output"])))
        except Exception as error:
            failures.append({"status": "invalid_pair", "seed": seed, "message": str(error)})
    summary = {
        "attempts": attempts,
        "accepted": len(examples),
        "timeouts": sum(1 for item in failures if item["status"] == "timeout"),
        "failures": sum(1 for item in failures if item["status"] != "timeout"),
        "failure_examples": failures,
    }
    return examples, summary


def candidate_output(
    ast: dict[str, Any],
    source: tuple[tuple[int, ...], ...],
    color: int,
) -> tuple[tuple[int, ...], ...]:
    return normalize_grid(eval_ast(ast, source, {"c0": color}))


def match_demonstrations(
    examples: list[tuple[Any, Any]],
    ast: dict[str, Any],
    colors: list[int],
) -> tuple[list[int], int, int]:
    task_nontrivial = any(source != target for source, target in examples)
    exact_colors: list[int] = []
    invalid_candidates = 0
    identity_candidates = 0
    for color in colors:
        outputs: list[tuple[tuple[int, ...], ...]] = []
        try:
            for source, _ in examples:
                outputs.append(candidate_output(ast, source, color))
        except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError):
            invalid_candidates += 1
            continue
        identity = all(output == source for output, (source, _) in zip(outputs, examples))
        if identity:
            identity_candidates += 1
            continue
        if task_nontrivial and all(output == target for output, (_, target) in zip(outputs, examples)):
            exact_colors.append(color)
    return exact_colors, invalid_candidates, identity_candidates


def run_fresh_gate(
    arcgen_root: Path,
    task_id: str,
    ast: dict[str, Any],
    color: int,
    case_count: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    totals = {
        "requested_cases": case_count,
        "generated_cases": 0,
        "generation_timeouts": 0,
        "generation_errors": 0,
        "primary_runtime_errors": 0,
        "independent_runtime_errors": 0,
        "runtime_disagreements": 0,
        "target_mismatches": 0,
        "verifier_rejections": 0,
        "passed_cases": 0,
    }
    records: list[dict[str, Any]] = []
    for case_index in range(case_count):
        seed = seed_for("lexigen-v31-fresh", task_id, case_index)
        pair, generation = generate_case(arcgen_root, task_id, seed, timeout_seconds)
        record: dict[str, Any] = {
            "case_index": case_index,
            "seed": seed,
            "generation_status": generation["status"],
        }
        if pair is None:
            if generation["status"] == "timeout":
                totals["generation_timeouts"] += 1
            else:
                totals["generation_errors"] += 1
            record.update({"passed": False, "generation": generation})
            records.append(record)
            continue
        totals["generated_cases"] += 1
        try:
            source = normalize_grid(pair["input"])
            target = normalize_grid(pair["output"])
        except Exception as error:
            totals["generation_errors"] += 1
            record.update({"passed": False, "message": str(error)})
            records.append(record)
            continue
        try:
            primary = candidate_output(ast, source, color)
        except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            totals["primary_runtime_errors"] += 1
            record.update({"passed": False, "primary_status": type(error).__name__})
            records.append(record)
            continue
        try:
            independent = independent_execute(source, color)
        except Exception as error:
            totals["independent_runtime_errors"] += 1
            record.update({"passed": False, "independent_status": type(error).__name__})
            records.append(record)
            continue
        runtime_agrees = primary == independent
        target_matches = primary == target and independent == target
        verifier_accepts = verify_relation(source, primary, color) and verify_relation(source, target, color)
        if not runtime_agrees:
            totals["runtime_disagreements"] += 1
        if not target_matches:
            totals["target_mismatches"] += 1
        if not verifier_accepts:
            totals["verifier_rejections"] += 1
        passed = runtime_agrees and target_matches and verifier_accepts
        if passed:
            totals["passed_cases"] += 1
        record.update({
            "pair_sha256": sha256_json({"input": source, "output": target}),
            "primary_sha256": sha256_json(primary),
            "independent_sha256": sha256_json(independent),
            "target_sha256": sha256_json(target),
            "runtime_agrees": runtime_agrees,
            "target_matches": target_matches,
            "verifier_accepts": verifier_accepts,
            "passed": passed,
        })
        records.append(record)

    failure_fields = {
        key: value for key, value in totals.items()
        if key not in {"requested_cases", "generated_cases", "passed_cases"}
    }
    passed = (
        totals["generated_cases"] == case_count
        and totals["passed_cases"] == case_count
        and all(value == 0 for value in failure_fields.values())
    )
    return {
        "schema": "lexigen-v31-immediate-fresh-gate-v1",
        "color": color,
        "case_count": case_count,
        "totals": totals,
        "passed": passed,
        "case_records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = HERE / "V31_PRECOMMIT.json"
    precommit = load(precommit_path)
    allowed = list(precommit["fresh_identity_selection"]["task_ids"])
    if args.task_id not in allowed:
        raise RuntimeError("task identity is outside the frozen v31 set")
    demo = precommit["demonstration_gate"]
    examples, generation = generate_demonstrations(
        args.arcgen_root,
        args.task_id,
        int(demo["examples_per_task"]),
        int(demo["generator_attempts_per_task"]),
        int(demo["per_generation_timeout_seconds"]),
    )
    demonstrations = [
        {"input": source, "output": target}
        for source, target in examples
    ]
    base_report = {
        "schema": "lexigen-v31-task-report-v1",
        "task_id": args.task_id,
        "precommit_sha256": sha256_file(precommit_path),
        "accepted_examples": len(examples),
        "generation": generation,
        "demonstration_sha256": sha256_json(demonstrations),
        "candidate_count": 10,
    }
    if len(examples) != int(demo["examples_per_task"]):
        report = {
            **base_report,
            "status": "generator_invalid",
            "exact_colors": [],
            "invalid_candidate_count": 0,
            "identity_candidate_count": 0,
            "selected_color": None,
            "fresh_gate": None,
        }
    else:
        ast = precommit["motif"]["ast"]
        colors = [int(value) for value in precommit["motif"]["candidate_colors"]]
        exact_colors, invalid_count, identity_count = match_demonstrations(
            examples, ast, colors
        )
        if len(exact_colors) == 0:
            status = "no_program"
            selected_color = None
            fresh_gate = None
        elif len(exact_colors) > 1:
            status = "ambiguous"
            selected_color = None
            fresh_gate = None
        else:
            selected_color = exact_colors[0]
            fresh = precommit["immediate_fresh_gate"]
            fresh_gate = run_fresh_gate(
                args.arcgen_root,
                args.task_id,
                ast,
                selected_color,
                int(fresh["case_count"]),
                int(fresh["per_generation_timeout_seconds"]),
            )
            status = "fresh_pass" if fresh_gate["passed"] else "fresh_fail"
        report = {
            **base_report,
            "status": status,
            "exact_colors": exact_colors,
            "invalid_candidate_count": invalid_count,
            "identity_candidate_count": identity_count,
            "selected_color": selected_color,
            "fresh_gate": fresh_gate,
        }

    write(args.output, report)
    print(json.dumps({
        "task_id": args.task_id,
        "status": report["status"],
        "accepted_examples": report["accepted_examples"],
        "exact_colors": report["exact_colors"],
        "fresh_passed": bool(report["fresh_gate"] and report["fresh_gate"]["passed"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
