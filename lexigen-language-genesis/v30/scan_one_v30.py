from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
if str(V25) not in sys.path:
    sys.path.insert(0, str(V25))

from runtime_v25 import RuntimeV25Error, as_grid, canonical, eval_ast, sha256_json


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def seed_for(task_id: str, attempt: int) -> int:
    text = f"lexigen-v30-validation:{task_id}:{attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_examples(
    task_id: str,
    arcgen_root: Path,
    count: int,
    attempts_limit: int,
    timeout_seconds: int,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    examples: list[tuple[Any, Any]] = []
    attempts = timeouts = failures = 0
    failure_examples: list[dict[str, Any]] = []
    worker = V25 / "generate_case_v25.py"
    while len(examples) < count and attempts < attempts_limit:
        seed = seed_for(task_id, attempts)
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
                failure_examples.append({
                    "seed": seed,
                    "type": "subprocess_error",
                    "stderr": completed.stderr[-500:],
                })
            continue
        try:
            pair = json.loads(completed.stdout)
            examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
        except Exception as error:
            failures += 1
            if len(failure_examples) < 5:
                failure_examples.append({
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                })
    return examples, {
        "attempts": attempts,
        "timeouts": timeouts,
        "failures": failures,
        "failure_examples": failure_examples,
    }


def contains_param_color(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("op") == "param_color":
            return True
        return any(contains_param_color(child) for child in value.values())
    if isinstance(value, list):
        return any(contains_param_color(child) for child in value)
    return False


def assignments(ast: dict[str, Any], colors: list[int]) -> Iterator[dict[str, int]]:
    if contains_param_color(ast):
        for color in colors:
            yield {"c0": int(color)}
    else:
        yield {}


def evaluate_candidates(
    examples: list[tuple[Any, Any]],
    grammar: dict[str, Any],
    precommit: dict[str, Any],
) -> dict[str, Any]:
    sources = tuple(source for source, _ in examples)
    targets = tuple(target for _, target in examples)
    task_nontrivial = any(source != target for source, target in examples)
    colors = [int(value) for value in precommit["enumeration"]["literal_color_domain"]]
    maximum = int(precommit["enumeration"]["maximum_concrete_candidates_per_task"])
    tested = runtime_invalid = identity_candidates = 0
    exact: list[dict[str, Any]] = []

    for structural_index, candidate in enumerate(grammar["candidates"]):
        ast = candidate["ast"]
        for parameters in assignments(ast, colors):
            if tested >= maximum:
                return {
                    "schema": "lexigen-v30-task-enumeration-v1",
                    "task_nontrivial": task_nontrivial,
                    "concrete_candidates_tested": tested,
                    "runtime_invalid_candidates": runtime_invalid,
                    "identity_candidates_rejected": identity_candidates,
                    "exact_candidate_count": len(exact),
                    "exact_candidates": exact,
                    "selected_candidate": exact[0] if exact else None,
                    "candidate_cap_reached": True,
                }
            tested += 1
            try:
                outputs = tuple(eval_ast(ast, source, parameters) for source in sources)
            except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError):
                runtime_invalid += 1
                continue
            if outputs == sources:
                identity_candidates += 1
                continue
            if task_nontrivial and outputs == targets:
                exact.append({
                    "structural_index": structural_index,
                    "depth": int(candidate["depth"]),
                    "nodes": int(candidate["nodes"]),
                    "ast_sha256": candidate["ast_sha256"],
                    "parameters": parameters,
                    "concrete_program_sha256": sha256_json({
                        "ast": ast,
                        "parameters": parameters,
                    }),
                })

    return {
        "schema": "lexigen-v30-task-enumeration-v1",
        "task_nontrivial": task_nontrivial,
        "concrete_candidates_tested": tested,
        "runtime_invalid_candidates": runtime_invalid,
        "identity_candidates_rejected": identity_candidates,
        "exact_candidate_count": len(exact),
        "exact_candidates": exact,
        "selected_candidate": exact[0] if exact else None,
        "candidate_cap_reached": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = HERE / "V30_PRECOMMIT.json"
    grammar_path = HERE / "V30_GRAMMAR.json"
    manifest_path = HERE / "V30_GRAMMAR_MANIFEST.json"
    precommit = load(precommit_path)
    grammar = load(grammar_path)
    manifest = load(manifest_path)
    if args.task_id not in precommit["validation_task_ids"]:
        raise RuntimeError("task identity is outside the frozen v30 validation split")
    if grammar["precommit_sha256"] != sha256_json(precommit):
        raise RuntimeError("grammar is not bound to the frozen precommit")
    if manifest["precommit_sha256"] != sha256_file(precommit_path):
        raise RuntimeError("manifest is not bound to the frozen precommit")
    if manifest["grammar_file_sha256"] != sha256_file(grammar_path):
        raise RuntimeError("generated grammar does not match the frozen manifest")
    if manifest["candidate_sha256"] != grammar["candidate_sha256"]:
        raise RuntimeError("candidate sequence does not match the frozen manifest")

    examples, generation = generate_examples(
        args.task_id,
        args.arcgen_root,
        int(precommit["examples_per_task"]),
        int(precommit["generator_attempts_per_task"]),
        int(precommit["per_generation_timeout_seconds"]),
    )
    demonstrations = [
        {"input": source, "output": target}
        for source, target in examples
    ]
    if len(examples) != int(precommit["examples_per_task"]):
        report = {
            "schema": "lexigen-v30-task-scan-v1",
            "task_id": args.task_id,
            "status": "generator_invalid",
            "accepted_examples": len(examples),
            "generation": generation,
            "demonstration_sha256": sha256_json(demonstrations),
            "precommit_sha256": sha256_file(precommit_path),
            "grammar_sha256": sha256_file(grammar_path),
            "grammar_manifest_sha256": sha256_file(manifest_path),
            "enumeration": None,
        }
    else:
        enumeration = evaluate_candidates(examples, grammar, precommit)
        report = {
            "schema": "lexigen-v30-task-scan-v1",
            "task_id": args.task_id,
            "status": "completed",
            "accepted_examples": len(examples),
            "generation": generation,
            "demonstration_sha256": sha256_json(demonstrations),
            "precommit_sha256": sha256_file(precommit_path),
            "grammar_sha256": sha256_file(grammar_path),
            "grammar_manifest_sha256": sha256_file(manifest_path),
            "enumeration": enumeration,
        }
    write(args.output, report)
    summary = {
        "task_id": args.task_id,
        "status": report["status"],
        "accepted_examples": report["accepted_examples"],
    }
    if report["enumeration"] is not None:
        summary.update({
            "candidates_tested": report["enumeration"]["concrete_candidates_tested"],
            "exact_candidates": report["enumeration"]["exact_candidate_count"],
            "candidate_cap_reached": report["enumeration"]["candidate_cap_reached"],
        })
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
