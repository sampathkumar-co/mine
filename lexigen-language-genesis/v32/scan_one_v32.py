from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V30 = HERE.parent / "v30"
V25 = HERE.parent / "v25"
for path in (HERE, V30, V25):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from independent_runtime_v32 import IndependentRuntimeError, evaluate_independent, normalize_grid
from memoized_evaluator_v32 import evaluate_candidates_memoized
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


def seed_for(namespace: str, task_id: str, index: int) -> int:
    text = f"{namespace}:{task_id}:{index}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


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
        seed = seed_for("lexigen-v32-demonstration", task_id, attempts)
        pair, status = generate_case(arcgen_root, task_id, seed, timeout_seconds)
        attempts += 1
        if pair is None:
            failures.append(status)
            continue
        try:
            source = normalize_grid(pair["input"])
            target = normalize_grid(pair["output"])
            examples.append((source, target))
        except Exception as error:
            failures.append({
                "status": "invalid_pair",
                "seed": seed,
                "message": str(error),
            })
    return examples, {
        "attempts": attempts,
        "accepted": len(examples),
        "timeouts": sum(1 for item in failures if item["status"] == "timeout"),
        "failures": sum(1 for item in failures if item["status"] != "timeout"),
        "failure_examples": failures,
    }


def primary_output(ast: dict[str, Any], source: Any, parameters: dict[str, int]) -> Any:
    return normalize_grid(eval_ast(ast, source, parameters))


def independent_output(ast: dict[str, Any], source: Any, parameters: dict[str, int]) -> Any:
    return normalize_grid(evaluate_independent(ast, source, parameters))


def selected_ast(
    grammar: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    index = int(selected_candidate["structural_index"])
    candidate = grammar["candidates"][index]
    if candidate["ast_sha256"] != selected_candidate["ast_sha256"]:
        raise RuntimeError("selected candidate AST hash mismatch")
    return candidate["ast"]


def run_fresh_gate(
    arcgen_root: Path,
    task_id: str,
    ast: dict[str, Any],
    parameters: dict[str, int],
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
        "passed_cases": 0,
    }
    records: list[dict[str, Any]] = []
    for case_index in range(case_count):
        seed = seed_for("lexigen-v32-fresh", task_id, case_index)
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
            primary = primary_output(ast, source, parameters)
        except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            totals["primary_runtime_errors"] += 1
            record.update({"passed": False, "primary_status": type(error).__name__})
            records.append(record)
            continue
        try:
            independent = independent_output(ast, source, parameters)
        except (IndependentRuntimeError, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            totals["independent_runtime_errors"] += 1
            record.update({"passed": False, "independent_status": type(error).__name__})
            records.append(record)
            continue
        runtime_agrees = primary == independent
        target_matches = primary == target and independent == target
        if not runtime_agrees:
            totals["runtime_disagreements"] += 1
        if not target_matches:
            totals["target_mismatches"] += 1
        passed = runtime_agrees and target_matches
        if passed:
            totals["passed_cases"] += 1
        record.update({
            "pair_sha256": sha256_json({"input": source, "output": target}),
            "primary_sha256": sha256_json(primary),
            "independent_sha256": sha256_json(independent),
            "target_sha256": sha256_json(target),
            "runtime_agrees": runtime_agrees,
            "target_matches": target_matches,
            "passed": passed,
        })
        records.append(record)

    failures = {
        key: value for key, value in totals.items()
        if key not in {"requested_cases", "generated_cases", "passed_cases"}
    }
    passed = (
        totals["generated_cases"] == case_count
        and totals["passed_cases"] == case_count
        and all(value == 0 for value in failures.values())
    )
    return {
        "schema": "lexigen-v32-immediate-fresh-gate-v1",
        "case_count": case_count,
        "selected_ast_sha256": sha256_json(ast),
        "parameters": parameters,
        "totals": totals,
        "passed": passed,
        "case_records": records,
    }


def verify_bindings(
    v32_precommit_path: Path,
    v32_precommit: dict[str, Any],
    v30_precommit_path: Path,
    v30_precommit: dict[str, Any],
    grammar_path: Path,
    grammar: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    source = v32_precommit["source"]
    if sha256_file(v32_precommit_path) != "555194c0d7d35caab361e81a02bd79002fdb3f1837e4180b644ac1f31ffdbe2e":
        raise RuntimeError("v32 precommit hash mismatch")
    if sha256_file(v30_precommit_path) != source["v30_precommit_sha256"]:
        raise RuntimeError("v30 precommit hash mismatch")
    if sha256_file(manifest_path) != source["v30_grammar_manifest_sha256"]:
        raise RuntimeError("v30 grammar manifest hash mismatch")
    if grammar["precommit_sha256"] != sha256_json(v30_precommit):
        raise RuntimeError("generated grammar is not bound to v30 precommit")
    if grammar["candidate_sha256"] != source["candidate_sequence_sha256"]:
        raise RuntimeError("candidate sequence hash mismatch")
    if manifest["candidate_sha256"] != source["candidate_sequence_sha256"]:
        raise RuntimeError("manifest candidate sequence hash mismatch")
    if len(grammar["candidates"]) != int(source["structural_candidate_count"]):
        raise RuntimeError("structural candidate count mismatch")
    if sha256_file(grammar_path) != manifest["grammar_file_sha256"]:
        raise RuntimeError("generated grammar file hash mismatch")
    if sha256_file(HERE / "memoized_evaluator_v32.py") != source["memoized_recovery_scanner_sha256"]:
        raise RuntimeError("memoized evaluator source hash mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    v32_precommit_path = HERE / "V32_PRECOMMIT.json"
    v30_precommit_path = V30 / "V30_PRECOMMIT.json"
    grammar_path = V30 / "V30_GRAMMAR.json"
    manifest_path = V30 / "V30_GRAMMAR_MANIFEST.json"
    v32_precommit = load(v32_precommit_path)
    v30_precommit = load(v30_precommit_path)
    grammar = load(grammar_path)
    manifest = load(manifest_path)
    verify_bindings(
        v32_precommit_path,
        v32_precommit,
        v30_precommit_path,
        v30_precommit,
        grammar_path,
        grammar,
        manifest_path,
        manifest,
    )
    allowed = list(v32_precommit["fresh_identity_selection"]["task_ids"])
    if args.task_id not in allowed:
        raise RuntimeError("task identity is outside the frozen v32 set")

    demo = v32_precommit["demonstration_gate"]
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
        "schema": "lexigen-v32-task-report-v1",
        "task_id": args.task_id,
        "precommit_sha256": sha256_file(v32_precommit_path),
        "v30_precommit_sha256": sha256_file(v30_precommit_path),
        "grammar_sha256": sha256_file(grammar_path),
        "grammar_manifest_sha256": sha256_file(manifest_path),
        "accepted_examples": len(examples),
        "generation": generation,
        "demonstration_sha256": sha256_json(demonstrations),
    }
    if len(examples) != int(demo["examples_per_task"]):
        report = {
            **base_report,
            "status": "generator_invalid",
            "enumeration": None,
            "selected_candidate": None,
            "fresh_gate": None,
        }
    else:
        enumeration = evaluate_candidates_memoized(
            examples,
            grammar,
            v30_precommit,
        )
        selected = enumeration["selected_candidate"]
        if selected is None:
            status = "no_program"
            fresh_gate = None
        else:
            ast = selected_ast(grammar, selected)
            fresh = v32_precommit["immediate_fresh_gate"]
            fresh_gate = run_fresh_gate(
                args.arcgen_root,
                args.task_id,
                ast,
                {str(key): int(value) for key, value in selected["parameters"].items()},
                int(fresh["case_count"]),
                int(fresh["per_generation_timeout_seconds"]),
            )
            status = "fresh_pass" if fresh_gate["passed"] else "fresh_fail"
        report = {
            **base_report,
            "status": status,
            "enumeration": enumeration,
            "selected_candidate": selected,
            "fresh_gate": fresh_gate,
        }

    write(args.output, report)
    summary: dict[str, Any] = {
        "task_id": args.task_id,
        "status": report["status"],
        "accepted_examples": report["accepted_examples"],
        "fresh_passed": bool(
            report["fresh_gate"] and report["fresh_gate"]["passed"]
        ),
    }
    if report["enumeration"] is not None:
        summary.update({
            "candidates_tested": report["enumeration"]["concrete_candidates_tested"],
            "exact_candidates": report["enumeration"]["exact_candidate_count"],
            "selected_candidate": report["selected_candidate"],
        })
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
