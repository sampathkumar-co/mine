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
V30 = HERE.parent / "v30"
if str(V25) not in sys.path:
    sys.path.insert(0, str(V25))

from runtime_v25 import RuntimeV25Error, eval_ast


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_grid(value: Any) -> tuple[tuple[int, ...], ...]:
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("invalid grid")
    return grid


def seed_for(case_index: int) -> int:
    text = f"lexigen-v30-fresh:9565186b:{case_index}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def independent_background(grid: tuple[tuple[int, ...], ...]) -> int:
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[cell] = counts.get(cell, 0) + 1
    return min(counts, key=lambda color: (-counts[color], color))


def independent_execute(grid: tuple[tuple[int, ...], ...], color: int) -> tuple[tuple[int, ...], ...]:
    background = independent_background(grid)
    return tuple(
        tuple(cell if cell == background else int(color) for cell in row)
        for row in grid
    )


def derive_verifier(ast: dict[str, Any], parameters: dict[str, int]) -> dict[str, Any]:
    expected = {
        "op": "paint",
        "grid": {"op": "input_grid"},
        "points": {"op": "non_background_points"},
        "colour": {"op": "param_color", "name": "c0"},
    }
    if ast != expected:
        raise RuntimeError("frozen candidate AST is outside the supported verifier synthesis rule")
    if set(parameters) != {"c0"}:
        raise RuntimeError("unexpected parameter inventory")
    return {
        "schema": "lexigen-v30-background-preserving-recolor-verifier-v1",
        "background_rule": "most frequent color; smaller color wins ties",
        "foreground_color": int(parameters["c0"]),
        "dimensions_preserved": True,
    }


def verify_relation(
    source: tuple[tuple[int, ...], ...],
    output: tuple[tuple[int, ...], ...],
    verifier: dict[str, Any],
) -> bool:
    if len(source) != len(output) or any(len(a) != len(b) for a, b in zip(source, output)):
        return False
    expected = independent_execute(source, int(verifier["foreground_color"]))
    return output == expected


def generate_case(arcgen_root: Path, task_id: str, seed: int, timeout: int) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    worker = V25 / "generate_case_v25.py"
    command = [sys.executable, str(worker), "--arcgen-root", str(arcgen_root), "--task-id", task_id, "--seed", str(seed)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, {"status": "timeout"}
    if completed.returncode != 0:
        return None, {"status": "subprocess_error", "stderr": completed.stderr[-500:]}
    try:
        return json.loads(completed.stdout), {"status": "ok"}
    except Exception as error:
        return None, {"status": "decode_error", "message": str(error)}


def validate_bindings(precommit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    grammar = load(V30 / "V30_GRAMMAR.json")
    manifest = load(V30 / "V30_GRAMMAR_MANIFEST.json")
    if sha256_file(V30 / "V30_PRECOMMIT.json") != precommit["v30_precommit_sha256"]:
        raise RuntimeError("v30 precommit hash mismatch")
    if sha256_file(V30 / "V30_GRAMMAR.json") != precommit["v30_grammar_sha256"]:
        raise RuntimeError("v30 grammar hash mismatch")
    if sha256_file(V30 / "V30_GRAMMAR_MANIFEST.json") != precommit["v30_grammar_manifest_sha256"]:
        raise RuntimeError("v30 grammar manifest hash mismatch")
    selected = precommit["selected_candidate"]
    candidate = grammar["candidates"][int(selected["structural_index"])]
    for key in ("depth", "nodes", "ast_sha256"):
        if candidate[key] != selected[key]:
            raise RuntimeError(f"selected candidate {key} mismatch")
    if candidate["ast"] != selected["ast"]:
        raise RuntimeError("selected candidate AST mismatch")
    if grammar["candidate_sha256"] != manifest["candidate_sha256"]:
        raise RuntimeError("candidate sequence mismatch")
    parameters = {str(k): int(v) for k, v in selected["parameters"].items()}
    verifier = derive_verifier(candidate["ast"], parameters)
    return candidate["ast"], parameters, verifier


def case_digest(source: Any, target: Any) -> str:
    return sha256_json({"input": source, "output": target})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    precommit_path = HERE / "V30_956_FRESH_PRECOMMIT.json"
    precommit = load(precommit_path)
    ast, parameters, verifier = validate_bindings(precommit)

    fresh = precommit["fresh_validation"]
    case_count = int(fresh["case_count"])
    timeout = int(fresh["per_generation_timeout_seconds"])
    task_id = str(precommit["source_task_id"])
    records: list[dict[str, Any]] = []
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

    for case_index in range(case_count):
        seed = seed_for(case_index)
        pair, generation = generate_case(args.arcgen_root, task_id, seed, timeout)
        record: dict[str, Any] = {"case_index": case_index, "seed": seed, "generation_status": generation["status"]}
        if pair is None:
            if generation["status"] == "timeout":
                totals["generation_timeouts"] += 1
            else:
                totals["generation_errors"] += 1
            record["generation"] = generation
            record["passed"] = False
            records.append(record)
            continue
        totals["generated_cases"] += 1
        try:
            source = normalize_grid(pair["input"])
            target = normalize_grid(pair["output"])
        except Exception as error:
            totals["generation_errors"] += 1
            record.update({"generation_status": "invalid_pair", "message": str(error), "passed": False})
            records.append(record)
            continue

        record["pair_sha256"] = case_digest(source, target)
        try:
            primary = normalize_grid(eval_ast(ast, source, parameters))
        except (RuntimeV25Error, ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            totals["primary_runtime_errors"] += 1
            record.update({"primary_status": type(error).__name__, "passed": False})
            records.append(record)
            continue
        try:
            independent = independent_execute(source, int(parameters["c0"]))
        except Exception as error:
            totals["independent_runtime_errors"] += 1
            record.update({"independent_status": type(error).__name__, "passed": False})
            records.append(record)
            continue

        runtime_agrees = primary == independent
        target_matches = primary == target and independent == target
        verifier_accepts = verify_relation(source, primary, verifier) and verify_relation(source, target, verifier)
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
            "primary_sha256": sha256_json(primary),
            "independent_sha256": sha256_json(independent),
            "target_sha256": sha256_json(target),
            "runtime_agrees": runtime_agrees,
            "target_matches": target_matches,
            "verifier_accepts": verifier_accepts,
            "passed": passed,
        })
        records.append(record)
        if (case_index + 1) % 100 == 0:
            print(json.dumps({"completed": case_index + 1, "passed": totals["passed_cases"]}, sort_keys=True), flush=True)

    fresh_gate_passed = (
        len(records) == case_count
        and totals["generated_cases"] == case_count
        and totals["passed_cases"] == case_count
        and all(value == 0 for key, value in totals.items() if key not in {"requested_cases", "generated_cases", "passed_cases"})
    )
    report = {
        "schema": "lexigen-v30-9565186b-fresh-validation-report-v1",
        "source_task_id": task_id,
        "precommit_sha256": sha256_file(precommit_path),
        "selected_candidate": precommit["selected_candidate"],
        "verifier": verifier,
        "case_count": case_count,
        "totals": totals,
        "fresh_gate_passed": fresh_gate_passed,
        "fresh_case_validation_demonstrated": fresh_gate_passed,
        "repeated_task_level_transfer_demonstrated": False,
        "outside_human_reproduction_completed": False,
        "world_level_breakthrough": False,
        "case_records": records,
    }
    write(args.output, report)
    print(json.dumps({
        "case_count": case_count,
        "passed_cases": totals["passed_cases"],
        "fresh_gate_passed": fresh_gate_passed,
        "report_sha256": sha256_file(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
