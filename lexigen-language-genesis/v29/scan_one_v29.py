from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V25 = HERE.parent / "v25"
for folder in (HERE, V25):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from antiunify_v29 import instantiate_template
from runtime_v25 import as_grid, canonical, eval_ast, sha256_json


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_for(task_id: str, attempt: int) -> int:
    text = f"lexigen-v29:validation:{task_id}:{attempt}"
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) & 0xFFFFFFFF


def generate_examples(
    task_id: str,
    arcgen_root: Path,
    count: int,
    maximum_attempts: int,
    timeout_seconds: int,
) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
    examples: list[tuple[Any, Any]] = []
    attempts = failures = timeouts = 0
    failure_types: dict[str, int] = {}
    worker = V25 / "generate_case_v25.py"
    while len(examples) < count and attempts < maximum_attempts:
        seed = seed_for(task_id, attempts)
        attempts += 1
        command = [
            sys.executable,
            str(worker),
            "--arcgen-root", str(arcgen_root),
            "--task-id", task_id,
            "--seed", str(seed),
        ]
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timeouts += 1
            continue
        if process.returncode != 0:
            failures += 1
            tail = process.stderr.strip().splitlines()
            name = tail[-1].split(":", 1)[0] if tail else "GeneratorProcessError"
            failure_types[name] = failure_types.get(name, 0) + 1
            continue
        try:
            pair = json.loads(process.stdout)
            examples.append((as_grid(pair["input"]), as_grid(pair["output"])))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            failures += 1
            name = type(error).__name__
            failure_types[name] = failure_types.get(name, 0) + 1
    return examples, {
        "attempts": attempts,
        "failures": failures,
        "timeouts": timeouts,
        "failure_types": failure_types,
    }


def parameter_names(value: Any) -> list[str]:
    found: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            if item.get("op") == "param_color":
                found.add(str(item["name"]))
            for child in item.values():
                visit(child)

    visit(value)
    return sorted(found)


def substitute_colours(value: Any, arguments: dict[str, int]) -> Any:
    if isinstance(value, list):
        return [substitute_colours(item, arguments) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("op") == "param_color":
        name = str(value["name"])
        if name not in arguments:
            raise RuntimeError(f"missing colour argument: {name}")
        return {"op": "literal_color", "value": int(arguments[name])}
    return {key: substitute_colours(item, arguments) for key, item in value.items()}


def candidate_instantiations(template: dict[str, Any]):
    holes = list(template["operator_holes"])
    hole_names = [str(item["name"]) for item in holes]
    hole_choices = [list(item["allowed_choices"]) for item in holes]
    colour_names = parameter_names(template["template_ast"])
    for operator_values in itertools.product(*hole_choices):
        operator_arguments = dict(zip(hole_names, operator_values))
        operator_ast = instantiate_template(template, operator_arguments)
        for colour_values in itertools.product(range(10), repeat=len(colour_names)):
            colour_arguments = dict(zip(colour_names, colour_values))
            concrete = substitute_colours(operator_ast, colour_arguments)
            yield operator_arguments, colour_arguments, concrete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    precommit_path = HERE / "V29_PRECOMMIT.json"
    library_path = HERE / "V29_TEMPLATE_LIBRARY.json"
    precommit = load(precommit_path)
    library = load(library_path)
    if args.task_id not in precommit["validation_task_ids"]:
        raise RuntimeError("task identity is outside frozen validation denominator")
    if library["eligible_template_count"] == 0:
        raise RuntimeError("frozen template library is empty")

    examples, generation = generate_examples(
        args.task_id,
        args.arcgen_root,
        int(precommit["validation_examples_per_task"]),
        int(precommit["maximum_generator_attempts_per_task"]),
        int(precommit["per_generation_timeout_seconds"]),
    )

    report: dict[str, Any] = {
        "schema": "lexigen-v29-validation-task-report-v1",
        "task_id": args.task_id,
        "precommit_sha256": file_sha256(precommit_path),
        "template_library_sha256": file_sha256(library_path),
        "accepted_examples": len(examples),
        "generation": generation,
        "validation_outputs_opened": True,
        "replacement_used": False,
        "human_survivor_selection_used": False,
    }
    if len(examples) != int(precommit["validation_examples_per_task"]):
        report.update({
            "status": "generator_invalid",
            "candidate_instantiations_tested": 0,
            "exact_instantiation_count": 0,
            "exact_instantiations": [],
        })
        write(args.output, report)
        print(json.dumps({
            "task_id": args.task_id,
            "status": report["status"],
            "accepted_examples": len(examples),
        }, sort_keys=True))
        return

    demonstration_payload = [
        {"input": source, "output": target} for source, target in examples
    ]
    report["demonstration_sha256"] = sha256_json(demonstration_payload)
    exact: list[dict[str, Any]] = []
    tested = 0
    for entry in library["templates"]:
        template_path = HERE / "templates" / f"template-{entry['template_sha256']}.json"
        if file_sha256(template_path) != entry["template_file_sha256"]:
            raise RuntimeError("frozen template file changed")
        template = load(template_path)
        for operator_arguments, colour_arguments, program in candidate_instantiations(template):
            tested += 1
            try:
                matches = all(eval_ast(program, source) == target for source, target in examples)
            except Exception:
                matches = False
            if not matches:
                continue
            exact.append({
                "template_sha256": entry["template_sha256"],
                "operator_arguments": operator_arguments,
                "colour_arguments": colour_arguments,
                "concrete_program_sha256": sha256_json(program),
                "concrete_program": program,
            })
    exact.sort(key=canonical)
    status = "unique_exact" if len(exact) == 1 else "ambiguous" if exact else "no_program"
    report.update({
        "status": status,
        "candidate_instantiations_tested": tested,
        "exact_instantiation_count": len(exact),
        "exact_instantiations": exact,
    })
    write(args.output, report)
    print(json.dumps({
        "task_id": args.task_id,
        "status": status,
        "accepted_examples": len(examples),
        "candidates_tested": tested,
        "exact_instantiations": len(exact),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
