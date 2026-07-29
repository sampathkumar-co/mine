from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V17 = HERE.parent / "v17"
for folder in (HERE, V17):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from constructive_dsl_v17 import synthesize as synthesize_v17
from invent_production_v19 import invent_production
from portable_runtime_v19 import execute_production_portable
from primitive_runtime_v19 import (
    FORBIDDEN_OPS,
    as_grid,
    canonical,
    execute_production,
    sha256_json,
    walk_ops,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def generate(task_id: str, arcgen_root: Path, seed: int):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    random.seed(seed)
    pair = importlib.import_module(f"tasks.task_{task_id}").generate()
    return as_grid(pair["input"]), as_grid(pair["output"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--output", type=Path, default=HERE / "V19_REPORT.json")
    args = parser.parse_args()

    precommit = load_json(HERE / "V19_PRECOMMIT.json")
    gate = int(precommit["development_gate"])
    task_id = str(precommit["development_task_id"])
    package_path = (
        args.package_root
        / f"v13-campaign-{gate:02d}"
        / "redacted-task.json"
    )
    if file_sha256(package_path) != precommit["development_redacted_task_sha256"]:
        raise RuntimeError("development package hash changed")
    package = load_json(package_path)
    examples = [
        (as_grid(item["input"]), as_grid(item["output"]))
        for item in package["train"]
    ]

    v17_failure = None
    try:
        synthesize_v17(examples)
    except RuntimeError as error:
        v17_failure = str(error)
    if v17_failure is None:
        raise RuntimeError("frozen v17 unexpectedly produced a program")

    production, arguments, source_program, invention = invent_production(examples)
    forbidden = sorted(
        (set(walk_ops(source_program)) | set(walk_ops(production))) & FORBIDDEN_OPS
    )
    if forbidden:
        raise RuntimeError(f"forbidden named operators found: {forbidden}")
    if task_id in canonical(source_program) or task_id in canonical(production):
        raise RuntimeError("task identity leaked into invented semantics")

    demo_primary = sum(
        execute_production(production, arguments, source) == target
        for source, target in examples
    )
    demo_portable = sum(
        execute_production_portable(production, arguments, source) == target
        for source, target in examples
    )
    if demo_primary != len(examples) or demo_portable != len(examples):
        raise RuntimeError("demonstration replay failed")

    accepted = attempts = rejections = 0
    primary_exact = portable_exact = agreement = 0
    rejection_examples = []
    while accepted < args.cases:
        seed = 6_300_000 + attempts
        attempts += 1
        try:
            source, target = generate(task_id, args.arcgen_root, seed)
        except (ValueError, IndexError, TypeError, RuntimeError) as error:
            rejections += 1
            if len(rejection_examples) < 20:
                rejection_examples.append({
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                })
            if attempts > args.cases * 5 + 1000:
                raise RuntimeError("too many public generator rejections")
            continue
        primary = execute_production(production, arguments, source)
        portable = execute_production_portable(production, arguments, source)
        primary_exact += primary == target
        portable_exact += portable == target
        agreement += primary == portable
        accepted += 1
    if primary_exact != accepted or portable_exact != accepted or agreement != accepted:
        raise RuntimeError("fresh dual-runtime gate failed")

    write_json(HERE / "production" / "v19-production.json", production)
    write_json(HERE / "production" / "v19-arguments.json", arguments)
    write_json(HERE / "production" / "v19-source-program.json", source_program)

    report = {
        "schema": "lexigen-v19-production-invention-report-v1",
        "development_gate": gate,
        "development_task_id": task_id,
        "demonstrations": len(examples),
        "demonstration_primary_exact": demo_primary,
        "demonstration_portable_exact": demo_portable,
        "accepted_fresh_cases": accepted,
        "generator_attempts": attempts,
        "generator_rejections": rejections,
        "generator_rejection_examples": rejection_examples,
        "primary_exact": primary_exact,
        "portable_exact": portable_exact,
        "runtime_agreement": agreement,
        "frozen_v17_ablation_failed": True,
        "frozen_v17_failure": v17_failure,
        "production_sha256": sha256_json(production),
        "source_program_sha256": sha256_json(source_program),
        "arguments_sha256": sha256_json(arguments),
        "forbidden_opcode_hits": forbidden,
        "task_id_hits": [],
        "hidden_outputs_opened": False,
        "invention": invention,
        "claim_boundary": {
            "autonomous_grammar_production_candidate": True,
            "sealed_external_success": False,
            "transfer_demonstrated": False,
            "world_level_breakthrough": False,
            "remaining_human_bias": "typed primitive inventory and bounded clause enumeration",
        },
    }
    write_json(args.output, report)
    print("SUMMARY", json.dumps({
        "fresh_cases": accepted,
        "production_sha256": report["production_sha256"],
        "primary_exact": primary_exact,
        "portable_exact": portable_exact,
        "runtime_agreement": agreement,
        "v17_ablation_failed": report["frozen_v17_ablation_failed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
