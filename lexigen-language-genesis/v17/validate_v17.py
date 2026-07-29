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
for folder in (HERE, HERE.parent / "v15"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from constructive_dsl_v17 import (
    FORBIDDEN_OPS,
    as_grid,
    canonical,
    execute,
    sha256_json,
    synthesize,
    walk_ops,
)
from portable_constructive_dsl_v17 import execute_portable

TARGET_GATES = (1, 2, 3)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_families(v14_evidence: Path) -> dict[int, dict[str, Any]]:
    evidence = json.loads(v14_evidence.read_text(encoding="utf-8"))
    return {
        int(item["gate"]): item
        for item in evidence["families_report"]
        if int(item["gate"]) in TARGET_GATES
    }


def load_demonstrations(package_root: Path, gate: int):
    package_path = (
        package_root
        / f"v13-campaign-{gate:02d}"
        / "redacted-task.json"
    )
    package = json.loads(package_path.read_text(encoding="utf-8"))
    return [
        (as_grid(item["input"]), as_grid(item["output"]))
        for item in package["train"]
    ]


def generate(task: str, arcgen_root: Path, seed: int):
    if str(arcgen_root) not in sys.path:
        sys.path.insert(0, str(arcgen_root))
    random.seed(seed)
    pair = importlib.import_module(f"tasks.task_{task}").generate()
    return as_grid(pair["input"]), as_grid(pair["output"])


def freeze_contract(program, demonstrations, grammar_hash: str, portable_hash: str):
    demo_digest = sha256_json([
        {"input": source, "output": target}
        for source, target in demonstrations
    ])
    payload = {
        "schema": "lexigen-v17-program-contract-v1",
        "program_sha256": sha256_json(program),
        "grammar_sha256": grammar_hash,
        "portable_runtime_sha256": portable_hash,
        "demonstration_sha256": demo_digest,
        "verification": "independent-dual-runtime-exact-output",
        "task_id_available_to_synthesizer": False,
    }
    return {
        **payload,
        "contract_sha256": sha256_json(payload),
    }


def validate_fresh(program, task: str, gate: int, arcgen_root: Path, count: int):
    accepted = attempts = rejections = 0
    primary_exact = portable_exact = agreement = 0
    while accepted < count:
        seed = 5_100_000 + gate * 100_000 + attempts
        attempts += 1
        try:
            source, target = generate(task, arcgen_root, seed)
        except (ValueError, IndexError, TypeError, RuntimeError):
            rejections += 1
            if attempts > count * 5 + 1000:
                raise RuntimeError(f"too many generator rejections for gate {gate}")
            continue
        primary = execute(program, source)
        portable = execute_portable(program, source)
        primary_exact += primary == target
        portable_exact += portable == target
        agreement += primary == portable
        accepted += 1
    return {
        "accepted_cases": accepted,
        "generator_attempts": attempts,
        "generator_rejections": rejections,
        "primary_exact": primary_exact,
        "portable_exact": portable_exact,
        "runtime_agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--output", type=Path, default=HERE / "V17_REPORT.json")
    args = parser.parse_args()

    families = load_families(args.v14_evidence)
    if sorted(families) != list(TARGET_GATES):
        raise RuntimeError("frozen target-family set is incomplete")
    programs_dir = HERE / "programs"
    contracts_dir = HERE / "contracts"
    programs_dir.mkdir(exist_ok=True)
    contracts_dir.mkdir(exist_ok=True)
    grammar_hash = file_sha256(HERE / "constructive_dsl_v17.py")
    portable_hash = file_sha256(HERE / "portable_constructive_dsl_v17.py")
    reports = []
    for gate in TARGET_GATES:
        item = families[gate]
        task = str(item["task"])
        demonstrations = load_demonstrations(args.package_root, gate)
        program, search = synthesize(demonstrations)
        forbidden = sorted(set(walk_ops(program)) & FORBIDDEN_OPS)
        if forbidden:
            raise RuntimeError(f"forbidden opcodes in gate {gate}: {forbidden}")
        demo_primary = sum(
            execute(program, source) == target
            for source, target in demonstrations
        )
        demo_portable = sum(
            execute_portable(program, source) == target
            for source, target in demonstrations
        )
        if demo_primary != len(demonstrations) or demo_portable != len(demonstrations):
            raise RuntimeError(f"demonstration replay failed for gate {gate}")
        contract = freeze_contract(
            program,
            demonstrations,
            grammar_hash,
            portable_hash,
        )
        fresh = validate_fresh(program, task, gate, args.arcgen_root, args.cases)
        if not (
            fresh["primary_exact"]
            == fresh["portable_exact"]
            == fresh["runtime_agreement"]
            == fresh["accepted_cases"]
        ):
            raise RuntimeError(f"fresh dual-runtime gate failed for gate {gate}")
        program_path = programs_dir / f"v17-program-{gate:02d}.json"
        contract_path = contracts_dir / f"v17-contract-{gate:02d}.json"
        program_path.write_bytes(
            (json.dumps(program, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        contract_path.write_bytes(
            (json.dumps(contract, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        report = {
            "gate": gate,
            "task": task,
            "demonstrations": len(demonstrations),
            "demonstration_primary_exact": demo_primary,
            "demonstration_portable_exact": demo_portable,
            "program_sha256": sha256_json(program),
            "contract_sha256": contract["contract_sha256"],
            "search": search,
            **fresh,
        }
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)

    summary = {
        "schema": "lexigen-v17-constructive-primitive-report-v1",
        "families": len(reports),
        "target_gates": list(TARGET_GATES),
        "cases_per_family": args.cases,
        "fresh_cases": sum(item["accepted_cases"] for item in reports),
        "fresh_failures": 0,
        "runtime_disagreements": 0,
        "grammar_sha256": grammar_hash,
        "portable_runtime_sha256": portable_hash,
        "named_scene_opcodes_allowed": False,
        "task_ids_available_to_synthesizer": False,
        "human_supplied_low_level_grammar": True,
        "human_supplied_search_schemas": True,
        "autonomous_semantic_substrate_invention": False,
        "world_level_breakthrough": False,
        "reports": reports,
    }
    args.output.write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print("SUMMARY", json.dumps({
        "families": summary["families"],
        "fresh_cases": summary["fresh_cases"],
        "fresh_failures": summary["fresh_failures"],
        "runtime_disagreements": summary["runtime_disagreements"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
