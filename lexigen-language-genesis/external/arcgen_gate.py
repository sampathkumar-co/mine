from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any

from arc_language import (
    as_grid,
    canonical_json,
    execute_program,
    language_artifact,
    synthesize,
    to_json_grid,
)

PROTOCOL = "arcgen-gate-v1"
DEMONSTRATIONS = 6
HIDDEN_TESTS = 20
TASK_PATTERN = re.compile(r"^from tasks import task_([0-9a-f]{8})$", re.MULTILINE)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def eligible_task_ids(arcgen_root: Path) -> list[str]:
    source = (arcgen_root / "task_list.py").read_text(encoding="utf-8")
    task_ids = sorted(set(TASK_PATTERN.findall(source)))
    if len(task_ids) < 400:
        raise RuntimeError(f"expected at least 400 ARC-GEN task IDs, found {len(task_ids)}")
    return task_ids


def select_task(lexigen_commit: str, arcgen_commit: str, task_ids: list[str]) -> tuple[str, int, str]:
    commitment = f"{PROTOCOL}|{lexigen_commit}|{arcgen_commit}".encode("utf-8")
    digest = sha256_bytes(commitment)
    index = int(digest, 16) % len(task_ids)
    return task_ids[index], index, digest


def command_select(args: argparse.Namespace) -> None:
    lexigen_root = args.lexigen_root.resolve()
    arcgen_root = args.arcgen_root.resolve()
    lexigen_commit = git_commit(lexigen_root)
    arcgen_commit = git_commit(arcgen_root)
    task_ids = eligible_task_ids(arcgen_root)
    task_id, index, digest = select_task(lexigen_commit, arcgen_commit, task_ids)
    record = {
        "protocol": PROTOCOL,
        "lexigen_commit": lexigen_commit,
        "arcgen_commit": arcgen_commit,
        "eligible_task_count": len(task_ids),
        "selection_digest": digest,
        "selection_index": index,
        "selected_task_id": task_id,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


def generate_pairs(arcgen_root: Path, task_id: str, count: int) -> list[dict[str, Any]]:
    import sys

    sys.path.insert(0, str(arcgen_root))
    try:
        import task_list  # type: ignore

        generator, _ = task_list.task_list()[task_id]
        pairs = []
        for example_id in range(count):
            random.seed(2025 + example_id)
            pair = generator()
            if set(pair) != {"input", "output"}:
                raise RuntimeError("ARC-GEN pair has unexpected fields")
            as_grid(pair["input"])
            as_grid(pair["output"])
            pairs.append(pair)
        return pairs
    finally:
        sys.path.pop(0)


def command_seal(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection["protocol"] != PROTOCOL:
        raise RuntimeError("protocol mismatch")
    arcgen_root = args.arcgen_root.resolve()
    if git_commit(arcgen_root) != selection["arcgen_commit"]:
        raise RuntimeError("ARC-GEN commit differs from selection commitment")
    pairs = generate_pairs(arcgen_root, selection["selected_task_id"], DEMONSTRATIONS + HIDDEN_TESTS)
    training = pairs[:DEMONSTRATIONS]
    hidden = pairs[DEMONSTRATIONS:]
    redacted = {
        "protocol": PROTOCOL,
        "selected_task_id": selection["selected_task_id"],
        "train": training,
        "test": [{"input": pair["input"]} for pair in hidden],
    }
    sealed = {
        "protocol": PROTOCOL,
        "selected_task_id": selection["selected_task_id"],
        "outputs": [pair["output"] for pair in hidden],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    redacted_path = args.output_dir / "redacted-task.json"
    sealed_path = args.output_dir / "sealed-outputs.json"
    redacted_path.write_text(json.dumps(redacted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sealed_path.write_text(json.dumps(sealed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        **selection,
        "demonstration_count": DEMONSTRATIONS,
        "hidden_test_count": HIDDEN_TESTS,
        "redacted_task_sha256": sha256_file(redacted_path),
        "sealed_outputs_sha256": sha256_file(sealed_path),
    }
    manifest_path = args.output_dir / "seal-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_task_id": selection["selected_task_id"],
                "demonstration_count": DEMONSTRATIONS,
                "hidden_test_count": HIDDEN_TESTS,
                "redacted_task_sha256": manifest["redacted_task_sha256"],
                "sealed_outputs_sha256": manifest["sealed_outputs_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_solve(args: argparse.Namespace) -> None:
    package = json.loads(args.redacted.read_text(encoding="utf-8"))
    examples = [(as_grid(pair["input"]), as_grid(pair["output"])) for pair in package["train"]]
    result = synthesize(examples, max_depth=3, candidate_budget=75_000)
    report: dict[str, Any] = {
        "protocol": PROTOCOL,
        "selected_task_id": package["selected_task_id"],
        "redacted_task_sha256": sha256_file(args.redacted),
        "candidate_budget": 75_000,
        "max_depth": 3,
        "inventory_size": result.inventory_size,
        "candidates_tested": result.candidates_tested,
        "signatures_seen": result.signatures_seen,
        "single_primitive_baseline_found": result.baseline_program is not None,
        "language_program_found": result.program is not None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if result.program is not None:
        artifact = language_artifact(result.program, examples)
        predictions = [
            to_json_grid(execute_program(result.program, as_grid(pair["input"])))
            for pair in package["test"]
        ]
        artifact_path = args.output_dir / "candidate-language-artifact.json"
        predictions_path = args.output_dir / "candidate-predictions.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        predictions_payload = {
            "protocol": PROTOCOL,
            "selected_task_id": package["selected_task_id"],
            "artifact_sha256": sha256_file(artifact_path),
            "predictions": predictions,
        }
        predictions_path.write_text(
            json.dumps(predictions_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report.update(
            {
                "artifact_name": artifact["name"],
                "artifact_sha256": sha256_file(artifact_path),
                "predictions_sha256": sha256_file(predictions_path),
                "program_depth": len(result.program),
                "operational_semantics": artifact["operational_semantics"],
            }
        )
    report_path = args.output_dir / "candidate-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


def command_score(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256_file(args.sealed) != manifest["sealed_outputs_sha256"]:
        raise RuntimeError("sealed output hash mismatch")
    sealed = json.loads(args.sealed.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if predictions["selected_task_id"] != sealed["selected_task_id"]:
        raise RuntimeError("task identity mismatch")
    expected = sealed["outputs"]
    actual = predictions["predictions"]
    if len(expected) != len(actual):
        raise RuntimeError("prediction count mismatch")
    correctness = [prediction == target for prediction, target in zip(actual, expected)]
    result = {
        "protocol": PROTOCOL,
        "selected_task_id": sealed["selected_task_id"],
        "sealed_outputs_sha256": manifest["sealed_outputs_sha256"],
        "predictions_sha256": sha256_file(args.predictions),
        "exact_pairs": sum(correctness),
        "pair_count": len(correctness),
        "pair_accuracy": sum(correctness) / len(correctness),
        "task_solved": all(correctness),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select")
    select.add_argument("--lexigen-root", type=Path, required=True)
    select.add_argument("--arcgen-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.set_defaults(function=command_select)

    seal = commands.add_parser("seal")
    seal.add_argument("--selection", type=Path, required=True)
    seal.add_argument("--arcgen-root", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    seal.set_defaults(function=command_seal)

    solve = commands.add_parser("solve")
    solve.add_argument("--redacted", type=Path, required=True)
    solve.add_argument("--output-dir", type=Path, required=True)
    solve.set_defaults(function=command_solve)

    score = commands.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--sealed", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(function=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
