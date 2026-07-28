from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from portable_runtime_v7 import as_grid as portable_grid
from portable_runtime_v7 import execute_portable
from semantic_ast_v7 import as_grid, build_certificate, canonical_json, execute_ast, synthesize_ast


def load_examples(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    train = payload.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError("training file must contain a non-empty train list")
    return [
        (as_grid(item["input"]), as_grid(item["output"]))
        for item in train
    ], payload


def run(training_path: Path, output_dir: Path) -> dict[str, Any]:
    examples, source_payload = load_examples(training_path)
    result = synthesize_ast(examples)
    if result.ast is None:
        raise AssertionError("v7 grammar could not reconstruct the demonstrations")
    if result.ambiguous:
        raise AssertionError("minimum-description semantic explanation remains ambiguous")

    portable_agreement = all(
        execute_ast(result.ast, source)
        == execute_portable(result.ast, portable_grid(source))
        == target
        for source, target in examples
    )
    certificate = build_certificate(result.ast, examples, portable_agreement)
    if not all(
        (
            certificate["exact_reconstruction"],
            certificate["portable_runtime_agreement"],
            certificate["deterministic"],
        )
    ):
        raise AssertionError("v7 verifier certificate failed")

    artifact = {
        "schema": "lexigen-v7-semantic-artifact-v1",
        "name": "relational_semantics_" + certificate["ast_sha256"][:12],
        "semantic_ast": result.ast,
        "verifier_certificate": certificate,
        "provenance": {
            "source_protocol": source_payload.get("protocol"),
            "source_task_id": source_payload.get("selected_task_id"),
            "source_redacted_sha256": source_payload.get("redacted_task_sha256"),
            "source_sealed_outputs_sha256": source_payload.get("sealed_outputs_sha256"),
            "published_after_permanent_failure": source_payload.get("status"),
            "candidate_asts_tested": result.candidates_tested,
            "exact_candidate_count": result.exact_candidate_count,
            "human_supplied_meta_grammar": True,
            "sealed_outputs_accessed": False,
        },
    }
    report = {
        "version": "v7",
        "status": "typed semantic synthesis milestone; no external breakthrough claim",
        "artifact_name": artifact["name"],
        "demonstration_count": len(examples),
        "semantic_ast": result.ast,
        "version_space_features": sorted({candidate["match"]["feature"] for candidate in result.exact_candidates}),
        "candidates_tested": result.candidates_tested,
        "exact_candidate_count": result.exact_candidate_count,
        "ambiguous": result.ambiguous,
        "portable_runtime_agreement": portable_agreement,
        "gate": {
            "relational_ast_synthesized": result.ast["match"]["feature"] in {"area", "bbox", "normalised_points"},
            "minimum_description_area_relation": result.ast["match"]["feature"] == "area",
            "uniform_frame_semantics": result.ast["scene"]["hole_boundary"] == "all",
            "source_erasure_synthesized": result.ast["render"]["erase_source"] is True,
            "exact_reconstruction": certificate["exact_reconstruction"],
            "independent_runtime": certificate["portable_runtime_agreement"],
        },
        "claim_boundary": (
            "v7 constructs the operation from lower-level relational semantics and corrects the earlier hand-written exact-shape assumption: "
            "the published demonstrations support a shorter area relation. The typed meta-grammar remains human supplied, "
            "so this is not yet autonomous open-ended language invention."
        ),
    }
    if not all(report["gate"].values()):
        raise AssertionError(f"v7 gate failed: {report['gate']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "v7-semantic-artifact.json"
    report_path = output_dir / "v7-report.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/v7"))
    args = parser.parse_args()
    run(args.training, args.output_dir)


if __name__ == "__main__":
    main()
