from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PRECOMMIT = HERE / "V30_PRECOMMIT.json"
GRAMMAR = HERE / "V30_GRAMMAR.json"
SOURCES = HERE.parent / "v29" / "V29_SOURCE_STRUCTURES.json"
MANIFEST = HERE / "V30_GRAMMAR_MANIFEST.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def main() -> None:
    precommit = load(PRECOMMIT)
    grammar = load(GRAMMAR)
    sources = load(SOURCES)
    candidate_asts = {canonical(item["ast"]) for item in grammar["candidates"]}
    source_asts = [canonical(item["structure"]) for item in sources["structures"]]

    assert grammar["structural_candidate_count"] == 23916
    assert grammar["structural_cap_reached"] is False
    assert grammar["candidate_sha256"] == "50a604639fef3158bbd2821afdf2f6f2d5562ea5d6eecb7fd5f0e8563661cd25"
    assert all(ast in candidate_asts for ast in source_asts)
    assert len(precommit["validation_task_ids"]) == 20
    assert precommit["validation_generators_imported"] == 0
    assert precommit["validation_outputs_opened"] is False

    source_hashes = sorted(
        hashlib.sha256(ast.encode("utf-8")).hexdigest()
        for ast in source_asts
    )
    manifest = {
        "schema": "lexigen-v30-frozen-grammar-manifest-v1",
        "precommit_sha256": sha256_file(PRECOMMIT),
        "source_structures_sha256": sha256_file(SOURCES),
        "grammar_file_sha256": sha256_file(GRAMMAR),
        "candidate_sha256": grammar["candidate_sha256"],
        "structural_candidate_count": grammar["structural_candidate_count"],
        "support_expression_counts": grammar["support_expression_counts"],
        "source_program_ast_sha256": source_hashes,
        "all_source_programs_present": True,
        "validation_task_count": len(precommit["validation_task_ids"]),
        "validation_generators_imported": 0,
        "validation_outputs_opened": False,
        "world_level_breakthrough": False,
    }
    write_json(MANIFEST, manifest)
    print(json.dumps({
        "manifest_sha256": sha256_file(MANIFEST),
        "grammar_file_sha256": manifest["grammar_file_sha256"],
        "candidate_sha256": manifest["candidate_sha256"],
        "structural_candidate_count": manifest["structural_candidate_count"],
        "all_source_programs_present": True,
        "validation_outputs_opened": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
