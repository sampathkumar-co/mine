from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compiler_v15 import compile_pipeline
from ir_runtime_v15 import as_grid, execute
from macro_miner_v15 import canonical, compress, expand, mine_macros, tree_size


def load_programs(v14_evidence: Path, package_root: Path):
    evidence = json.loads(v14_evidence.read_text(encoding="utf-8"))
    programs = []
    examples_by_gate = {}
    metadata = []
    for family in evidence["families_report"]:
        gate = int(family["gate"])
        package_path = package_root / f"v13-campaign-{gate:02d}" / "redacted-task.json"
        package = json.loads(package_path.read_text(encoding="utf-8"))
        examples = [(as_grid(item["input"]), as_grid(item["output"])) for item in package["train"]]
        ast = compile_pipeline(family["pipeline"])
        programs.append(ast)
        examples_by_gate[gate] = examples
        metadata.append({"gate": gate, "task": family["task"], "pipeline": family["pipeline"]})
    return programs, examples_by_gate, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V15_INDUCTION_REPORT.json")
    args = parser.parse_args()

    programs, examples_by_gate, metadata = load_programs(args.v14_evidence, args.package_root)
    demonstration_checks = []
    for ast, item in zip(programs, metadata):
        exact = sum(execute(ast, source) == target for source, target in examples_by_gate[item["gate"]])
        demonstration_checks.append({
            "gate": item["gate"],
            "task": item["task"],
            "examples": len(examples_by_gate[item["gate"]]),
            "exact": exact,
            "ast_sha256": hashlib.sha256(canonical(ast).encode()).hexdigest(),
            "ast_size": tree_size(ast),
        })
    if any(item["exact"] != item["examples"] for item in demonstration_checks):
        raise SystemExit("compiled IR failed demonstration replay")

    macros = mine_macros(programs, limit=8)
    compressed = [compress(program, macros) for program in programs]
    expanded = [expand(program, macros) for program in compressed]
    if expanded != programs:
        raise SystemExit("macro expansion did not reconstruct original ASTs")

    expanded_checks = []
    for ast, item in zip(expanded, metadata):
        exact = sum(execute(ast, source) == target for source, target in examples_by_gate[item["gate"]])
        expanded_checks.append({"gate": item["gate"], "exact": exact})
    if any(item["exact"] != len(examples_by_gate[item["gate"]]) for item in expanded_checks):
        raise SystemExit("expanded induced language changed semantics")

    original_size = sum(tree_size(program) for program in programs)
    compressed_program_size = sum(tree_size(program) for program in compressed)
    library_size = sum(tree_size(macro.template) for macro in macros)
    report = {
        "schema": "lexigen-v15-induced-language-report-v1",
        "source_programs": len(programs),
        "demonstration_checks": demonstration_checks,
        "macros": [
            {
                "name": macro.name,
                "occurrences": macro.occurrences,
                "score": macro.score,
                "template": macro.template,
                "template_size": tree_size(macro.template),
            }
            for macro in macros
        ],
        "original_ast_size": original_size,
        "compressed_program_size": compressed_program_size,
        "induced_library_size": library_size,
        "total_induced_description_length": compressed_program_size + library_size,
        "raw_compression_ratio": (
            original_size / (compressed_program_size + library_size)
            if compressed_program_size + library_size else 0.0
        ),
        "saved_nodes_per_full_replay": original_size - compressed_program_size,
        "break_even_replays": (
            (library_size + original_size - compressed_program_size - 1)
            // (original_size - compressed_program_size)
            if original_size > compressed_program_size else None
        ),
        "dynamic_description_length_at_10000_replays": compressed_program_size * 10000 + library_size,
        "dynamic_compression_ratio_at_10000_replays": (
            original_size * 10000 / (compressed_program_size * 10000 + library_size)
            if compressed_program_size * 10000 + library_size else 0.0
        ),
        "exact_macro_expansion": True,
        "semantic_replay_exact": True,
        "human_supplied_v14_scene_atoms": True,
        "human_supplied_macro_definitions": False,
        "world_level_breakthrough": False,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
