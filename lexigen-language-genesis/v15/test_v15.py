from __future__ import annotations

import json
import os
from pathlib import Path

from compiler_v15 import compile_pipeline
from induce_language_v15 import load_programs
from ir_runtime_v15 import execute
from macro_miner_v15 import compress, expand, mine_macros, tree_size

HERE = Path(__file__).resolve().parent
EVIDENCE = Path(os.environ.get("V13_EVIDENCE_ROOT", r"C:\Users\SAMPATH\AppData\Local\Temp\lexigen-v13-campaign-engine\lexigen-language-genesis\external\evidence\v13-campaign"))
V14_EVIDENCE = HERE.parent / "v14" / "V14_EVIDENCE.json"


def test_compiled_ir_replays_all_demonstrations() -> None:
    programs, examples, metadata = load_programs(V14_EVIDENCE, EVIDENCE)
    assert len(programs) == 9
    for ast, item in zip(programs, metadata):
        assert all(execute(ast, source) == target for source, target in examples[item["gate"]])


def test_induced_macros_expand_exactly() -> None:
    programs, _, _ = load_programs(V14_EVIDENCE, EVIDENCE)
    macros = mine_macros(programs)
    assert len(macros) >= 3
    for program in programs:
        assert expand(compress(program, macros), macros) == program


def test_induced_library_contains_shared_rectangle_program() -> None:
    programs, _, _ = load_programs(V14_EVIDENCE, EVIDENCE)
    macros = mine_macros(programs)
    templates = [json.dumps(macro.template, sort_keys=True) for macro in macros]
    assert any('"op": "render_concentric"' in template for template in templates)
    assert any('"op": "rect_objects"' in template for template in templates)


def test_macro_compression_reduces_repeated_rectangle_ast() -> None:
    programs, _, metadata = load_programs(V14_EVIDENCE, EVIDENCE)
    macros = mine_macros(programs)
    by_gate = {item["gate"]: program for item, program in zip(metadata, programs)}
    for gate in (6, 8):
        compressed = compress(by_gate[gate], macros)
        assert tree_size(compressed) < tree_size(by_gate[gate])
        assert expand(compressed, macros) == by_gate[gate]

