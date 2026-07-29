from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from induce_language_v15 import load_programs
from ir_runtime_v15 import as_grid, execute
from macro_miner_v15 import instantiate, mine_macros

SOURCE_TASKS = {"c920a713", "eb5a1d5d"}


def examples_from_validation(module):
    package = module.validate()
    items = list(package.get("train", [])) + list(package.get("test", []))
    return [(as_grid(item["input"]), as_grid(item["output"])) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v14-evidence", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "V15_HELDOUT_MACRO_SCAN.json")
    args = parser.parse_args()

    programs, _, _ = load_programs(args.v14_evidence, args.package_root)
    macros = mine_macros(programs, limit=8)
    rectangle = next(
        macro for macro in macros
        if isinstance(macro.template, dict) and macro.template.get("op") == "render_concentric"
    )
    if str(args.arcgen_root) not in sys.path:
        sys.path.insert(0, str(args.arcgen_root))

    checked = imported = 0
    matches = []
    failures = []
    task_files = sorted((args.arcgen_root / "tasks").glob("task_*.py"))
    for path in task_files:
        task = path.stem.removeprefix("task_")
        if task in SOURCE_TASKS:
            continue
        checked += 1
        try:
            module = importlib.import_module(f"tasks.task_{task}")
            examples = examples_from_validation(module)
        except Exception as exc:
            failures.append({"task": task, "type": type(exc).__name__, "message": str(exc)[:200]})
            continue
        imported += 1
        if not examples:
            continue
        for mode in ("colours", "components"):
            ast = instantiate(rectangle.template, {"v0_str": mode})
            try:
                exact = sum(execute(ast, source) == target for source, target in examples)
            except Exception:
                exact = 0
            if exact == len(examples):
                matches.append({
                    "task": task,
                    "mode": mode,
                    "examples": len(examples),
                })
        if checked % 100 == 0:
            print("PROGRESS", checked, imported, len(matches), len(failures), flush=True)

    report = {
        "schema": "lexigen-v15-heldout-macro-scan-v1",
        "source_tasks_excluded": sorted(SOURCE_TASKS),
        "task_files": len(task_files),
        "checked_families": checked,
        "imported_families": imported,
        "import_or_validation_errors": len(failures),
        "matches": matches,
        "match_count": len(matches),
        "errors": failures,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps({key: report[key] for key in (
        "checked_families",
        "imported_families",
        "import_or_validation_errors",
        "match_count",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
