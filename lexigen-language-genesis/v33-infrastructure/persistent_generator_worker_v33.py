from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any


def write_response(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def clear_task_module(module_name: str) -> None:
    for name in tuple(sys.modules):
        if name == module_name or name.startswith(module_name + "."):
            del sys.modules[name]


def generate(module_name: str, seed: int) -> dict[str, Any]:
    random.seed(seed)
    clear_task_module(module_name)
    importlib.invalidate_caches()
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        module = importlib.import_module(module_name)
        pair = module.generate()
    return {
        "input": pair["input"],
        "output": pair["output"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.arcgen_root))
    module_name = f"tasks.task_{args.task_id}"
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request_id: int | None = None
        try:
            request = json.loads(line)
            request_id = int(request["request_id"])
            seed = int(request["seed"])
            pair = generate(module_name, seed)
            write_response({
                "request_id": request_id,
                "status": "ok",
                "pair": pair,
            })
        except BaseException as error:
            write_response({
                "request_id": request_id,
                "status": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=8),
            })


if __name__ == "__main__":
    main()
