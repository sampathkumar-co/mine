from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcgen-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.arcgen_root))
    random.seed(args.seed)
    module = importlib.import_module(f"tasks.task_{args.task_id}")
    with contextlib.redirect_stdout(io.StringIO()):
        pair = module.generate()
    result = {"input": pair["input"], "output": pair["output"]}
    sys.stdout.write(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
