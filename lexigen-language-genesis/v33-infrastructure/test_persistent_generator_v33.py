from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V25_GENERATOR = HERE.parent / "v25" / "generate_case_v25.py"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from persistent_generator_client_v33 import PersistentGeneratorClient


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def original_pair(arcgen_root: Path, task_id: str, seed: int) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(V25_GENERATOR),
            "--arcgen-root",
            str(arcgen_root),
            "--task-id",
            task_id,
            "--seed",
            str(seed),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def make_random_task(root: Path, task_id: str) -> None:
    write(root / "tasks" / "__init__.py", "")
    write(
        root / "tasks" / f"task_{task_id}.py",
        "import random\n"
        "IMPORT_VALUE = random.randint(0, 10**9)\n"
        "def generate():\n"
        "    return {\n"
        "        'input': [[IMPORT_VALUE]],\n"
        "        'output': [[random.randint(0, 10**9)]],\n"
        "    }\n",
    )


def test_seed_before_reimport_matches_original_process() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        task_id = "abcd1234"
        make_random_task(root, task_id)
        with PersistentGeneratorClient(
            root,
            task_id,
            maximum_same_seed_retries=1,
        ) as client:
            for seed in (0, 1, 17, 2**32 - 1):
                expected = original_pair(root, task_id, seed)
                actual, status = client.generate_case(root, task_id, seed, 5)
                assert status == {"status": "ok", "seed": seed}
                assert actual == expected
            assert client.restart_count == 0
            assert client.same_seed_retry_count == 0
            assert client.protocol_error_count == 0


def make_flaky_worker(path: Path) -> None:
    write(
        path,
        "import argparse, json, pathlib, sys, time\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--arcgen-root')\n"
        "p.add_argument('--task-id')\n"
        "p.add_argument('--marker', required=True)\n"
        "a=p.parse_args()\n"
        "m=pathlib.Path(a.marker)\n"
        "for line in sys.stdin:\n"
        "    r=json.loads(line)\n"
        "    if not m.exists():\n"
        "        m.write_text('timed-out-once')\n"
        "        time.sleep(2)\n"
        "    pair={'input': [[r['seed']]], 'output': [[r['seed']]]}\n"
        "    print(json.dumps({'request_id':r['request_id'],'status':'ok','pair':pair}), flush=True)\n",
    )


def test_timeout_restarts_and_retries_identical_seed() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        worker = root / "flaky_worker.py"
        marker = root / "marker.txt"
        make_flaky_worker(worker)
        with PersistentGeneratorClient(
            root,
            "deadbeef",
            worker_path=worker,
            maximum_same_seed_retries=1,
            extra_worker_arguments=("--marker", str(marker)),
        ) as client:
            pair, status = client.generate_case(root, "deadbeef", 123456, 0.2)
            assert pair == {"input": [[123456]], "output": [[123456]]}
            assert status == {"status": "ok", "seed": 123456}
            assert client.restart_count == 1
            assert client.same_seed_retry_count == 1
            assert client.protocol_error_count == 0


def main() -> None:
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"SUMMARY {len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
