from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_WORKER = HERE / "persistent_generator_worker_v33.py"


class PersistentGeneratorClient:
    def __init__(
        self,
        arcgen_root: Path,
        task_id: str,
        *,
        worker_path: Path = DEFAULT_WORKER,
        maximum_same_seed_retries: int = 1,
        extra_worker_arguments: tuple[str, ...] = (),
    ) -> None:
        self.arcgen_root = Path(arcgen_root)
        self.task_id = str(task_id)
        self.worker_path = Path(worker_path)
        self.maximum_same_seed_retries = int(maximum_same_seed_retries)
        self.extra_worker_arguments = tuple(extra_worker_arguments)
        self.process: subprocess.Popen[str] | None = None
        self.responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=40)
        self.request_id = 0
        self.restart_count = 0
        self.same_seed_retry_count = 0
        self.protocol_error_count = 0

    def _command(self) -> list[str]:
        return [
            sys.executable,
            str(self.worker_path),
            "--arcgen-root",
            str(self.arcgen_root),
            "--task-id",
            self.task_id,
            *self.extra_worker_arguments,
        ]

    def _read_stdout(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            try:
                response = json.loads(line)
            except Exception as error:
                self.responses.put({
                    "status": "protocol_error",
                    "message": str(error),
                    "raw": line[-500:],
                })
                continue
            self.responses.put(response)

    def _read_stderr(self, process: subprocess.Popen[str]) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            self.stderr_tail.append(line.rstrip())

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.responses = queue.Queue()
        self.process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        threading.Thread(
            target=self._read_stdout,
            args=(self.process,),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._read_stderr,
            args=(self.process,),
            daemon=True,
        ).start()

    def _stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    def close(self) -> None:
        self._stop()

    def _request_once(
        self,
        seed: int,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any] | None, str]:
        self._start()
        assert self.process is not None
        assert self.process.stdin is not None
        self.request_id += 1
        request_id = self.request_id
        try:
            self.process.stdin.write(
                json.dumps(
                    {"request_id": request_id, "seed": int(seed)},
                    separators=(",", ":"),
                )
                + "\n"
            )
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return None, "worker_exit"
        try:
            response = self.responses.get(timeout=timeout_seconds)
        except queue.Empty:
            return None, "timeout"
        if response.get("status") == "protocol_error":
            self.protocol_error_count += 1
            return response, "protocol_error"
        if int(response.get("request_id", -1)) != request_id:
            self.protocol_error_count += 1
            return response, "request_id_mismatch"
        return response, str(response.get("status"))

    def generate_case(
        self,
        arcgen_root: Path,
        task_id: str,
        seed: int,
        timeout_seconds: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        if Path(arcgen_root).resolve() != self.arcgen_root.resolve():
            raise RuntimeError("persistent client ARC-GEN root mismatch")
        if str(task_id) != self.task_id:
            raise RuntimeError("persistent client task identity mismatch")

        for attempt in range(self.maximum_same_seed_retries + 1):
            response, status = self._request_once(seed, timeout_seconds)
            if status == "ok" and response is not None:
                return response["pair"], {"status": "ok", "seed": int(seed)}
            if status == "error" and response is not None:
                message = (
                    f"{response.get('error_type')}: {response.get('message')}\n"
                    f"{response.get('traceback', '')}"
                )
                return None, {
                    "status": "subprocess_error",
                    "seed": int(seed),
                    "stderr": message[-500:],
                }
            self._stop()
            self.restart_count += 1
            if attempt < self.maximum_same_seed_retries:
                self.same_seed_retry_count += 1
                continue
            if status == "timeout":
                return None, {"status": "timeout", "seed": int(seed)}
            return None, {
                "status": "subprocess_error",
                "seed": int(seed),
                "stderr": "\n".join(self.stderr_tail)[-500:],
            }
        raise AssertionError("unreachable retry loop")

    def __enter__(self) -> "PersistentGeneratorClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
