from __future__ import annotations

from typing import Callable

import lexigen_chacha_native

Problem = dict[str, bytes]
Solution = dict[str, bytes]


def _native(problem: Problem, threads: int) -> Solution:
    ciphertext, tag = lexigen_chacha_native.encrypt(
        problem["key"],
        problem["nonce"],
        problem["plaintext"],
        problem["associated_data"],
        threads,
    )
    return {"ciphertext": ciphertext, "tag": tag}


def native_parallel2(problem: Problem) -> Solution:
    return _native(problem, 2)


def native_parallel4(problem: Problem) -> Solution:
    return _native(problem, 4)


def native_parallel8(problem: Problem) -> Solution:
    return _native(problem, 8)


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "native_parallel2": native_parallel2,
    "native_parallel4": native_parallel4,
    "native_parallel8": native_parallel8,
}
