from __future__ import annotations

import hashlib
from typing import Callable

import lexigen_sha256_native

Problem = dict[str, object]
Solution = dict[str, bytes]


def _plaintext(problem: Problem) -> bytes:
    plaintext = problem["plaintext"]
    if not isinstance(plaintext, bytes):
        raise TypeError("plaintext must be bytes")
    return plaintext


def hashlib_builtin(problem: Problem) -> Solution:
    return {"digest": hashlib.sha256(_plaintext(problem)).digest()}


def openssl_sha256_oneshot(problem: Problem) -> Solution:
    return {"digest": lexigen_sha256_native.sha256_oneshot(_plaintext(problem))}


def openssl_evp_q_digest(problem: Problem) -> Solution:
    return {"digest": lexigen_sha256_native.evp_q_digest(_plaintext(problem))}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "hashlib_builtin": hashlib_builtin,
    "openssl_sha256_oneshot": openssl_sha256_oneshot,
    "openssl_evp_q_digest": openssl_evp_q_digest,
}
