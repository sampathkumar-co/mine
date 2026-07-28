from __future__ import annotations

import mmap
import os
import struct
from typing import Callable, TypedDict

from Crypto.Cipher import ChaCha20
from Crypto.Hash import Poly1305


class Problem(TypedDict):
    key: bytes
    nonce: bytes
    plaintext: bytes
    associated_data: bytes


class Solution(TypedDict):
    ciphertext: bytes
    tag: bytes


def _poly1305_tag(key: bytes, nonce: bytes, aad: bytes, ciphertext: bytes) -> bytes:
    mac = Poly1305.new(key=key, cipher=ChaCha20, nonce=nonce)
    mac.update(aad)
    aad_padding = (-len(aad)) % 16
    if aad_padding:
        mac.update(b"\x00" * aad_padding)
    mac.update(ciphertext)
    ciphertext_padding = (-len(ciphertext)) % 16
    if ciphertext_padding:
        mac.update(b"\x00" * ciphertext_padding)
    mac.update(struct.pack("<QQ", len(aad), len(ciphertext)))
    return mac.digest()


def _chunks(length: int, workers: int) -> list[tuple[int, int]]:
    blocks = (length + 63) // 64
    active = max(1, min(workers, blocks or 1))
    base, remainder = divmod(blocks, active)
    result: list[tuple[int, int]] = []
    start_block = 0
    for index in range(active):
        count = base + (1 if index < remainder else 0)
        end_block = start_block + count
        start = start_block * 64
        end = min(length, end_block * 64)
        result.append((start, end))
        start_block = end_block
    return result


def _solve(problem: Problem, workers: int) -> Solution:
    if os.name != "posix" or not hasattr(os, "fork"):
        raise RuntimeError("revision 2 requires Linux fork semantics")

    key = problem["key"]
    nonce = problem["nonce"]
    plaintext = problem["plaintext"]
    aad = problem["associated_data"]
    length = len(plaintext)
    output = mmap.mmap(-1, max(1, length))
    pids: list[int] = []

    try:
        for start, end in _chunks(length, workers):
            pid = os.fork()
            if pid == 0:
                try:
                    cipher = ChaCha20.new(key=key, nonce=nonce)
                    cipher.seek(64 + start)
                    encrypted = cipher.encrypt(memoryview(plaintext)[start:end])
                    if encrypted:
                        output.seek(start)
                        output.write(encrypted)
                    os._exit(0)
                except BaseException:
                    os._exit(1)
            pids.append(pid)

        failed = False
        for pid in pids:
            _, status = os.waitpid(pid, 0)
            if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
                failed = True
        if failed:
            raise RuntimeError("parallel ChaCha worker failed")

        ciphertext = output[:length] if length else b""
    finally:
        for pid in pids:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        output.close()

    tag = _poly1305_tag(key, nonce, aad, ciphertext)
    return {"ciphertext": ciphertext, "tag": tag}


def fork2(problem: Problem) -> Solution:
    return _solve(problem, 2)


def fork4(problem: Problem) -> Solution:
    return _solve(problem, 4)


def fork8(problem: Problem) -> Solution:
    return _solve(problem, 8)


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "fork2": fork2,
    "fork4": fork4,
    "fork8": fork8,
}
