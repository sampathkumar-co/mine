from __future__ import annotations

import ctypes
import ctypes.util
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypedDict

from cryptography.hazmat.primitives.poly1305 import Poly1305


class Problem(TypedDict):
    key: bytes
    nonce: bytes
    plaintext: bytes
    associated_data: bytes


class Solution(TypedDict):
    ciphertext: bytes
    tag: bytes


_libcrypto_name = ctypes.util.find_library("crypto")
if not _libcrypto_name:
    raise RuntimeError("system libcrypto was not found")
_libcrypto = ctypes.CDLL(_libcrypto_name)

_libcrypto.EVP_chacha20.restype = ctypes.c_void_p
_libcrypto.EVP_CIPHER_CTX_new.restype = ctypes.c_void_p
_libcrypto.EVP_CIPHER_CTX_free.argtypes = [ctypes.c_void_p]
_libcrypto.EVP_EncryptInit_ex.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_libcrypto.EVP_EncryptInit_ex.restype = ctypes.c_int
_libcrypto.EVP_EncryptUpdate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_void_p,
    ctypes.c_int,
]
_libcrypto.EVP_EncryptUpdate.restype = ctypes.c_int
_libcrypto.EVP_EncryptFinal_ex.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
]
_libcrypto.EVP_EncryptFinal_ex.restype = ctypes.c_int

_cipher = _libcrypto.EVP_chacha20()
if not _cipher:
    raise RuntimeError("system OpenSSL does not provide EVP_chacha20")

_python = ctypes.pythonapi
_python.PyBytes_FromStringAndSize.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_python.PyBytes_FromStringAndSize.restype = ctypes.py_object
_python.PyBytes_AsString.argtypes = [ctypes.py_object]
_python.PyBytes_AsString.restype = ctypes.c_void_p


def _bytes_pointer(value: bytes) -> int:
    pointer = ctypes.cast(ctypes.c_char_p(value), ctypes.c_void_p).value
    if pointer is None:
        raise RuntimeError("could not obtain bytes pointer")
    return int(pointer)


def _new_bytes(size: int) -> tuple[bytes, int]:
    value = _python.PyBytes_FromStringAndSize(None, size)
    pointer = _python.PyBytes_AsString(value)
    if pointer is None:
        raise RuntimeError("could not allocate output bytes")
    return value, int(pointer)


def _xor_range(
    key: bytes,
    nonce: bytes,
    counter: int,
    source_address: int,
    destination_address: int,
    length: int,
) -> None:
    if length > 2**31 - 1:
        raise OverflowError("one ChaCha range exceeds the OpenSSL integer length")
    context = _libcrypto.EVP_CIPHER_CTX_new()
    if not context:
        raise RuntimeError("EVP_CIPHER_CTX_new failed")
    try:
        iv = struct.pack("<I", counter) + nonce
        if _libcrypto.EVP_EncryptInit_ex(
            context,
            _cipher,
            None,
            ctypes.c_void_p(_bytes_pointer(key)),
            ctypes.c_void_p(_bytes_pointer(iv)),
        ) != 1:
            raise RuntimeError("EVP_EncryptInit_ex failed")
        produced = ctypes.c_int()
        if _libcrypto.EVP_EncryptUpdate(
            context,
            ctypes.c_void_p(destination_address),
            ctypes.byref(produced),
            ctypes.c_void_p(source_address),
            length,
        ) != 1:
            raise RuntimeError("EVP_EncryptUpdate failed")
        if produced.value != length:
            raise RuntimeError("OpenSSL produced an unexpected ciphertext length")
        final_length = ctypes.c_int()
        if _libcrypto.EVP_EncryptFinal_ex(
            context,
            ctypes.c_void_p(destination_address + length),
            ctypes.byref(final_length),
        ) != 1:
            raise RuntimeError("EVP_EncryptFinal_ex failed")
        if final_length.value != 0:
            raise RuntimeError("raw ChaCha20 unexpectedly emitted final bytes")
    finally:
        _libcrypto.EVP_CIPHER_CTX_free(context)


def _poly1305_tag(
    one_time_key: bytes,
    associated_data: bytes,
    ciphertext: bytes,
) -> bytes:
    authenticator = Poly1305(one_time_key)
    authenticator.update(associated_data)
    associated_padding = (-len(associated_data)) % 16
    if associated_padding:
        authenticator.update(b"\x00" * associated_padding)
    authenticator.update(ciphertext)
    ciphertext_padding = (-len(ciphertext)) % 16
    if ciphertext_padding:
        authenticator.update(b"\x00" * ciphertext_padding)
    authenticator.update(struct.pack("<QQ", len(associated_data), len(ciphertext)))
    return authenticator.finalize()


def _chunks(length: int, workers: int) -> list[tuple[int, int, int]]:
    blocks = (length + 63) // 64
    active_workers = max(1, min(workers, blocks or 1))
    base, remainder = divmod(blocks, active_workers)
    chunks: list[tuple[int, int, int]] = []
    start_block = 0
    for worker_index in range(active_workers):
        block_count = base + (1 if worker_index < remainder else 0)
        end_block = start_block + block_count
        start = start_block * 64
        end = min(length, end_block * 64)
        chunks.append((1 + start_block, start, end))
        start_block = end_block
    return chunks


def _solve(problem: Problem, workers: int) -> Solution:
    key = problem["key"]
    nonce = problem["nonce"]
    plaintext = problem["plaintext"]
    associated_data = problem["associated_data"]
    if len(key) != 32 or len(nonce) != 12:
        raise ValueError("invalid ChaCha20-Poly1305 key or nonce length")

    ciphertext, ciphertext_address = _new_bytes(len(plaintext))
    one_time_stream, one_time_address = _new_bytes(64)
    zero_block = bytes(64)
    _xor_range(
        key,
        nonce,
        0,
        _bytes_pointer(zero_block),
        one_time_address,
        64,
    )
    one_time_key = one_time_stream[:32]

    if plaintext:
        plaintext_address = _bytes_pointer(plaintext)
        chunks = _chunks(len(plaintext), workers)
        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [
                executor.submit(
                    _xor_range,
                    key,
                    nonce,
                    counter,
                    plaintext_address + start,
                    ciphertext_address + start,
                    end - start,
                )
                for counter, start, end in chunks
            ]
            for future in futures:
                future.result()

    tag = _poly1305_tag(one_time_key, associated_data, ciphertext)
    return {"ciphertext": ciphertext, "tag": tag}


def direct2(problem: Problem) -> Solution:
    return _solve(problem, 2)


def direct4(problem: Problem) -> Solution:
    return _solve(problem, 4)


def direct8(problem: Problem) -> Solution:
    return _solve(problem, 8)


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "direct2": direct2,
    "direct4": direct4,
    "direct8": direct8,
}
