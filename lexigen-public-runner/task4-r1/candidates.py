from __future__ import annotations

from typing import Callable

from Crypto.Cipher import ChaCha20_Poly1305
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from nacl.bindings import crypto_aead_chacha20poly1305_ietf_encrypt

Problem = dict[str, bytes]
Solution = dict[str, bytes]
TAG_SIZE = 16


def _split_combined(combined: bytes) -> Solution:
    return {
        "ciphertext": combined[:-TAG_SIZE],
        "tag": combined[-TAG_SIZE:],
    }


def cryptography_direct(problem: Problem) -> Solution:
    combined = ChaCha20Poly1305(problem["key"]).encrypt(
        problem["nonce"],
        problem["plaintext"],
        problem["associated_data"],
    )
    return _split_combined(combined)


def libsodium_ietf(problem: Problem) -> Solution:
    combined = crypto_aead_chacha20poly1305_ietf_encrypt(
        problem["plaintext"],
        problem["associated_data"],
        problem["nonce"],
        problem["key"],
    )
    return _split_combined(combined)


def pycryptodome(problem: Problem) -> Solution:
    cipher = ChaCha20_Poly1305.new(
        key=problem["key"],
        nonce=problem["nonce"],
    )
    associated_data = problem["associated_data"]
    if associated_data:
        cipher.update(associated_data)
    ciphertext, tag = cipher.encrypt_and_digest(problem["plaintext"])
    return {"ciphertext": ciphertext, "tag": tag}


CANDIDATES: dict[str, Callable[[Problem], Solution]] = {
    "cryptography_direct": cryptography_direct,
    "libsodium_ietf": libsodium_ietf,
    "pycryptodome": pycryptodome,
}
