from __future__ import annotations

import os
import time

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from candidates import CANDIDATES


for size in (0, 1, 15, 16, 17, 63, 64, 65, 1000, 1_000_003):
    for associated_size in (0, 1, 16, 17, 32):
        key = os.urandom(32)
        nonce = os.urandom(12)
        plaintext = os.urandom(size)
        associated_data = os.urandom(associated_size)
        expected = ChaCha20Poly1305(key).encrypt(nonce, plaintext, associated_data)
        problem = {
            "key": key,
            "nonce": nonce,
            "plaintext": plaintext,
            "associated_data": associated_data,
        }
        for name, candidate in CANDIDATES.items():
            result = candidate(problem)
            assert result["ciphertext"] + result["tag"] == expected, (
                size,
                associated_size,
                name,
            )

benchmark_size = 64 * 1024 * 1024
key = os.urandom(32)
nonce = os.urandom(12)
plaintext = os.urandom(benchmark_size)
associated_data = os.urandom(32)
problem = {
    "key": key,
    "nonce": nonce,
    "plaintext": plaintext,
    "associated_data": associated_data,
}

started = time.perf_counter()
combined = ChaCha20Poly1305(key).encrypt(nonce, plaintext, associated_data)
reference_ciphertext = combined[:-16]
reference_tag = combined[-16:]
reference_seconds = time.perf_counter() - started

print(f"reference_seconds={reference_seconds:.9f}")
for name, candidate in CANDIDATES.items():
    started = time.perf_counter()
    result = candidate(problem)
    candidate_seconds = time.perf_counter() - started
    assert result["ciphertext"] == reference_ciphertext
    assert result["tag"] == reference_tag
    print(
        f"{name}_seconds={candidate_seconds:.9f} "
        f"speedup={reference_seconds / candidate_seconds:.6f}"
    )

print("synthetic exactness and diagnostic benchmark passed")
