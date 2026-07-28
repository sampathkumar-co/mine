from __future__ import annotations

import os

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from candidates import CANDIDATES


for size in (0, 1, 15, 16, 17, 63, 64, 65, 1000, 1_000_003):
    for aad_size in (0, 1, 16, 17, 32):
        key = os.urandom(32)
        nonce = os.urandom(12)
        plaintext = os.urandom(size)
        associated_data = os.urandom(aad_size)
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
                aad_size,
                name,
            )

print("synthetic exactness passed")
