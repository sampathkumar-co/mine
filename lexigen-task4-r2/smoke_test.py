from __future__ import annotations

import hashlib

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

import lexigen_chacha_native


def deterministic_bytes(label: bytes, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        output.extend(hashlib.sha256(label + counter.to_bytes(8, "little")).digest())
        counter += 1
    return bytes(output[:length])


def main() -> None:
    sizes = [0, 1, 15, 16, 63, 64, 65, 4096, 1_048_613]
    for case_index, size in enumerate(sizes):
        key = deterministic_bytes(b"key" + bytes([case_index]), 32)
        nonce = deterministic_bytes(b"nonce" + bytes([case_index]), 12)
        plaintext = deterministic_bytes(b"plaintext" + bytes([case_index]), size)
        associated_data = b"" if case_index % 2 else deterministic_bytes(b"aad" + bytes([case_index]), 32)
        expected = ChaCha20Poly1305(key).encrypt(nonce, plaintext, associated_data)
        expected_ciphertext, expected_tag = expected[:-16], expected[-16:]
        for threads in (1, 2, 4, 8):
            ciphertext, tag = lexigen_chacha_native.encrypt(
                key, nonce, plaintext, associated_data, threads
            )
            assert ciphertext == expected_ciphertext, (case_index, size, threads, "ciphertext")
            assert tag == expected_tag, (case_index, size, threads, "tag")
            print(f"case={case_index} size={size} threads={threads} exact=true", flush=True)
    print("LEXIGEN_TASK4_R2_SYNTHETIC_EXACTNESS_PASS")


if __name__ == "__main__":
    main()
