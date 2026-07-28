from __future__ import annotations

import lexigen_chacha_native

Problem = dict[str, bytes]
Solution = dict[str, bytes]


def solve(problem: Problem) -> Solution:
    ciphertext, tag = lexigen_chacha_native.encrypt(
        problem["key"],
        problem["nonce"],
        problem["plaintext"],
        problem["associated_data"],
        4,
    )
    return {"ciphertext": ciphertext, "tag": tag}
