from __future__ import annotations

from functools import lru_cache
from typing import Callable

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

Problem = dict[str, object]
Solution = dict[str, bytes]
Candidate = Callable[[Problem], Solution]

AES_KEY_SIZES = {16, 24, 32}
GCM_TAG_SIZE = 16


def _inputs(problem: Problem) -> tuple[bytes, bytes, bytes, bytes | None]:
    key = problem["key"]
    nonce = problem["nonce"]
    plaintext = problem["plaintext"]
    aad = problem.get("associated_data")
    if not isinstance(key, bytes) or len(key) not in AES_KEY_SIZES:
        raise ValueError("invalid AES key")
    if not isinstance(nonce, bytes):
        raise TypeError("nonce must be bytes")
    if not isinstance(plaintext, bytes):
        raise TypeError("plaintext must be bytes")
    if aad is not None and not isinstance(aad, bytes):
        raise TypeError("associated_data must be bytes or None")
    return key, nonce, plaintext, aad


def _high_level_exact(problem: Problem) -> Solution:
    key, nonce, plaintext, aad = _inputs(problem)
    combined = AESGCM(key).encrypt(nonce, plaintext, aad)
    if len(combined) < GCM_TAG_SIZE:
        raise ValueError("encrypted output shorter than GCM tag")
    return {"ciphertext": combined[:-GCM_TAG_SIZE], "tag": combined[-GCM_TAG_SIZE:]}


def _native_cipher_exact(problem: Problem) -> Solution:
    key, nonce, plaintext, aad = _inputs(problem)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(nonce)).encryptor()
    if aad is not None:
        encryptor.authenticate_additional_data(aad)
    ciphertext = encryptor.update(plaintext)
    tail = encryptor.finalize()
    if tail:
        ciphertext += tail
    return {"ciphertext": ciphertext, "tag": bytes(encryptor.tag)}


def _generic_pipeline(problem: Problem, operators: tuple[str, ...]) -> Solution:
    # Frozen generic instantiation: only a real native-backend operator changes execution.
    # Bytes have no meaningful dtype, batch, active-set, frontier, reduced-representation,
    # or approximate-error semantics here, so those operators preserve exact AESGCM.
    if "native_backend_substitution" in operators:
        return _native_cipher_exact(problem)
    return _high_level_exact(problem)


def _learned_pipeline(problem: Problem, learned_template: str) -> Solution:
    if learned_template == "precision_backend_error_budget":
        # Exact byte verifier => zero numerical error budget. Precision cannot be relaxed;
        # only the exact lower-level native cipher path is admissible.
        return _native_cipher_exact(problem)
    if learned_template in {"bit_frontier_restriction", "certified_active_core"}:
        # Every plaintext/AAD byte contributes to AES-GCM output/authentication, so no
        # graph frontier or skippable active core exists. Preserve the recipe's exact fallback.
        return _high_level_exact(problem)
    raise ValueError(f"unknown learned template: {learned_template}")


@lru_cache(maxsize=None)
def _implementation(operators: tuple[str, ...], learned_template: str | None) -> Candidate:
    if learned_template is None:
        def candidate(problem: Problem) -> Solution:
            return _generic_pipeline(problem, operators)
    else:
        def candidate(problem: Problem) -> Solution:
            return _learned_pipeline(problem, learned_template)
    return candidate


PROPOSALS: dict[str, list[tuple[str, tuple[str, ...], tuple[str, ...], str | None]]] = {
    "v5_full": [
        ("3304c859d463a501bd86", ("bit_parallel_representation", "sparse_frontier_search", "early_certificate_exit"), ("TM-BFR-01",), "bit_frontier_restriction"),
        ("a6102573c9f355414229", ("active_set_decomposition", "early_certificate_exit", "risk_aware_staging"), ("TM-CAC-01",), "certified_active_core"),
        ("b1ef08a2d68a248c0821", ("dtype_specialization", "native_backend_substitution", "risk_aware_staging"), ("TM-PBEB-01",), "precision_backend_error_budget"),
        ("7653e3865aa7a6def4dc", ("bit_parallel_representation", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("c8350e5b9ffb9c400fc8", ("vectorized_batch_kernel", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("ef8d1d6c3a4fa6d0be21", ("vectorized_batch_kernel", "bit_parallel_representation", "sparse_frontier_search"), (), None),
    ],
    "v5_no_transfer": [
        ("d14c06bd6ae45a8dd009", ("bit_parallel_representation", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("14a8ffbc5159ff111ac9", ("vectorized_batch_kernel", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("477905d60634240ebda9", ("vectorized_batch_kernel", "bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("1b859d0377c9b2a19b53", ("native_backend_substitution", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("0a141855078f60fe2b98", ("risk_aware_staging", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("974e0d50c53bf65218c5", ("vectorized_batch_kernel", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
    ],
    "random_search": [
        ("ed1f4796fdbc83a45b55", ("zero_copy_representation", "vectorized_batch_kernel", "bit_parallel_representation"), (), None),
        ("0520aab8496f3d685f92", ("dtype_specialization", "vectorized_batch_kernel", "bounded_exact_refinement"), (), None),
        ("e9c9180b0957bb3f7ca7", ("native_backend_substitution", "risk_aware_staging", "bounded_exact_refinement"), (), None),
        ("01baad563ce94255e3e2", ("native_backend_substitution", "sparse_frontier_search"), (), None),
        ("ae3b52160647eaf9707e", ("risk_aware_staging",), (), None),
        ("8615e4a35db08222a26b", ("dtype_specialization",), (), None),
    ],
    "static_template": [
        ("dbfcd2af539b0b2636e7", ("bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("8fd871e046faa7e4d37c", ("reduced_representation", "bounded_exact_refinement"), (), None),
        ("820b1c309b6117eb268d", ("active_set_decomposition", "early_certificate_exit"), (), None),
        ("8f1dafda0d3fbc099aa9", ("zero_copy_representation", "vectorized_batch_kernel"), (), None),
        ("d044a19fd4551034dc11", ("dtype_specialization", "risk_aware_staging"), (), None),
        ("357e80313b8b9dc3cf36", ("contiguous_layout", "vectorized_batch_kernel"), (), None),
    ],
    "v4_compatible": [
        ("bd9a928b0a959b433de2", ("bit_parallel_representation", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("885bf4f21e819b330732", ("vectorized_batch_kernel", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("695b907772d8a69a1186", ("vectorized_batch_kernel", "bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("d9863922b850e9717a05", ("risk_aware_staging", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("cdae8cbf0d73bd4d047c", ("vectorized_batch_kernel", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("af7d36f83a386b7726b9", ("risk_aware_staging", "bit_parallel_representation", "sparse_frontier_search"), (), None),
    ],
}

CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {}
PROVENANCE: dict[str, list[dict[str, object]]] = {}
for arm, rows in PROPOSALS.items():
    CANDIDATES_BY_ARM[arm] = []
    PROVENANCE[arm] = []
    for rank, (proposal_id, operators, transfer_ids, learned_template) in enumerate(rows, 1):
        name = f"{arm}_r{rank}_{proposal_id}"
        fn = _implementation(operators, learned_template)
        implementation_class = (
            "native_cipher_exact"
            if learned_template == "precision_backend_error_budget" or "native_backend_substitution" in operators
            else "high_level_exact"
        )
        CANDIDATES_BY_ARM[arm].append((name, fn))
        PROVENANCE[arm].append({
            "candidate": name,
            "proposal_id": proposal_id,
            "rank": rank,
            "operators": list(operators),
            "transfer_ids": list(transfer_ids),
            "learned_template": learned_template,
            "implementation_class": implementation_class,
            "semantic_signature": ["learned" if learned_template else "generic", learned_template or "none", list(operators), implementation_class],
        })

if sum(len(rows) for rows in CANDIDATES_BY_ARM.values()) != 30:
    raise RuntimeError("expected exactly 30 frozen Task 2 candidates")
