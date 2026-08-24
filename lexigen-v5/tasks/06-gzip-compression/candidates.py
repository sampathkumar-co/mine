from __future__ import annotations

import gzip
import zlib
from typing import Callable


def reference_exact(problem: dict) -> dict[str, bytes]:
    data = bytes(problem["plaintext"])
    return {"compressed_data": gzip.compress(data, compresslevel=9, mtime=0)}


def direct_zlib_gzip(problem: dict) -> dict[str, bytes]:
    data = bytes(problem["plaintext"])
    return {"compressed_data": zlib.compress(data, level=9, wbits=31)}


def reference_memoryview(problem: dict) -> dict[str, bytes]:
    data = memoryview(problem["plaintext"])
    return {"compressed_data": gzip.compress(data, compresslevel=9, mtime=0)}


def make_reference(_: str) -> Callable[[dict], dict[str, bytes]]:
    return reference_exact


def make_zero_copy(_: str) -> Callable[[dict], dict[str, bytes]]:
    return reference_memoryview


def make_native(_: str) -> Callable[[dict], dict[str, bytes]]:
    return direct_zlib_gzip


PROPOSALS = {
    "v5_full": [
        ("3304c859d463a501bd86", ["bit_parallel_representation","sparse_frontier_search","early_certificate_exit"], ["TM-BFR-01"], "bit_frontier_restriction"),
        ("41510e43e8fafb598496", ["reduced_representation","bounded_exact_refinement","risk_aware_staging"], ["TM-RRR-01"], "reduced_representation_refinement"),
        ("a6102573c9f355414229", ["active_set_decomposition","early_certificate_exit","risk_aware_staging"], ["TM-CAC-01"], "certified_active_core"),
        ("e909b567bac8aa01b86e", ["bit_parallel_representation","reduced_representation","bounded_exact_refinement"], [], None),
        ("2c6961f9ca6711ce3a3f", ["bit_parallel_representation","sparse_frontier_search","reduced_representation"], [], None),
        ("7653e3865aa7a6def4dc", ["bit_parallel_representation","sparse_frontier_search","bounded_exact_refinement"], [], None),
    ],
    "v5_no_transfer": [
        ("66c5848a3c8a4f51b562", ["bit_parallel_representation","reduced_representation","bounded_exact_refinement"], [], None),
        ("b93eda021fe3bc5d89cb", ["bit_parallel_representation","sparse_frontier_search","reduced_representation"], [], None),
        ("d14c06bd6ae45a8dd009", ["bit_parallel_representation","sparse_frontier_search","bounded_exact_refinement"], [], None),
        ("2c6f67fc6c6a0adc20f9", ["vectorized_batch_kernel","bit_parallel_representation","reduced_representation"], [], None),
        ("14a8ffbc5159ff111ac9", ["vectorized_batch_kernel","bit_parallel_representation","bounded_exact_refinement"], [], None),
        ("477905d60634240ebda9", ["vectorized_batch_kernel","bit_parallel_representation","sparse_frontier_search"], [], None),
    ],
    "random_search": [
        ("bcb2865badc647aa6bc2", ["vectorized_batch_kernel","bit_parallel_representation","bounded_exact_refinement"], [], None),
        ("3c492e53c279d6973df5", ["zero_copy_representation","sparse_frontier_search","reduced_representation"], [], None),
        ("93a27c8bc55defa5a597", ["dtype_specialization","native_backend_substitution","reduced_representation"], [], None),
        ("578671e42d1dbdb474df", ["zero_copy_representation","native_backend_substitution","reduced_representation"], [], None),
        ("4ff38c49ab25e45cbe27", ["native_backend_substitution","risk_aware_staging"], [], None),
        ("104070913b379afc4fb7", ["reduced_representation"], [], None),
    ],
    "static_template": [
        ("dbfcd2af539b0b2636e7", ["bit_parallel_representation","sparse_frontier_search"], [], None),
        ("8fd871e046faa7e4d37c", ["reduced_representation","bounded_exact_refinement"], [], None),
        ("820b1c309b6117eb268d", ["active_set_decomposition","early_certificate_exit"], [], None),
        ("8f1dafda0d3fbc099aa9", ["zero_copy_representation","vectorized_batch_kernel"], [], None),
        ("357e80313b8b9dc3cf36", ["contiguous_layout","vectorized_batch_kernel"], [], None),
        ("d044a19fd4551034dc11", ["dtype_specialization","risk_aware_staging"], [], None),
    ],
    "v4_compatible": [
        ("bd9a928b0a959b433de2", ["bit_parallel_representation","sparse_frontier_search","bounded_exact_refinement"], [], None),
        ("885bf4f21e819b330732", ["vectorized_batch_kernel","bit_parallel_representation","bounded_exact_refinement"], [], None),
        ("695b907772d8a69a1186", ["vectorized_batch_kernel","bit_parallel_representation","sparse_frontier_search"], [], None),
        ("d9863922b850e9717a05", ["risk_aware_staging","bit_parallel_representation","bounded_exact_refinement"], [], None),
        ("cdae8cbf0d73bd4d047c", ["vectorized_batch_kernel","sparse_frontier_search","bounded_exact_refinement"], [], None),
        ("af7d36f83a386b7726b9", ["risk_aware_staging","bit_parallel_representation","sparse_frontier_search"], [], None),
    ],
}

CANDIDATES_BY_ARM: dict[str, list[tuple[str, Callable]]] = {}
CANDIDATE_META: dict[str, dict] = {}
for arm, rows in PROPOSALS.items():
    built = []
    for rank, (pid, operators, transfer_ids, learned_template) in enumerate(rows, 1):
        name = f"{arm}_r{rank}_{pid}"
        if "native_backend_substitution" in operators:
            fn = make_native(name)
            implementation_class = "zlib_direct_gzip_level9"
        elif "zero_copy_representation" in operators:
            fn = make_zero_copy(name)
            implementation_class = "gzip_level9_memoryview"
        else:
            fn = make_reference(name)
            implementation_class = "gzip_reference_level9"
        built.append((name, fn))
        CANDIDATE_META[name] = {
            "arm": arm,
            "rank": rank,
            "proposal_id": pid,
            "operators": operators,
            "transfer_ids": transfer_ids,
            "learned_template": learned_template,
            "implementation_class": implementation_class,
            "mapping_policy": "native_backend_substitution->direct zlib gzip; zero_copy_representation->memoryview wrapper; otherwise conservative canonical gzip because no safe task-semantic realization follows from the frozen operator set"
        }
    CANDIDATES_BY_ARM[arm] = built

if sum(len(v) for v in CANDIDATES_BY_ARM.values()) != 30:
    raise RuntimeError("Task 6 candidate budget mismatch")
