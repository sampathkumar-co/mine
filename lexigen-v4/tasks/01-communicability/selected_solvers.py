from __future__ import annotations

from collections.abc import Callable

from candidates import Problem, Solution
from candidates import no_transfer_sort_bit_risk
from candidates import random_contiguous_vector_risk
from candidates import template_vectorized_batch
from candidates import v3_dtype_specialization
from candidates import v4_sort_bit_risk

SELECTED: dict[str, tuple[str, Callable[[Problem], Solution]]] = {
    "v4_full": ("v4_sort_bit_risk", v4_sort_bit_risk),
    "v4_no_transfer": ("no_transfer_sort_bit_risk", no_transfer_sort_bit_risk),
    "random_search": ("random_contiguous_vector_risk", random_contiguous_vector_risk),
    "template_synthesis": ("template_vectorized_batch", template_vectorized_batch),
    "v3_compatible": ("v3_dtype_specialization", v3_dtype_specialization),
}
