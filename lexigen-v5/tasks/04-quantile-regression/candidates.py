from __future__ import annotations

from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from sklearn.linear_model import QuantileRegressor

Problem = dict[str, object]
Solution = dict[str, object]
Candidate = Callable[[Problem], Solution]


def _inputs(problem: Problem) -> tuple[np.ndarray, np.ndarray, float, bool]:
    X = np.asarray(problem["X"], dtype=float)
    y = np.asarray(problem["y"], dtype=float)
    q = float(problem["quantile"])
    fit = bool(problem["fit_intercept"])
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("invalid quantile-regression input shape")
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be in (0,1)")
    return X, y, q, fit


def _pack(X: np.ndarray, coef: np.ndarray, intercept: float) -> Solution:
    return {
        "coef": np.asarray(coef, dtype=float).tolist(),
        "intercept": [float(intercept)],
        "predictions": np.asarray(X @ coef + intercept, dtype=float).tolist(),
    }


def _reference_sklearn(problem: Problem) -> Solution:
    X, y, q, fit = _inputs(problem)
    model = QuantileRegressor(quantile=q, alpha=0.0, fit_intercept=fit, solver="highs")
    model.fit(X, y)
    intercept = float(model.intercept_) if fit else 0.0
    return _pack(X, np.asarray(model.coef_, dtype=float), intercept)


def _same_split_lp(problem: Problem, method: str = "highs") -> Solution:
    """Source-equivalent LP with sklearn's split parameters but no estimator/validation layer."""
    X, y, q, fit = _inputs(problem)
    n, p = X.shape
    n_params = p + int(fit)
    c = np.concatenate([np.zeros(2 * n_params), np.full(n, q), np.full(n, 1.0 - q)])
    Xc = sparse.csc_matrix(X)
    eye = sparse.eye(n, dtype=X.dtype, format="csc")
    if fit:
        ones = sparse.csc_matrix(np.ones((n, 1), dtype=X.dtype))
        A_eq = sparse.hstack([ones, Xc, -ones, -Xc, eye, -eye], format="csc")
    else:
        A_eq = sparse.hstack([Xc, -Xc, eye, -eye], format="csc")
    result = linprog(c=c, A_eq=A_eq, b_eq=y, method=method)
    if not result.success or result.x is None:
        raise RuntimeError(f"HiGHS split LP failed: {result.message}")
    params = result.x[:n_params] - result.x[n_params : 2 * n_params]
    if fit:
        intercept = float(params[0])
        coef = np.asarray(params[1:], dtype=float)
    else:
        intercept = 0.0
        coef = np.asarray(params, dtype=float)
    return _pack(X, coef, intercept)


def _free_parameter_lp(problem: Problem, method: str = "highs") -> Solution:
    """Reduced exact LP: regression parameters are free variables instead of positive/negative splits."""
    X, y, q, fit = _inputs(problem)
    n, p = X.shape
    if fit:
        design = sparse.hstack(
            [sparse.csc_matrix(np.ones((n, 1), dtype=X.dtype)), sparse.csc_matrix(X)],
            format="csc",
        )
        n_params = p + 1
    else:
        design = sparse.csc_matrix(X)
        n_params = p
    eye = sparse.eye(n, dtype=X.dtype, format="csc")
    A_eq = sparse.hstack([design, eye, -eye], format="csc")
    c = np.concatenate([np.zeros(n_params), np.full(n, q), np.full(n, 1.0 - q)])
    bounds = [(None, None)] * n_params + [(0.0, None)] * (2 * n)
    result = linprog(c=c, A_eq=A_eq, b_eq=y, bounds=bounds, method=method)
    if not result.success or result.x is None:
        return _same_split_lp(problem)
    params = np.asarray(result.x[:n_params], dtype=float)
    if fit:
        intercept = float(params[0])
        coef = params[1:]
    else:
        intercept = 0.0
        coef = params
    return _pack(X, coef, intercept)


def _dual_active_core(problem: Problem) -> Solution:
    """Learned CAC recipe: solve the smaller dual, recover a full-rank zero-residual core, certify KKT, else exact fallback."""
    X, y, q, fit = _inputs(problem)
    n, p = X.shape
    design = np.column_stack([np.ones(n, dtype=float), X]) if fit else X
    n_params = design.shape[1]
    result = linprog(
        c=-y,
        A_eq=design.T,
        b_eq=np.zeros(n_params, dtype=float),
        bounds=[(q - 1.0, q)] * n,
        method="highs",
    )
    if not result.success or result.x is None:
        return _same_split_lp(problem)
    dual = np.asarray(result.x, dtype=float)
    interior_distance = np.minimum(dual - (q - 1.0), q - dual)
    order = np.argsort(-interior_distance, kind="stable")
    chosen: list[int] = []
    rank = 0
    for raw_idx in order:
        idx = int(raw_idx)
        trial = chosen + [idx]
        trial_rank = int(np.linalg.matrix_rank(design[trial], tol=1e-10))
        if trial_rank > rank:
            chosen.append(idx)
            rank = trial_rank
            if rank == n_params:
                break
    if rank != n_params or len(chosen) != n_params:
        return _same_split_lp(problem)
    try:
        params = np.linalg.solve(design[chosen], y[chosen])
    except np.linalg.LinAlgError:
        return _same_split_lp(problem)

    residual = y - design @ params
    dual_feas = np.max(np.abs(design.T @ dual)) <= 2e-7
    pos_ok = np.all(np.abs(dual[residual > 2e-7] - q) <= 2e-6)
    neg_ok = np.all(np.abs(dual[residual < -2e-7] - (q - 1.0)) <= 2e-6)
    if not (dual_feas and pos_ok and neg_ok and np.all(np.isfinite(params))):
        return _same_split_lp(problem)

    if fit:
        intercept = float(params[0])
        coef = np.asarray(params[1:], dtype=float)
    else:
        intercept = 0.0
        coef = np.asarray(params, dtype=float)
    return _pack(X, coef, intercept)


def _generic_pipeline(problem: Problem, operators: tuple[str, ...]) -> Solution:
    ops = set(operators)
    if "active_set_decomposition" in ops:
        # Generic active-set operator is intentionally not allowed to instantiate the learned dual CAC composition.
        return _same_split_lp(problem, method="highs-ds")
    if "reduced_representation" in ops:
        return _free_parameter_lp(problem)
    if ops.intersection({"native_backend_substitution", "vectorized_batch_kernel", "zero_copy_representation", "contiguous_layout", "dtype_specialization"}):
        return _same_split_lp(problem)
    return _reference_sklearn(problem)


def _learned_pipeline(problem: Problem, learned_template: str) -> Solution:
    if learned_template == "certified_active_core":
        return _dual_active_core(problem)
    if learned_template == "reduced_representation_refinement":
        return _free_parameter_lp(problem)
    if learned_template == "bit_frontier_restriction":
        # Continuous dense regression has no legitimate bit/frontier restriction; preserve exact fallback.
        return _reference_sklearn(problem)
    raise ValueError(f"unknown learned template: {learned_template}")


PROPOSALS: dict[str, list[tuple[str, tuple[str, ...], tuple[str, ...], str | None]]] = {
    "v5_full": [
        ("3304c859d463a501bd86", ("bit_parallel_representation", "sparse_frontier_search", "early_certificate_exit"), ("TM-BFR-01",), "bit_frontier_restriction"),
        ("41510e43e8fafb598496", ("reduced_representation", "bounded_exact_refinement", "risk_aware_staging"), ("TM-RRR-01",), "reduced_representation_refinement"),
        ("a6102573c9f355414229", ("active_set_decomposition", "early_certificate_exit", "risk_aware_staging"), ("TM-CAC-01",), "certified_active_core"),
        ("4abf2b51384c560522e8", ("active_set_decomposition", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("c50e493c5549a408f3e5", ("active_set_decomposition", "bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("514b3e8a41ba1f8b73a1", ("active_set_decomposition", "bit_parallel_representation", "reduced_representation"), (), None),
    ],
    "v5_no_transfer": [
        ("91e027e622f2d9a98240", ("active_set_decomposition", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("b2614109e1a5ccc10c14", ("active_set_decomposition", "bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("20375ceceffce4d406a4", ("active_set_decomposition", "bit_parallel_representation", "reduced_representation"), (), None),
        ("4a4e1871b7f7b48b9485", ("active_set_decomposition", "bit_parallel_representation", "sort_partition_reduction"), (), None),
        ("d69e86803f54c5a83d06", ("vectorized_batch_kernel", "active_set_decomposition", "bit_parallel_representation"), (), None),
        ("66c5848a3c8a4f51b562", ("bit_parallel_representation", "reduced_representation", "bounded_exact_refinement"), (), None),
    ],
    "random_search": [
        ("399ba5e6f15e49b3e885", ("vectorized_batch_kernel", "sparse_frontier_search", "reduced_representation"), (), None),
        ("281d4a03f9bc5812f7af", ("dtype_specialization", "vectorized_batch_kernel", "bit_parallel_representation"), (), None),
        ("3667208f0eec7d49161f", ("contiguous_layout", "risk_aware_staging", "bit_parallel_representation"), (), None),
        ("d59864ed9ab0297d5542", ("zero_copy_representation", "early_certificate_exit", "sparse_frontier_search"), (), None),
        ("83aa0674a4d1af4a6b66", ("vectorized_batch_kernel", "bounded_exact_refinement"), (), None),
        ("f809298bda98f85e0e1b", ("native_backend_substitution",), (), None),
    ],
    "static_template": [
        ("dbfcd2af539b0b2636e7", ("bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("8fd871e046faa7e4d37c", ("reduced_representation", "bounded_exact_refinement"), (), None),
        ("820b1c309b6117eb268d", ("active_set_decomposition", "early_certificate_exit"), (), None),
        ("8f1dafda0d3fbc099aa9", ("zero_copy_representation", "vectorized_batch_kernel"), (), None),
        ("357e80313b8b9dc3cf36", ("contiguous_layout", "vectorized_batch_kernel"), (), None),
        ("d044a19fd4551034dc11", ("dtype_specialization", "risk_aware_staging"), (), None),
    ],
    "v4_compatible": [
        ("f9f3239b6866512e4f68", ("active_set_decomposition", "bit_parallel_representation", "bounded_exact_refinement"), (), None),
        ("9f5f55df04a5ad23f542", ("active_set_decomposition", "bit_parallel_representation", "sparse_frontier_search"), (), None),
        ("ec4b9c17aaa3767d4f6d", ("active_set_decomposition", "bit_parallel_representation", "sort_partition_reduction"), (), None),
        ("7c30efb65d2c20ff8cc9", ("vectorized_batch_kernel", "active_set_decomposition", "bit_parallel_representation"), (), None),
        ("3df5ed91505aea4ed6cb", ("active_set_decomposition", "sparse_frontier_search", "bounded_exact_refinement"), (), None),
        ("0dde88a4a159a3ad0e40", ("bit_parallel_representation", "bounded_exact_refinement", "sort_partition_reduction"), (), None),
    ],
}


@lru_cache(maxsize=None)
def _implementation(operators: tuple[str, ...], learned_template: str | None) -> Candidate:
    if learned_template is None:
        def candidate(problem: Problem) -> Solution:
            return _generic_pipeline(problem, operators)
    else:
        def candidate(problem: Problem) -> Solution:
            return _learned_pipeline(problem, learned_template)
    return candidate


CANDIDATES_BY_ARM: dict[str, list[tuple[str, Candidate]]] = {}
PROVENANCE: dict[str, list[dict[str, object]]] = {}
for arm, rows in PROPOSALS.items():
    CANDIDATES_BY_ARM[arm] = []
    PROVENANCE[arm] = []
    for rank, (proposal_id, operators, transfer_ids, learned_template) in enumerate(rows, 1):
        name = f"{arm}_r{rank}_{proposal_id}"
        fn = _implementation(operators, learned_template)
        if learned_template == "certified_active_core":
            implementation_class = "dual_certified_active_core"
        elif learned_template == "reduced_representation_refinement":
            implementation_class = "free_parameter_lp_learned_rrr"
        elif learned_template == "bit_frontier_restriction":
            implementation_class = "reference_exact_fallback"
        elif "active_set_decomposition" in operators:
            implementation_class = "split_parameter_highs_ds_generic"
        elif "reduced_representation" in operators:
            implementation_class = "free_parameter_lp_generic"
        elif set(operators).intersection({"native_backend_substitution", "vectorized_batch_kernel", "zero_copy_representation", "contiguous_layout", "dtype_specialization"}):
            implementation_class = "split_parameter_highs_generic"
        else:
            implementation_class = "reference_exact_fallback"
        CANDIDATES_BY_ARM[arm].append((name, fn))
        PROVENANCE[arm].append({
            "candidate": name,
            "proposal_id": proposal_id,
            "rank": rank,
            "operators": list(operators),
            "transfer_ids": list(transfer_ids),
            "learned_template": learned_template,
            "implementation_class": implementation_class,
            "semantic_signature": ["learned" if learned_template else "generic", learned_template or "none", implementation_class],
        })

if sum(len(rows) for rows in CANDIDATES_BY_ARM.values()) != 30:
    raise RuntimeError("expected exactly 30 frozen Task 4 candidates")
