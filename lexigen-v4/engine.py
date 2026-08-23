from __future__ import annotations

import ast
import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

ENGINE_VERSION = "lexigen-v4.0.0-frozen"


@dataclass(frozen=True)
class Fingerprint:
    source_sha256: str
    verifier_sha256: str
    features: tuple[str, ...]
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    dependency_calls: tuple[str, ...]
    numeric_constants: tuple[float, ...]
    ast_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Operator:
    name: str
    category: str
    requires_any: tuple[str, ...]
    avoids: tuple[str, ...]
    provides: tuple[str, ...]
    base_benefit: float
    base_risk: float
    composition_group: str


@dataclass(frozen=True)
class Proposal:
    arm: str
    rank: int
    operators: tuple[str, ...]
    score: float
    predicted_benefit: float
    correctness_risk: float
    rationale: tuple[str, ...]
    proposal_id: str


TRANSFER_MEMORY: dict[str, dict[str, object]] = {
    "structure_aware_initialization": {
        "success_families": ["clustering"],
        "failures": [],
        "weight": 1.20,
        "lesson": "Use generator-implied grouping only when represented directly in the permitted specification.",
    },
    "active_set_decomposition": {
        "success_families": ["convex_optimization"],
        "failures": [],
        "weight": 1.15,
        "lesson": "Exploit low-dimensional active constraints while retaining an exact global certificate.",
    },
    "native_one_shot_backend": {
        "success_families": ["encoding"],
        "failures": ["large_streaming_crypto"],
        "weight": 0.20,
        "lesson": "Wrapper removal helps only when wrapper cost is material relative to the native kernel.",
    },
    "mixed_precision_with_local_refinement": {
        "success_families": [],
        "failures": ["unconditional_low_precision", "full_precision_fallback"],
        "weight": 0.65,
        "lesson": "Correct only the numerically weak subspace instead of recomputing the complete decomposition.",
    },
    "zero_copy_representation": {
        "success_families": ["encoding", "cryptography"],
        "failures": [],
        "weight": 0.45,
        "lesson": "Avoid materialising equivalent buffers when the native API accepts the original representation.",
    },
    "bounded_exact_refinement": {
        "success_families": ["clustering", "convex_optimization"],
        "failures": [],
        "weight": 0.85,
        "lesson": "Begin from a structural solution and spend bounded work only to restore the official certificate.",
    },
}


OPERATORS: tuple[Operator, ...] = (
    Operator("zero_copy_representation", "representation", ("bytes", "array", "matrix"), (), ("lower_overhead",), 0.35, 0.05, "representation"),
    Operator("contiguous_layout", "representation", ("array", "matrix", "tensor"), (), ("native_layout",), 0.30, 0.04, "representation"),
    Operator("dtype_specialization", "representation", ("numeric", "matrix", "array"), ("exact_integer",), ("reduced_bandwidth",), 0.52, 0.38, "precision"),
    Operator("mixed_precision_with_local_refinement", "numerical", ("matrix", "decomposition", "tolerance"), ("bit_exact",), ("reduced_decomposition_cost", "risk_guard"), 0.88, 0.34, "precision"),
    Operator("structure_aware_initialization", "algorithm", ("grouped_generator", "cluster", "block_structure"), (), ("warm_start",), 0.92, 0.16, "initialization"),
    Operator("bounded_exact_refinement", "algorithm", ("warm_start", "approximate_verifier", "iterative"), (), ("certificate_restoration",), 0.72, 0.12, "refinement"),
    Operator("active_set_decomposition", "algorithm", ("constraints", "projection", "convex"), (), ("small_active_core",), 0.90, 0.19, "decomposition"),
    Operator("symmetric_decomposition", "algorithm", ("matrix", "symmetric", "decomposition"), (), ("specialized_kernel",), 0.58, 0.17, "decomposition"),
    Operator("sort_partition_reduction", "algorithm", ("order_statistic", "threshold", "topk", "quantile"), (), ("reduced_search_space",), 0.68, 0.10, "decomposition"),
    Operator("bit_parallel_representation", "algorithm", ("boolean", "bit", "set", "graph"), (), ("word_parallelism",), 0.84, 0.13, "representation"),
    Operator("sparse_frontier_search", "algorithm", ("graph", "sparse", "discrete"), (), ("frontier_restriction",), 0.68, 0.11, "search"),
    Operator("dynamic_programming_state_compression", "algorithm", ("sequence", "discrete", "recurrence"), (), ("state_reuse",), 0.76, 0.12, "search"),
    Operator("native_one_shot_backend", "backend", ("bytes", "encoding", "crypto", "hash"), (), ("lower_overhead",), 0.48, 0.05, "backend"),
    Operator("vectorized_batch_kernel", "backend", ("array", "batch", "numeric"), (), ("batch_parallelism",), 0.62, 0.08, "backend"),
    Operator("closed_form_reduction", "algorithm", ("formula", "projection", "statistics", "linear"), (), ("remove_iteration",), 0.82, 0.14, "decomposition"),
    Operator("risk_aware_staging", "control", ("tolerance", "approximate_verifier", "decomposition", "iterative"), (), ("risk_guard",), 0.50, -0.12, "control"),
    Operator("early_certificate_exit", "control", ("certificate", "verifier", "iterative"), (), ("bounded_work",), 0.38, -0.08, "control"),
)

V3_ALLOWED = {
    "zero_copy_representation",
    "contiguous_layout",
    "native_one_shot_backend",
    "vectorized_batch_kernel",
    "dtype_specialization",
}


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()
        self.calls: set[str] = set()
        self.strings: list[str] = []
        self.numbers: list[float] = []
        self.counts: dict[str, int] = {}

    def generic_visit(self, node: ast.AST) -> None:
        name = type(node).__name__
        self.counts[name] = self.counts.get(name, 0) + 1
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id.lower())
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        self.names.add(".".join(reversed(parts)).lower())
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self.calls.add(node.func.id.lower())
        elif isinstance(node.func, ast.Attribute):
            parts: list[str] = []
            current: ast.AST = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            self.calls.add(".".join(reversed(parts)).lower())
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append(node.value)
        elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            self.numbers.append(float(node.value))
        self.generic_visit(node)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(visitor: _Visitor) -> set[str]:
    tokens: set[str] = set(visitor.names) | set(visitor.calls)
    for value in visitor.strings:
        tokens.update(part.lower() for part in value.replace("-", "_").split("_") if part)
    return tokens


def _derive_features(tokens: set[str], counts: dict[str, int], numbers: Sequence[float]) -> set[str]:
    features: set[str] = set()
    joined = " ".join(sorted(tokens))

    rules = {
        "bytes": ("bytes", "plaintext", "digest", "encode", "decode", "hash"),
        "array": ("numpy", "np.asarray", "ndarray", "array", "tensor"),
        "matrix": ("svd", "eigh", "matrix", "matmul", "dot", "transpose", "linalg"),
        "decomposition": ("svd", "eigh", "eig", "qr", "cholesky", "polar"),
        "graph": ("graph", "edge", "vertex", "node", "adjacency", "neighbor"),
        "set": ("set", "subset", "cover", "union", "intersection"),
        "bit": ("bit", "xor", "and", "or", "mask"),
        "boolean": ("bool", "true", "false"),
        "cluster": ("cluster", "kmeans", "centroid", "labels"),
        "projection": ("project", "projection", "simplex", "cvar"),
        "constraints": ("constraint", "feasible", "bound", "lambda", "dual"),
        "convex": ("convex", "cvx", "projection", "dual"),
        "statistics": ("mean", "median", "variance", "quantile", "probability"),
        "order_statistic": ("sort", "partition", "quantile", "median", "topk"),
        "threshold": ("threshold", "cutoff", "tolerance", "tol"),
        "topk": ("topk", "argpartition", "largest", "smallest"),
        "sequence": ("sequence", "string", "list", "prefix", "suffix"),
        "recurrence": ("dynamic", "memo", "recurrence", "dp"),
        "sparse": ("sparse", "csr", "csc", "adjacency"),
        "batch": ("batch", "axis", "vectorize", "broadcast"),
        "iterative": ("max_iter", "iteration", "while", "converge"),
        "certificate": ("is_solution", "valid", "verify", "certificate"),
        "verifier": ("is_solution", "verify", "allclose", "compare_digest"),
        "crypto": ("cryptography", "cipher", "encrypt", "decrypt", "digest"),
        "hash": ("sha", "hash", "digest"),
        "encoding": ("encode", "decode", "base64", "codec"),
        "linear": ("linear", "matrix", "dot", "matmul"),
        "formula": ("formula", "closed", "analytic"),
        "grouped_generator": ("reshape", "repeat", "stack", "block", "group"),
        "block_structure": ("reshape", "block", "stack", "chunk"),
        "symmetric": ("symmetric", "eigh", "hermitian"),
        "numeric": ("numpy", "float", "int", "scipy", "math"),
        "discrete": ("graph", "integer", "combin", "set", "sequence"),
    }
    for feature, needles in rules.items():
        if any(needle in joined for needle in needles):
            features.add(feature)

    if any(abs(value - round(value)) > 0.0 for value in numbers):
        features.add("tolerance")
    if any(0.0 < abs(value) < 0.1 for value in numbers):
        features.add("tolerance")
    if "compare_digest" in joined or "==" in joined:
        features.add("bit_exact")
    if counts.get("For", 0) + counts.get("While", 0) > 0:
        features.add("iterative")
    if counts.get("MatMult", 0) > 0:
        features.update(("matrix", "linear"))
    if counts.get("Compare", 0) > 0:
        features.add("verifier")
    return features


def fingerprint(task_source: str, verifier_source: str = "") -> Fingerprint:
    task_tree = ast.parse(task_source)
    verifier_tree = ast.parse(verifier_source) if verifier_source.strip() else ast.parse("")
    visitor = _Visitor()
    visitor.visit(task_tree)
    verifier_visitor = _Visitor()
    verifier_visitor.visit(verifier_tree)
    tokens = _tokens(visitor) | _tokens(verifier_visitor)
    counts = dict(visitor.counts)
    for key, value in verifier_visitor.counts.items():
        counts[key] = counts.get(key, 0) + value
    numbers = sorted(set(visitor.numbers + verifier_visitor.numbers))
    features = _derive_features(tokens, counts, numbers)

    strings = visitor.strings + verifier_visitor.strings
    likely_keys = sorted({value for value in strings if value.isidentifier() and len(value) <= 40})
    input_keys = tuple(key for key in likely_keys if key not in {"solve", "is_solution", "problem", "solution"})
    output_keys = tuple(key for key in input_keys if key in {"digest", "labels", "value", "result", "solution", "x", "y", "output"})

    return Fingerprint(
        source_sha256=_sha(task_source),
        verifier_sha256=_sha(verifier_source),
        features=tuple(sorted(features)),
        input_keys=input_keys,
        output_keys=output_keys,
        dependency_calls=tuple(sorted(visitor.calls | verifier_visitor.calls)),
        numeric_constants=tuple(numbers[:64]),
        ast_counts=tuple(sorted(counts.items())),
    )


def _eligible(operator: Operator, known: set[str]) -> bool:
    if operator.requires_any and not known.intersection(operator.requires_any):
        return False
    return not known.intersection(operator.avoids)


def _operator_score(operator: Operator, known: set[str], use_transfer: bool) -> tuple[float, float, tuple[str, ...]]:
    matched = sorted(known.intersection(operator.requires_any))
    transfer = float(TRANSFER_MEMORY.get(operator.name, {}).get("weight", 0.0)) if use_transfer else 0.0
    risk = max(0.0, operator.base_risk)
    score = operator.base_benefit + 0.16 * len(matched) + 0.35 * transfer - 0.55 * risk
    rationale = [f"matched:{item}" for item in matched]
    if use_transfer and operator.name in TRANSFER_MEMORY:
        rationale.append(f"transfer_weight:{transfer:.3f}")
        rationale.append(str(TRANSFER_MEMORY[operator.name]["lesson"]))
    return score, risk, tuple(rationale)


def _compose(operators: Sequence[Operator], features: set[str]) -> tuple[float, float, set[str], tuple[str, ...]]:
    known = set(features)
    total = 0.0
    risk = 0.0
    rationale: list[str] = []
    groups: set[str] = set()
    for operator in operators:
        if operator.composition_group in groups and operator.composition_group not in {"control"}:
            total -= 0.30
        groups.add(operator.composition_group)
        known.update(operator.provides)
        total += operator.base_benefit
        risk += operator.base_risk
        rationale.append(f"compose:{operator.name}")
    if "risk_guard" in known:
        risk *= 0.72
    if "warm_start" in known and "certificate_restoration" in known:
        total += 0.42
        rationale.append("synergy:warm_start_plus_certificate")
    if "reduced_decomposition_cost" in known and "risk_guard" in known:
        total += 0.36
        rationale.append("synergy:reduced_precision_plus_guard")
    if "lower_overhead" in known and "batch_parallelism" in known:
        total += 0.18
        rationale.append("synergy:zero_copy_batch")
    return total, max(0.0, risk), known, tuple(rationale)


def generate_proposals(
    task_fingerprint: Fingerprint,
    arm: str = "v4_full",
    limit: int = 6,
    random_seed: str = "LEXIGEN-V4",
) -> list[Proposal]:
    if arm not in {"v4_full", "v4_no_transfer", "random_search", "template_synthesis", "v3_compatible"}:
        raise ValueError(f"unknown arm: {arm}")
    if limit <= 0:
        return []

    features = set(task_fingerprint.features)
    pool = [operator for operator in OPERATORS if _eligible(operator, features)]
    if arm == "v3_compatible":
        pool = [operator for operator in pool if operator.name in V3_ALLOWED]
    use_transfer = arm == "v4_full"

    candidates: list[tuple[tuple[Operator, ...], float, float, tuple[str, ...]]] = []
    if arm in {"template_synthesis", "v3_compatible"}:
        lengths = (1,)
    else:
        lengths = (1, 2, 3)

    for length in lengths:
        for composition in itertools.combinations(pool, length):
            known = set(features)
            compatible = True
            score = 0.0
            risk = 0.0
            reasons: list[str] = []
            for operator in composition:
                if not _eligible(operator, known):
                    compatible = False
                    break
                operator_score, operator_risk, operator_reasons = _operator_score(operator, known, use_transfer)
                score += operator_score
                risk += operator_risk
                reasons.extend(operator_reasons)
                known.update(operator.provides)
            if not compatible:
                continue
            synergy_score, composed_risk, _, synergy_reasons = _compose(composition, features)
            score += 0.34 * synergy_score
            risk = 0.55 * risk + 0.45 * composed_risk
            score -= 0.65 * risk
            score -= 0.10 * max(0, length - 1)
            reasons.extend(synergy_reasons)
            candidates.append((composition, score, risk, tuple(reasons)))

    candidates.sort(key=lambda row: (-row[1], row[2], tuple(operator.name for operator in row[0])))
    if arm == "random_search":
        seed_material = f"{random_seed}\0{task_fingerprint.source_sha256}\0{task_fingerprint.verifier_sha256}"
        rng = random.Random(int(hashlib.sha256(seed_material.encode()).hexdigest(), 16))
        rng.shuffle(candidates)

    proposals: list[Proposal] = []
    seen: set[tuple[str, ...]] = set()
    for composition, score, risk, reasons in candidates:
        names = tuple(operator.name for operator in composition)
        if names in seen:
            continue
        seen.add(names)
        benefit = sum(operator.base_benefit for operator in composition)
        payload = {
            "engine": ENGINE_VERSION,
            "arm": arm,
            "operators": names,
            "fingerprint": task_fingerprint.source_sha256,
        }
        proposal_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
        proposals.append(
            Proposal(
                arm=arm,
                rank=len(proposals) + 1,
                operators=names,
                score=round(score, 12),
                predicted_benefit=round(benefit, 12),
                correctness_risk=round(risk, 12),
                rationale=reasons,
                proposal_id=proposal_id,
            )
        )
        if len(proposals) >= limit:
            break
    return proposals


def failure_update(failure_class: str) -> dict[str, tuple[str, ...]]:
    taxonomy = {
        "invalid_output": ("risk_aware_staging", "bounded_exact_refinement", "early_certificate_exit"),
        "quality_failure": ("bounded_exact_refinement", "risk_aware_staging", "active_set_decomposition"),
        "speed_floor_failure": ("zero_copy_representation", "closed_form_reduction", "structure_aware_initialization"),
        "harmonic_speed_failure": ("vectorized_batch_kernel", "sort_partition_reduction", "sparse_frontier_search"),
        "numerical_instability": ("mixed_precision_with_local_refinement", "risk_aware_staging", "symmetric_decomposition"),
        "memory_failure": ("zero_copy_representation", "contiguous_layout", "sparse_frontier_search"),
        "exception": ("risk_aware_staging", "bounded_exact_refinement"),
    }
    promoted = taxonomy.get(failure_class, ())
    return {"failure_class": (failure_class,), "promoted_operators": tuple(promoted)}


def serialise_fingerprint(value: Fingerprint) -> str:
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))


def serialise_proposals(values: Iterable[Proposal]) -> str:
    return json.dumps([asdict(value) for value in values], sort_keys=True, indent=2)
