from __future__ import annotations

import ast
import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ENGINE_VERSION = "lexigen-v5.0.0-causal-transfer-frozen"
PROPOSAL_LIMIT = 6


@dataclass(frozen=True)
class Fingerprint:
    source_sha256: str
    verifier_sha256: str
    features: tuple[str, ...]
    dependency_calls: tuple[str, ...]
    ast_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Operator:
    name: str
    requires_any: tuple[str, ...]
    benefit: float
    risk: float
    group: str


@dataclass(frozen=True)
class Proposal:
    arm: str
    rank: int
    operators: tuple[str, ...]
    score: float
    transfer_ids: tuple[str, ...]
    learned_template: str | None
    rationale: tuple[str, ...]
    proposal_id: str


OPERATORS: tuple[Operator, ...] = (
    Operator("zero_copy_representation", ("array", "matrix", "bytes"), 0.35, 0.05, "representation"),
    Operator("contiguous_layout", ("array", "matrix", "tensor"), 0.30, 0.04, "representation"),
    Operator("dtype_specialization", ("numeric", "array", "matrix"), 0.52, 0.36, "precision"),
    Operator("native_backend_substitution", ("numeric", "array", "bytes", "backend"), 0.48, 0.08, "backend"),
    Operator("vectorized_batch_kernel", ("numeric", "array", "batch"), 0.62, 0.08, "backend"),
    Operator("risk_aware_staging", ("tolerance", "approximate_verifier", "certificate"), 0.45, -0.12, "control"),
    Operator("early_certificate_exit", ("certificate", "verifier", "iterative"), 0.38, -0.08, "control"),
    Operator("active_set_decomposition", ("constraints", "projection", "threshold", "convex"), 0.90, 0.19, "decomposition"),
    Operator("bit_parallel_representation", ("graph", "set", "boolean", "discrete"), 0.84, 0.13, "representation"),
    Operator("sparse_frontier_search", ("graph", "sparse", "set", "discrete"), 0.68, 0.11, "search"),
    Operator("reduced_representation", ("matrix", "tensor", "decomposition", "high_dimensional"), 0.74, 0.15, "decomposition"),
    Operator("bounded_exact_refinement", ("tolerance", "certificate", "decomposition", "iterative"), 0.72, 0.12, "refinement"),
    Operator("sort_partition_reduction", ("order_statistic", "threshold", "topk"), 0.68, 0.10, "decomposition"),
)

OP_BY_NAME = {op.name: op for op in OPERATORS}

LEARNED_SIGNATURES: dict[str, tuple[str, ...]] = {
    "TM-BFR-01": ("bit_parallel_representation", "sparse_frontier_search", "early_certificate_exit"),
    "TM-CAC-01": ("active_set_decomposition", "early_certificate_exit", "risk_aware_staging"),
    "TM-RRR-01": ("reduced_representation", "bounded_exact_refinement", "risk_aware_staging"),
    "TM-PBEB-01": ("dtype_specialization", "native_backend_substitution", "risk_aware_staging"),
}

STATIC_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("zero_copy_representation", "vectorized_batch_kernel"),
    ("contiguous_layout", "vectorized_batch_kernel"),
    ("dtype_specialization", "risk_aware_staging"),
    ("active_set_decomposition", "early_certificate_exit"),
    ("bit_parallel_representation", "sparse_frontier_search"),
    ("reduced_representation", "bounded_exact_refinement"),
)

V4_COMPATIBLE_ALLOWED = {
    "zero_copy_representation",
    "contiguous_layout",
    "dtype_specialization",
    "vectorized_batch_kernel",
    "risk_aware_staging",
    "early_certificate_exit",
    "active_set_decomposition",
    "bit_parallel_representation",
    "sparse_frontier_search",
    "bounded_exact_refinement",
    "sort_partition_reduction",
}


class Visitor(ast.NodeVisitor):
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
        if parts:
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
            if parts:
                self.calls.add(".".join(reversed(parts)).lower())
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self.strings.append(node.value.lower())
        elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            self.numbers.append(float(node.value))
        self.generic_visit(node)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tokens(v: Visitor) -> set[str]:
    result = set(v.names) | set(v.calls)
    for value in v.strings:
        for token in value.replace("-", "_").replace(" ", "_").split("_"):
            if token:
                result.add(token)
    return result


def _derive_features(tokens: set[str], counts: dict[str, int], numbers: list[float], calls: set[str]) -> set[str]:
    joined = " ".join(sorted(tokens))
    rules = {
        "bytes": ("bytes", "encode", "decode", "digest", "hash"),
        "array": ("numpy", "np.asarray", "ndarray", "array", "tensor"),
        "matrix": ("matrix", "matmul", "dot", "linalg", "svd", "eig", "eigh", "qr"),
        "tensor": ("tensor", "unfold", "mode"),
        "decomposition": ("svd", "eig", "eigh", "qr", "cholesky", "decomposition"),
        "graph": ("graph", "edge", "vertex", "node", "adjacency", "neighbor"),
        "set": ("set", "subset", "union", "intersection", "cover"),
        "boolean": ("bool", "bit", "mask", "true", "false"),
        "sparse": ("sparse", "csr", "csc", "frontier", "neighbor"),
        "constraints": ("constraint", "feasible", "dual", "bound", "lambda"),
        "projection": ("projection", "project", "simplex", "cvar"),
        "convex": ("convex", "cvx", "projection", "dual"),
        "threshold": ("threshold", "cutoff", "tol", "tolerance"),
        "order_statistic": ("sort", "partition", "quantile", "median", "topk"),
        "topk": ("topk", "argpartition", "largest", "smallest"),
        "batch": ("batch", "broadcast", "axis", "vectorize"),
        "certificate": ("is_solution", "verify", "valid", "certificate", "allclose"),
        "verifier": ("is_solution", "verify", "allclose", "compare_digest"),
        "backend": ("scipy", "numpy", "torch", "jax", "cvxpy", "networkx"),
        "numeric": ("float", "numpy", "scipy", "math", "int"),
        "discrete": ("graph", "integer", "combin", "set", "subset", "bool", "bit"),
        "iterative": ("while", "iteration", "iter", "converge", "solve_ivp", "integrate"),
        "scientific_dynamics": ("solve_ivp", "ode", "differential", "derivative", "integrate"),
        "high_dimensional": ("tensor", "matrix", "reshape", "stack", "flatten"),
    }
    features: set[str] = set()
    for feature, needles in rules.items():
        if any(needle in joined for needle in needles):
            features.add(feature)
    if any(0.0 < abs(value) < 0.1 for value in numbers) or any(abs(value - round(value)) > 0 for value in numbers):
        features.add("tolerance")
    if "certificate" in features and "tolerance" in features:
        features.add("approximate_verifier")
    if counts.get("For", 0) + counts.get("While", 0) > 0:
        features.add("iterative")
    if any("solve_ivp" in call or "integrate" in call for call in calls):
        features.update(("iterative", "scientific_dynamics"))
    return features


def fingerprint(task_source: str, verifier_source: str = "") -> Fingerprint:
    visitor = Visitor()
    visitor.visit(ast.parse(task_source))
    verifier = Visitor()
    verifier.visit(ast.parse(verifier_source) if verifier_source.strip() else ast.parse(""))
    counts = dict(visitor.counts)
    for key, value in verifier.counts.items():
        counts[key] = counts.get(key, 0) + value
    tokens = _tokens(visitor) | _tokens(verifier)
    calls = visitor.calls | verifier.calls
    features = _derive_features(tokens, counts, visitor.numbers + verifier.numbers, calls)
    return Fingerprint(
        source_sha256=_sha(task_source),
        verifier_sha256=_sha(verifier_source),
        features=tuple(sorted(features)),
        dependency_calls=tuple(sorted(calls)),
        ast_counts=tuple(sorted(counts.items())),
    )


def _operator_compatible(op: Operator, features: set[str]) -> bool:
    return bool(features.intersection(op.requires_any))


def _composition_score(names: tuple[str, ...], features: set[str]) -> float:
    ops = [OP_BY_NAME[name] for name in names]
    compatible = sum(1 for op in ops if _operator_compatible(op, features))
    groups = len({op.group for op in ops})
    return round(sum(op.benefit - max(op.risk, 0.0) * 0.35 for op in ops) + 0.22 * compatible + 0.05 * groups, 6)


def applicable_transfer_templates(fp: Fingerprint) -> list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    features = set(fp.features)
    result: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []
    if features.intersection({"graph", "set", "boolean", "discrete", "sparse"}):
        result.append(("bit_frontier_restriction", "TM-BFR-01", LEARNED_SIGNATURES["TM-BFR-01"], ("discrete structure matches prior word/frontier recipe",)))
    if features.intersection({"constraints", "projection", "threshold", "convex", "certificate"}):
        result.append(("certified_active_core", "TM-CAC-01", LEARNED_SIGNATURES["TM-CAC-01"], ("constraint/certificate structure matches prior active-core recipe",)))
    if len(features.intersection({"matrix", "tensor", "decomposition", "array", "high_dimensional", "tolerance"})) >= 3:
        result.append(("reduced_representation_refinement", "TM-RRR-01", LEARNED_SIGNATURES["TM-RRR-01"], ("high-dimensional numerical structure matches reduced-representation recipe",)))
    precision_ok = bool(features.intersection({"numeric", "array", "matrix"})) and bool(features.intersection({"tolerance", "approximate_verifier", "certificate"}))
    long_horizon_risk = "scientific_dynamics" in features and "iterative" in features
    if precision_ok and not long_horizon_risk:
        result.append(("precision_backend_error_budget", "TM-PBEB-01", LEARNED_SIGNATURES["TM-PBEB-01"], ("approximate numeric verifier permits frozen precision/backend error-budget recipe",)))
    return result


def _proposal_id(arm: str, operators: tuple[str, ...], transfer_ids: tuple[str, ...], template: str | None) -> str:
    payload = json.dumps({"arm": arm, "operators": operators, "transfer_ids": transfer_ids, "template": template}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _build(arm: str, rows: list[tuple[tuple[str, ...], float, tuple[str, ...], str | None, tuple[str, ...]]]) -> list[Proposal]:
    unique: dict[tuple[tuple[str, ...], tuple[str, ...], str | None], tuple[float, tuple[str, ...]]] = {}
    for operators, score, transfer_ids, template, rationale in rows:
        key = (operators, transfer_ids, template)
        if key not in unique or score > unique[key][0]:
            unique[key] = (score, rationale)
    ordered = sorted(unique.items(), key=lambda item: (-item[1][0], item[0][0], item[0][1], str(item[0][2])))[:PROPOSAL_LIMIT]
    result: list[Proposal] = []
    for rank, (key, value) in enumerate(ordered, 1):
        operators, transfer_ids, template = key
        score, rationale = value
        result.append(Proposal(arm, rank, operators, score, transfer_ids, template, rationale, _proposal_id(arm, operators, transfer_ids, template)))
    return result


def _base_compositions(features: set[str], maximum: int = 3) -> list[tuple[str, ...]]:
    compatible = [op.name for op in OPERATORS if _operator_compatible(op, features)]
    if not compatible:
        compatible = ["risk_aware_staging", "early_certificate_exit"]
    rows: list[tuple[str, ...]] = []
    for size in range(1, min(maximum, len(compatible)) + 1):
        rows.extend(tuple(combo) for combo in itertools.combinations(compatible, size))
    return rows


def generate_proposals(task_source: str, verifier_source: str = "") -> dict[str, object]:
    fp = fingerprint(task_source, verifier_source)
    features = set(fp.features)
    base = _base_compositions(features)
    learned_exact = {tuple(value) for value in LEARNED_SIGNATURES.values()}

    full_rows: list[tuple[tuple[str, ...], float, tuple[str, ...], str | None, tuple[str, ...]]] = []
    for combo in base:
        full_rows.append((combo, _composition_score(combo, features), (), None, ("generic base-operator proposal",)))
    for template, causal_id, signature, rationale in applicable_transfer_templates(fp):
        full_rows.append((signature, _composition_score(signature, features) + 1.25, (causal_id,), template, rationale))

    no_transfer_rows = [
        (combo, _composition_score(combo, features), (), None, ("no-transfer generic base-operator proposal",))
        for combo in base
        if combo not in learned_exact
    ]

    rng = random.Random(int(hashlib.sha256((ENGINE_VERSION + "\0" + fp.source_sha256).encode()).hexdigest(), 16))
    random_pool = base[:]
    rng.shuffle(random_pool)
    random_rows = [(combo, _composition_score(combo, features), (), None, ("deterministic random base-space proposal",)) for combo in random_pool[:PROPOSAL_LIMIT]]

    static_rows = [
        (combo, _composition_score(combo, features), (), None, ("static template committed independently of transfer memory",))
        for combo in STATIC_TEMPLATES
        if all(name in OP_BY_NAME for name in combo)
    ]

    v4_rows = [
        (combo, _composition_score(combo, features), (), None, ("v4-compatible predecessor proposal",))
        for combo in base
        if set(combo).issubset(V4_COMPATIBLE_ALLOWED)
    ]

    arms = {
        "v5_full": _build("v5_full", full_rows),
        "v5_no_transfer": _build("v5_no_transfer", no_transfer_rows),
        "random_search": _build("random_search", random_rows),
        "static_template": _build("static_template", static_rows),
        "v4_compatible": _build("v4_compatible", v4_rows),
    }
    return {
        "engine_version": ENGINE_VERSION,
        "fingerprint": asdict(fp),
        "applicable_transfer_templates": [
            {"template": template, "causal_id": causal_id, "operators": list(signature), "rationale": list(rationale)}
            for template, causal_id, signature, rationale in applicable_transfer_templates(fp)
        ],
        "arms": {arm: [asdict(p) for p in proposals] for arm, proposals in arms.items()},
    }


def verify_transfer_memory(path: Path | None = None) -> dict[str, object]:
    memory_path = path or Path(__file__).with_name("TRANSFER_MEMORY.json")
    data = json.loads(memory_path.read_text(encoding="utf-8"))
    ids = {str(row["causal_id"]) for row in data["learned_templates"].values()}
    if ids != set(LEARNED_SIGNATURES):
        raise RuntimeError(f"transfer-memory causal IDs differ from frozen engine signatures: {ids}")
    if not data.get("frozen_before_v5_holdout_selection"):
        raise RuntimeError("transfer memory is not marked frozen before holdout selection")
    return {"memory_sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(), "causal_ids": sorted(ids)}


if __name__ == "__main__":
    print(json.dumps({"engine_version": ENGINE_VERSION, "transfer_memory": verify_transfer_memory()}, indent=2))
