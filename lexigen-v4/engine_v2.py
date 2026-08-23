from __future__ import annotations

import ast
import hashlib
import itertools
import json
import random
from dataclasses import asdict
from typing import Iterable, Sequence

from engine import (
    Fingerprint,
    OPERATORS,
    Proposal,
    TRANSFER_MEMORY,
    V3_ALLOWED,
    _Visitor,
    _compose,
    _operator_score,
    _sha,
    failure_update,
)

ENGINE_VERSION = "lexigen-v4.0.1-prelock"


def _lexical_atoms(visitor: _Visitor) -> set[str]:
    atoms: set[str] = set()
    values = set(visitor.names) | set(visitor.calls)
    for value in values:
        lowered = value.lower()
        atoms.add(lowered)
        for dotted in lowered.split("."):
            atoms.add(dotted)
            atoms.update(part for part in dotted.replace("-", "_").split("_") if part)
    for value in visitor.strings:
        lowered = value.lower()
        atoms.add(lowered)
        atoms.update(part for part in re_split(lowered) if part)
    return atoms


def re_split(value: str) -> list[str]:
    current = value.replace("-", "_").replace("/", "_").replace(".", "_")
    return current.split("_")


def _derive_features(atoms: set[str], counts: dict[str, int], numbers: Sequence[float]) -> set[str]:
    features: set[str] = set()
    rules = {
        "bytes": {"bytes", "bytearray", "plaintext", "payload", "digest", "encode", "decode"},
        "array": {"numpy", "np", "asarray", "ndarray", "array", "tensor"},
        "matrix": {"svd", "eigh", "eig", "matrix", "matmul", "dot", "transpose", "linalg", "cholesky", "qr"},
        "decomposition": {"svd", "eigh", "eig", "qr", "cholesky", "polar"},
        "graph": {"graph", "edge", "edges", "vertex", "vertices", "node", "nodes", "adjacency", "neighbor", "neighbors"},
        "set": {"set", "subset", "cover", "union", "intersection"},
        "bit": {"bit", "bits", "xor", "mask", "bitmask"},
        "boolean": {"bool", "boolean", "true", "false"},
        "cluster": {"cluster", "clusters", "kmeans", "centroid", "centroids", "labels"},
        "projection": {"project", "projection", "simplex", "cvar"},
        "constraints": {"constraint", "constraints", "feasible", "bound", "bounds", "dual", "lambda"},
        "convex": {"convex", "cvx", "cvxpy", "projection", "dual"},
        "statistics": {"mean", "median", "variance", "quantile", "probability", "entropy"},
        "order_statistic": {"sort", "sorted", "partition", "argpartition", "quantile", "median", "topk"},
        "threshold": {"threshold", "cutoff", "tolerance", "tol", "atol", "rtol"},
        "topk": {"topk", "argpartition", "largest", "smallest"},
        "sequence": {"sequence", "string", "prefix", "suffix", "substring", "alignment"},
        "recurrence": {"dynamic", "memo", "memoize", "recurrence", "dp"},
        "sparse": {"sparse", "csr", "csc", "adjacency"},
        "batch": {"batch", "axis", "vectorize", "vectorized", "broadcast"},
        "iterative": {"max_iter", "iteration", "iterations", "converge", "convergence"},
        "certificate": {"is_solution", "valid", "validate", "verify", "certificate"},
        "verifier": {"is_solution", "validate", "verify", "allclose", "compare_digest"},
        "crypto": {"cryptography", "cipher", "encrypt", "decrypt", "digest"},
        "hash": {"sha", "sha256", "sha512", "hash", "hashlib", "digest"},
        "encoding": {"encode", "decode", "base64", "codec", "encoding"},
        "linear": {"linear", "matrix", "dot", "matmul", "linalg"},
        "formula": {"formula", "closed_form", "analytic", "analytical"},
        "grouped_generator": {"reshape", "repeat", "stack", "vstack", "hstack", "block", "blocks", "group", "groups"},
        "block_structure": {"reshape", "block", "blocks", "stack", "vstack", "chunk", "chunks"},
        "symmetric": {"symmetric", "eigh", "hermitian"},
        "numeric": {"numpy", "np", "float", "float32", "float64", "int", "integer", "scipy", "math"},
        "discrete": {"graph", "integer", "combinatorial", "set", "sequence", "permutation"},
    }
    for feature, needles in rules.items():
        if atoms.intersection(needles):
            features.add(feature)

    if any(abs(value - round(value)) > 0.0 for value in numbers):
        features.add("tolerance")
    if any(0.0 < abs(value) < 0.1 for value in numbers):
        features.add("tolerance")
    if "compare_digest" in atoms:
        features.add("bit_exact")
    if counts.get("For", 0) + counts.get("While", 0) > 0:
        features.add("iterative")
    if counts.get("MatMult", 0) > 0:
        features.update(("matrix", "linear"))
    if counts.get("Compare", 0) > 0 or "is_solution" in atoms:
        features.add("verifier")
    if "verifier" in features and "tolerance" in features:
        features.add("approximate_verifier")
    if "verifier" in features:
        features.add("certificate")
    return features


def fingerprint(task_source: str, verifier_source: str = "") -> Fingerprint:
    task_tree = ast.parse(task_source)
    verifier_tree = ast.parse(verifier_source) if verifier_source.strip() else ast.parse("")
    visitor = _Visitor()
    visitor.visit(task_tree)
    verifier_visitor = _Visitor()
    verifier_visitor.visit(verifier_tree)
    atoms = _lexical_atoms(visitor) | _lexical_atoms(verifier_visitor)
    counts = dict(visitor.counts)
    for key, value in verifier_visitor.counts.items():
        counts[key] = counts.get(key, 0) + value
    numbers = sorted(set(visitor.numbers + verifier_visitor.numbers))
    features = _derive_features(atoms, counts, numbers)

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


def _count(known: set[str], values: Iterable[str]) -> int:
    return len(known.intersection(values))


def _eligible(operator_name: str, known: set[str]) -> bool:
    operator = next(operator for operator in OPERATORS if operator.name == operator_name)
    if known.intersection(operator.avoids):
        return False

    if operator_name == "zero_copy_representation":
        return bool(known.intersection({"bytes", "array", "matrix"}))
    if operator_name == "contiguous_layout":
        return bool(known.intersection({"array", "matrix"}))
    if operator_name == "dtype_specialization":
        return "numeric" in known and bool(known.intersection({"array", "matrix"})) and "bit_exact" not in known
    if operator_name == "mixed_precision_with_local_refinement":
        return {"matrix", "decomposition", "tolerance"}.issubset(known)
    if operator_name == "structure_aware_initialization":
        return _count(known, {"grouped_generator", "cluster", "block_structure"}) >= 2
    if operator_name == "bounded_exact_refinement":
        return "warm_start" in known and bool(known.intersection({"iterative", "approximate_verifier", "verifier"}))
    if operator_name == "active_set_decomposition":
        return _count(known, {"constraints", "projection", "convex"}) >= 2
    if operator_name == "symmetric_decomposition":
        return "matrix" in known and bool(known.intersection({"symmetric", "decomposition"}))
    if operator_name == "sort_partition_reduction":
        return bool(known.intersection({"order_statistic", "topk"}))
    if operator_name == "bit_parallel_representation":
        return _count(known, {"boolean", "bit", "set", "graph"}) >= 2
    if operator_name == "sparse_frontier_search":
        return "graph" in known and bool(known.intersection({"sparse", "discrete"}))
    if operator_name == "dynamic_programming_state_compression":
        return "recurrence" in known and bool(known.intersection({"sequence", "discrete"}))
    if operator_name == "native_one_shot_backend":
        return "bytes" in known and _count(known, {"encoding", "crypto", "hash"}) >= 1
    if operator_name == "vectorized_batch_kernel":
        return bool(known.intersection({"array", "matrix"})) and bool(known.intersection({"numeric", "batch"}))
    if operator_name == "closed_form_reduction":
        return _count(known, {"formula", "projection", "statistics", "linear", "iterative"}) >= 2
    if operator_name == "risk_aware_staging":
        return "tolerance" in known and bool(known.intersection({"decomposition", "iterative", "approximate_verifier"}))
    if operator_name == "early_certificate_exit":
        return "iterative" in known and bool(known.intersection({"certificate", "verifier"}))
    return False


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
    pool = [operator for operator in OPERATORS if _eligible(operator.name, features)]
    if arm == "v3_compatible":
        pool = [operator for operator in pool if operator.name in V3_ALLOWED]
    use_transfer = arm == "v4_full"
    lengths = (1,) if arm in {"template_synthesis", "v3_compatible"} else (1, 2, 3)

    candidates: list[tuple[tuple[object, ...], float, float, tuple[str, ...]]] = []
    for length in lengths:
        for composition in itertools.combinations(pool, length):
            known = set(features)
            compatible = True
            score = 0.0
            risk = 0.0
            reasons: list[str] = []
            for operator in composition:
                if not _eligible(operator.name, known):
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


def serialise_fingerprint(value: Fingerprint) -> str:
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))


def serialise_proposals(values: Iterable[Proposal]) -> str:
    return json.dumps([asdict(value) for value in values], sort_keys=True, indent=2)


__all__ = [
    "ENGINE_VERSION",
    "Fingerprint",
    "Proposal",
    "TRANSFER_MEMORY",
    "failure_update",
    "fingerprint",
    "generate_proposals",
    "serialise_fingerprint",
    "serialise_proposals",
]
