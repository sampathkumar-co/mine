from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict
from typing import Iterable

import engine_v2 as base
from engine import Fingerprint, Proposal, _Visitor, _sha, failure_update

ENGINE_VERSION = "lexigen-v4.0.2-prelock"


def _definition_atoms(tree: ast.AST) -> set[str]:
    atoms: set[str] = set()

    def add(value: str) -> None:
        lowered = value.lower()
        atoms.add(lowered)
        for dotted in lowered.split("."):
            atoms.add(dotted)
            atoms.update(part for part in dotted.replace("-", "_").split("_") if part)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, ast.alias):
            add(node.name)
            if node.asname:
                add(node.asname)
    return atoms


def fingerprint(task_source: str, verifier_source: str = "") -> Fingerprint:
    task_tree = ast.parse(task_source)
    verifier_tree = ast.parse(verifier_source) if verifier_source.strip() else ast.parse("")
    visitor = _Visitor()
    visitor.visit(task_tree)
    verifier_visitor = _Visitor()
    verifier_visitor.visit(verifier_tree)

    atoms = (
        base._lexical_atoms(visitor)
        | base._lexical_atoms(verifier_visitor)
        | _definition_atoms(task_tree)
        | _definition_atoms(verifier_tree)
    )
    counts = dict(visitor.counts)
    for key, value in verifier_visitor.counts.items():
        counts[key] = counts.get(key, 0) + value
    numbers = sorted(set(visitor.numbers + verifier_visitor.numbers))
    features = base._derive_features(atoms, counts, numbers)

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


def generate_proposals(
    task_fingerprint: Fingerprint,
    arm: str = "v4_full",
    limit: int = 6,
    random_seed: str = "LEXIGEN-V4",
) -> list[Proposal]:
    raw = base.generate_proposals(task_fingerprint, arm=arm, limit=limit, random_seed=random_seed)
    proposals: list[Proposal] = []
    for value in raw:
        payload = {
            "engine": ENGINE_VERSION,
            "arm": arm,
            "operators": value.operators,
            "fingerprint": task_fingerprint.source_sha256,
        }
        proposals.append(
            Proposal(
                arm=value.arm,
                rank=value.rank,
                operators=value.operators,
                score=value.score,
                predicted_benefit=value.predicted_benefit,
                correctness_risk=value.correctness_risk,
                rationale=value.rationale,
                proposal_id=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20],
            )
        )
    return proposals


def serialise_fingerprint(value: Fingerprint) -> str:
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))


def serialise_proposals(values: Iterable[Proposal]) -> str:
    return json.dumps([asdict(value) for value in values], sort_keys=True, indent=2)


__all__ = [
    "ENGINE_VERSION",
    "Fingerprint",
    "Proposal",
    "failure_update",
    "fingerprint",
    "generate_proposals",
    "serialise_fingerprint",
    "serialise_proposals",
]
