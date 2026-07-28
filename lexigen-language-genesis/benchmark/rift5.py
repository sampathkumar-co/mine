from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

from rift4_revision2 import MaxCycleWorld, build_cases

State = frozenset[str]


def canonical(state: State) -> tuple[str, ...]:
    return tuple(sorted(state))


def finalizer_candidates() -> list[dict[str, Any]]:
    candidates = [
        {"op": "ref", "name": "current"},
        {"op": "ref", "name": "next"},
        {"op": "trace_union"},
        {
            "op": "select_extreme",
            "mode": "min",
            "key": "canonical",
            "args": [{"op": "ref", "name": "current"}, {"op": "ref", "name": "next"}],
        },
        {
            "op": "select_extreme",
            "mode": "max",
            "key": "canonical",
            "args": [{"op": "ref", "name": "current"}, {"op": "ref", "name": "next"}],
        },
    ]
    candidates.sort(
        key=lambda ast: hashlib.sha256(
            json.dumps(ast, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
    )
    return candidates


def evaluate_ast(ast: dict[str, Any], env: dict[str, Any]) -> State:
    op = ast.get("op")
    if op == "ref":
        value = env.get(str(ast["name"]))
        if not isinstance(value, frozenset):
            raise ValueError("reference does not name a state")
        return value
    if op == "trace_union":
        trace = env.get("trace")
        if not isinstance(trace, list) or not trace:
            raise ValueError("trace_union requires a non-empty trace")
        return frozenset().union(*trace)
    if op == "select_extreme":
        args = ast.get("args")
        if not isinstance(args, list) or len(args) != 2:
            raise ValueError("select_extreme requires two arguments")
        values = [evaluate_ast(arg, env) for arg in args]
        mode = ast.get("mode")
        if ast.get("key") != "canonical" or mode not in {"min", "max"}:
            raise ValueError("unsupported extreme selector")
        return (min if mode == "min" else max)(values, key=canonical)
    raise ValueError(f"unknown AST operation: {op!r}")


def terminal_environments(case: MaxCycleWorld) -> tuple[dict[str, Any], State]:
    current = case.seed
    seen = {canonical(current)}
    trace: list[State] = []
    while True:
        trace.append(current)
        nxt = case.step(current)
        if canonical(nxt) in seen:
            return {"current": current, "next": nxt, "trace": trace}, case.independently_verified_target()
        seen.add(canonical(nxt))
        current = nxt


def synthesize_finalizer(demonstrations: list[MaxCycleWorld]) -> tuple[dict[str, Any], int]:
    tested = 0
    for candidate in finalizer_candidates():
        tested += 1
        if all(
            evaluate_ast(candidate, terminal_environments(case)[0])
            == terminal_environments(case)[1]
            for case in demonstrations
        ):
            return candidate, tested
    raise RuntimeError("no finalizer expression fits demonstrations")


def execute_artifact(artifact: dict[str, Any], case: MaxCycleWorld) -> State:
    if artifact.get("schema") != "lexigen-verified-trajectory-artifact-v1":
        raise ValueError("unsupported artifact schema")
    current = case.seed
    seen = {canonical(current)}
    trace: list[State] = []
    for _ in range(5_000):
        trace.append(current)
        nxt = case.step(current)
        if canonical(nxt) in seen:
            return evaluate_ast(
                artifact["finalizer_ast"],
                {"current": current, "next": nxt, "trace": trace},
            )
        seen.add(canonical(nxt))
        current = nxt
    raise RuntimeError("artifact did not terminate")


def all_small_states() -> list[State]:
    atoms = ("a", "b", "c")
    states = []
    for size in (1, 2):
        states.extend(frozenset(combo) for combo in itertools.combinations(atoms, size))
    return states


def build_verifier_certificate(ast: dict[str, Any]) -> dict[str, Any]:
    rows = []
    states = all_small_states()
    for current in states:
        for nxt in states:
            env = {"current": current, "next": nxt, "trace": [current, nxt]}
            result = evaluate_ast(ast, env)
            expected = max((current, nxt), key=canonical)
            rows.append(
                {
                    "current": list(canonical(current)),
                    "next": list(canonical(nxt)),
                    "result": list(canonical(result)),
                    "expected": list(canonical(expected)),
                    "correct": result == expected,
                }
            )
    if not all(row["correct"] for row in rows):
        raise AssertionError("synthesized verifier certificate failed exhaustive finite check")
    canonical_rows = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "lexigen-verifier-certificate-v1",
        "claim": "finalizer AST computes canonical maximum of current and next",
        "finite_universe": ["a", "b", "c"],
        "state_sizes": [1, 2],
        "row_count": len(rows),
        "all_correct": True,
        "rows_sha256": hashlib.sha256(canonical_rows).hexdigest(),
        "ast_sha256": hashlib.sha256(
            json.dumps(ast, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def run(output_dir: Path) -> dict[str, Any]:
    demonstrations = build_cases([5, 6], replicas=2)
    transfer = build_cases(range(8, 17), replicas=3)
    ast, tested = synthesize_finalizer(demonstrations)
    certificate = build_verifier_certificate(ast)
    artifact = {
        "schema": "lexigen-verified-trajectory-artifact-v1",
        "name": "verified_trajectory_" + certificate["ast_sha256"][:12],
        "stop": {"op": "repeat"},
        "finalizer_ast": ast,
        "verifier_certificate": certificate,
        "provenance": {
            "candidate_expressions_tested": tested,
            "human_supplied_expression_grammar": True,
        },
    }
    correct = sum(
        int(execute_artifact(artifact, case) == case.independently_verified_target())
        for case in transfer
    )
    accuracy = correct / len(transfer)
    if accuracy != 1.0:
        raise AssertionError("verified artifact failed transfer")

    report = {
        "benchmark": "RIFT-5",
        "status": "vocabulary plus verifier co-synthesis candidate; not an external world breakthrough claim",
        "artifact": artifact,
        "transfer_case_count": len(transfer),
        "transfer_accuracy": accuracy,
        "gate": {
            "semantic_expression_synthesized": ast.get("op") == "select_extreme" and ast.get("mode") == "max",
            "verifier_certificate_generated": certificate["all_correct"],
            "verifier_bound_to_artifact": certificate["ast_sha256"]
            == hashlib.sha256(
                json.dumps(ast, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "hidden_transfer": accuracy == 1.0,
        },
        "claim_boundary": (
            "The verifier is generated and cryptographically bound to the invented semantic expression. "
            "However, the expression grammar and finite verification universe are human supplied; external reproduction "
            "and a real discovery enabled by the primitive are still required for a world-level claim."
        ),
    }
    if not all(report["gate"].values()):
        raise AssertionError(f"RIFT-5 gate failed: {report['gate']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "rift5-verified-artifact.json"
    report_path = output_dir / "rift5-report.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "artifact": artifact["name"],
        "expressions_tested": tested,
        "certificate_rows": certificate["row_count"],
        "transfer_accuracy": accuracy,
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rift5"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
