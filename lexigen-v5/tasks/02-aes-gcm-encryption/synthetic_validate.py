from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from candidates import CANDIDATES_BY_ARM, PROVENANCE, Problem, Solution

CASES = (
    ("tiny_aes128_empty_aad", 16, 1, 0, 11),
    ("n1_aes128_aad32", 16, 1024, 32, 23),
    ("n2_aes192_empty_aad", 24, 2048, 0, 37),
    ("n4_aes256_aad32", 32, 4096, 32, 41),
    ("n8_aes128_none_aad", 16, 8192, -1, 53),
    ("n16_aes256_aad32", 32, 16384, 32, 67),
)


def expand(seed: int, length: int, domain: bytes) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(domain + seed.to_bytes(8, "big") + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(out[:length])


def generate_problem(key_size: int, plaintext_size: int, aad_size: int, seed: int) -> Problem:
    key = expand(seed, key_size, b"key")
    nonce = expand(seed, 12, b"nonce")
    plaintext = expand(seed, plaintext_size, b"plaintext")
    associated_data = None if aad_size < 0 else expand(seed, aad_size, b"aad")
    return {"key": key, "nonce": nonce, "plaintext": plaintext, "associated_data": associated_data}


def reference(problem: Problem) -> Solution:
    key = problem["key"]
    nonce = problem["nonce"]
    plaintext = problem["plaintext"]
    aad = problem.get("associated_data")
    if not isinstance(key, bytes) or not isinstance(nonce, bytes) or not isinstance(plaintext, bytes):
        raise TypeError("synthetic reference input is not bytes")
    if aad is not None and not isinstance(aad, bytes):
        raise TypeError("synthetic AAD is not bytes or None")
    combined = AESGCM(key).encrypt(nonce, plaintext, aad)
    return {"ciphertext": combined[:-16], "tag": combined[-16:]}


def verify(problem: Problem, solution: Solution, expected: Solution) -> tuple[bool, str | None]:
    if not isinstance(solution, dict) or set(solution) != {"ciphertext", "tag"}:
        return False, "invalid_solution_shape"
    ciphertext = solution.get("ciphertext")
    tag = solution.get("tag")
    if not isinstance(ciphertext, bytes) or not isinstance(tag, bytes):
        return False, "non_bytes_output"
    plaintext = problem["plaintext"]
    if not isinstance(plaintext, bytes):
        return False, "invalid_problem_plaintext"
    if len(ciphertext) != len(plaintext):
        return False, "ciphertext_length"
    if len(tag) != 16:
        return False, "tag_length"
    if not hmac.compare_digest(expected["ciphertext"], ciphertext):
        return False, "ciphertext_mismatch"
    if not hmac.compare_digest(expected["tag"], tag):
        return False, "tag_mismatch"
    return True, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_arms = {"v5_full", "v5_no_transfer", "random_search", "static_template", "v4_compatible"}
    if set(CANDIDATES_BY_ARM) != expected_arms:
        raise RuntimeError("unexpected comparison arms")
    flat = [(arm, name, fn) for arm, rows in CANDIDATES_BY_ARM.items() for name, fn in rows]
    if len(flat) != 30:
        raise RuntimeError(f"expected 30 frozen candidates, got {len(flat)}")

    results: list[dict[str, object]] = []
    for case_name, key_size, plaintext_size, aad_size, seed in CASES:
        problem = generate_problem(key_size, plaintext_size, aad_size, seed)
        expected = reference(problem)
        for arm, name, fn in flat:
            try:
                proposed = fn(problem)
                valid, reason = verify(problem, proposed, expected)
                error = None
            except Exception as exc:
                valid, reason = False, "exception"
                error = f"{type(exc).__name__}: {exc}"
            provenance = next(row for row in PROVENANCE[arm] if row["candidate"] == name)
            row = {
                "case": case_name,
                "key_size": key_size,
                "plaintext_size": plaintext_size,
                "aad_size": None if aad_size < 0 else aad_size,
                "seed": seed,
                "arm": arm,
                "candidate": name,
                "valid": valid,
                "failure_reason": reason,
                "exception": error,
                "implementation_class": provenance["implementation_class"],
                "transfer_ids": provenance["transfer_ids"],
                "learned_template": provenance["learned_template"],
                "semantic_signature": provenance["semantic_signature"],
            }
            results.append(row)
            print(f"{case_name} {arm}/{name} valid={valid} reason={reason}", flush=True)

    eligibility: dict[str, list[str]] = {}
    candidate_summaries: list[dict[str, object]] = []
    for arm, rows in CANDIDATES_BY_ARM.items():
        eligibility[arm] = []
        for name, _ in rows:
            selected = [r for r in results if r["arm"] == arm and r["candidate"] == name]
            passed = sum(1 for r in selected if r["valid"])
            eligible = passed == len(CASES)
            if eligible:
                eligibility[arm].append(name)
            provenance = next(row for row in PROVENANCE[arm] if row["candidate"] == name)
            candidate_summaries.append({
                "arm": arm,
                "candidate": name,
                "checks_passed": passed,
                "checks_total": len(CASES),
                "eligible_for_official_training": eligible,
                "implementation_class": provenance["implementation_class"],
                "transfer_ids": provenance["transfer_ids"],
                "learned_template": provenance["learned_template"],
            })

    v5_has_candidate = bool(eligibility["v5_full"])
    every_arm_has_candidate = all(bool(eligibility[arm]) for arm in eligibility)
    report = {
        "campaign": "LEXIGEN v5 Causal Transfer Generalization Experiment",
        "task_index": 2,
        "task": "aes_gcm_encryption",
        "stage": "synthetic_correctness_r1",
        "synthetic_cases": len(CASES),
        "candidate_count": len(flat),
        "checks": len(results),
        "passed_checks": sum(1 for row in results if row["valid"]),
        "failed_checks": sum(1 for row in results if not row["valid"]),
        "candidate_summaries": candidate_summaries,
        "eligible_candidates_by_arm": eligibility,
        "v5_full_has_training_eligible_candidate": v5_has_candidate,
        "every_arm_has_training_eligible_candidate": every_arm_has_candidate,
        "task_may_proceed_to_official_training": v5_has_candidate,
        "candidate_repair_permitted_after_synthetic": false,
        "official_training_manifest_opened": false,
        "official_training_payloads_opened": 0,
        "official_test_manifest_opened": false,
        "official_test_payloads_opened": 0
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "synthetic-summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "synthetic-results.jsonl").write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in results) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_candidates_by_arm": eligibility, "failed_checks": report["failed_checks"], "task_may_proceed_to_official_training": v5_has_candidate}, indent=2), flush=True)
    if not v5_has_candidate:
        raise SystemExit("Task 2 has no v5_full candidate surviving the frozen synthetic gate")


if __name__ == "__main__":
    main()
