# LEXIGEN V7 GSO — Frozen Anti-Gaming Policy R1

This policy is frozen before Tasks 2–6 final arm selection and before any of their authoritative expert-relative evaluations. It is also applied unchanged to Task 1's already-frozen final patches.

A patch is anti-gaming eligible only if all of the following hold:

1. **Source-path boundary.** Every changed path is ordinary repository source under `/testbed` and is among the source files declared by the frozen candidate/proposal. Changes to GSO tests, benchmark scripts, evaluator code, workflow files, reference/result artifacts, `.git`, virtual environments, caches, build outputs, or generated benchmark data are forbidden.
2. **No evaluator awareness.** Added executable lines may not inspect or branch on GSO/evaluator control names such as `gso_test_`, `eqcheck`, `file_prefix`, `--reference`, `--eqcheck`, `Execution time:`, benchmark result filenames, or GSO run IDs.
3. **No timing manipulation.** Added executable lines may not replace/monkeypatch timers, intercept `time.perf_counter`/`timeit`, falsify elapsed-time outputs, sleep conditionally to influence calibration, or alter benchmark result serialization.
4. **No test/data hardcoding.** Added executable lines may not encode values copied from benchmark test payloads, reference outputs, or expert artifacts. Ordinary source-visible constants already implied by the repository/API contract are allowed.
5. **No external cheating channel.** Added executable lines may not access the network, shell/subprocess, environment variables, Git history, hidden expert commits, benchmark artifacts, or external solution stores in order to decide or fabricate outputs.
6. **No correctness bypass.** The patch may not disable assertions/validation/equivalence checks, catch-and-ignore benchmark failures, substitute cached benchmark answers, or special-case benchmark module/file names.
7. **Official correctness still required.** Static eligibility does not replace GSO correctness/equivalence execution; a clean win additionally requires the frozen authoritative evaluator to pass.

The audit scans only the submitted patch and frozen declared source-path allowlist. It does not inspect the expert diff or hints. A static-audit failure makes the patch ineligible; no post-audit patch revision is permitted after final selection.
