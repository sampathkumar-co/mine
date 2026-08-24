# LEXIGEN V7 — GSO Real Causal-Transfer Pilot R1

## Mission
Test whether the algorithmic abstractions induced and sealed before any real holdout identity was known can causally improve autonomous performance optimization on fresh, repository-scale GSO tasks.

The learned library is inherited unchanged from `lexigen/v7-real-library-result-r1`. The earlier AlgoTune selection attempts are preserved only as denominator-feasibility incidents; no AlgoTune task source or payload was opened in those attempts.

## Frozen benchmark
- GSO harness repository: `gso-bench/gso`
- pinned harness commit: `7074865b48123b30a2e61d7dbc4887fcd990e681`
- pinned harness tree: `ce3b95c80bcf2e1dfbe87311671f7d1b8b4cc3b0`
- Hugging Face dataset: `gso-bench/gso`
- pinned dataset revision: `c2e4f1a58427cccd15e0e542f136bd204fb19284`
- test split row count: 102 instances
- pinned parquet path: `data/test-00000-of-00001.parquet`
- pinned parquet SHA-256: `bda458a7b5437c252f6cefdbc896f5f2868de51479e7a221ceda3f3ab74879bc`
- pinned Xet hash: `59cb48b81dd9033151c35123d78dea99ad1959ed33f737c8523f916643402373`
- selector dependency: `duckdb==1.4.1`
- runner target: Ubuntu 24.04 / Python 3.12

GSO is used because it provides real repository-scale optimization tasks, correctness/performance evaluation against expert optimization targets, prebuilt task environments, and a hack-detection layer.

## Frozen V7 learned library
The library is sealed by `lexigen-v7-real/LIBRARY_R1_RESULT.json`:
- `V7M-001`: `EXECUTE -> CERTIFY`
- `V7M-002`: `EXECUTE -> CERTIFY -> RECOVER`
- `V7M-003`: `CERTIFY -> RECOVER`
- `V7M-004`: `REDUCE -> EXECUTE`

The corresponding equal-size random library is also sealed. Neither library may change after GSO selection.

## Selection privacy boundary
Before selection, V7 may inspect only safe GSO metadata columns:
- `instance_id`
- `repo`
- `base_commit`
- `api`
- `created_at`
- `arch`
- `instance_image_tag`

The following values are forbidden before their protocol stage:
- `opt_commit`
- `gt_commit_message`
- `gt_diff`
- `hints_text`
- `prob_script`
- `tests`
- `setup_commands`
- `install_commands`
- expert patch contents
- prior model trajectories/submissions
- leaderboard per-instance solutions

Selection uses DuckDB remote-Parquet projection pushdown. The committed SQL names only the seven safe columns. DuckDB performs Parquet column projection and HTTP partial/range reads, so forbidden column values are not requested by the selection query. Dataset identity is checked through pinned revision/path metadata rather than downloading the full Parquet merely to hash it.

## Holdout selection
After protocol, selector, learned/random libraries, comparison arms, budgets, and gates are locked:
- deterministically select 6 instances,
- require 6 distinct repositories,
- maximum 1 instance per repository,
- selection seed `LEXIGEN-V7-GSO-REAL-TRANSFER-2026-08-24-A`,
- selected instances cannot be replaced because they are hard or inconvenient,
- commit selected instance IDs, repository names, base commits, safe API metadata, and inventory hashes before opening performance-test specifications.

## Comparison arms
Every selected instance receives equal model/evaluator/wall-clock budgets:
1. `v7_full`: frozen generic search plus the four learned V7 macros.
2. `v7_no_library`: identical search and primitive mechanism vocabulary, but learned macros unavailable.
3. `v7_random_library`: identical search with the frozen equal-size random macro library instead.

No arm may inspect the expert optimization commit/diff or GSO hints before its final patch is frozen.

## Search boundary
For each selected GSO instance:
1. commit `TASK_START.json` before opening the task performance specification;
2. open the selected base repository and selected instance's GSO performance-test specification only;
3. generate bounded proposals under each frozen arm;
4. run correctness/performance feedback under equal budgets;
5. freeze one patch per arm;
6. run the authoritative GSO evaluator;
7. run the GSO hack detector / equivalent frozen anti-gaming audit;
8. only after all arm evidence is sealed may the expert optimization diff be opened for post-hoc novelty/confound analysis.

## Per-task clean success
A `clean_gso_win` requires:
- all required correctness tests pass,
- GSO's native performance success criterion is met against the frozen task/expert target,
- the patch is not rejected by the frozen hack-detection audit,
- no invalid-output retry or post-evaluation patch revision is used.

## Causal transfer credit
A task earns `causal_transfer_win=true` only if:
1. `v7_full` earns a clean GSO win;
2. its selected patch provenance uses at least one learned V7 macro;
3. full and no-library patches are semantically/mechanistically non-equivalent;
4. no-library fails the native GSO success gate, or full exceeds no-library by a preregistered material performance margin;
5. random-library does not reproduce the same qualifying learned mechanism advantage;
6. exact learned-macro removal or semantic-equivalence-class ablation removes the qualifying advantage;
7. equal compute/evaluator budgets are preserved.

## Pilot success gate
All are required:
- at least 3/6 clean GSO wins for `v7_full`,
- wins span at least 3 repositories,
- at least 2 causal-transfer wins,
- causal wins span at least 2 repositories,
- at least 2 distinct learned macro IDs receive causal credit,
- `v7_full` beats `v7_no_library` by at least 2 task wins,
- `v7_full` beats `v7_random_library` by at least 2 task wins,
- median task-specific human solver contribution is zero,
- zero selected-instance swaps/drops after task specification access,
- zero post-result threshold changes.

Failure of any required condition fails the V7 GSO pilot.

## Claim boundary
Passing this pilot would be strong evidence that V7's learned abstraction library transfers beyond the AlgoTune environment into real repository-scale performance engineering. It would justify a larger V7 campaign. It is not automatically a world-level AI breakthrough; that additionally requires a larger denominator, independent reproduction, and a genuinely new external frontier result rather than merely matching hidden expert optimizations.
