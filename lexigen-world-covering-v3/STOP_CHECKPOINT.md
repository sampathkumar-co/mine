# LEXIGEN World Covering Record v3 — Pre-trigger Checkpoint

- Branch created independently from `main`; no v1/v2 trigger or workflow state is inherited.
- Exact deterministic v1 and corrected-v2 target lineages are reproduced and excluded before v3 selection.
- v3 engine, verifier, protocol, workflow, dependency and lock are frozen before snapshot access.
- No v3 snapshot access has occurred.
- No v3 target identity is known.
- `TRIGGER_ONCE` is absent.
- No v3 workflow has run.
- Neither laptop is used.

The only permitted next sequence is: snapshot-free validation on a separate branch, then add the v3 trigger as the final commit, open one draft PR, and never push another commit to that PR.
