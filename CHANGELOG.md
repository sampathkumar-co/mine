# Changelog

## 1.1.0 — Unreleased

Director OS v1.1 development has started with beat-aware sound direction:

- Local FFmpeg-backed rhythm analysis now estimates tempo, beat grids, phrase boundaries, and confidence for user-supplied music
- Music selection prefers rhythmically reliable tracks when energy matches are otherwise equal
- Rendering chooses a deterministic playback phase that aligns edit boundaries to beats and source changes to musical phrases
- Low-confidence or failed rhythm analysis falls back to the proven v1.0 music mix instead of blocking production
- Beat detection, timing-plan selection, and bounded aligned-music preparation are covered by unit tests
- The selected timing plan is now persisted in project intelligence and reused by rendering without a second analysis pass
- Phrase-aware ducking, musical section lifts, and restrained track-derived stings now follow the persisted timing plan

## 1.0.0 — 2026-07-24

Director OS v1.0 delivers the complete code-owned autonomous video-production platform:

- Persistent Director Contracts, project assets, durable workers, retries, progress, and quality control
- Transcription, scene analysis, multi-clip story construction, captions, cleanup, reframing, music, and reference/brand style
- Evidence-aware overlays, editorial critic, final-render repair boundaries, and real FFmpeg qualification
- Immutable natural-language revisions, partial rerendering, comparison, activation, undo, and Director Memory
- Director Camera readiness audits, pickup missions, capture validation, guided browser capture, and automatic insertion
- Authenticated workspaces, roles, invitations, rotating sessions, resumable and multipart uploads, and signed delivery
- Billing ledger, subscriptions, entitlements, webhook idempotency, privacy exports, deletion workflows, and audit history
- HTTPS deployment, migrations, readiness, metrics, rate limiting, backups, restore tooling, smoke checks, and load rehearsal
- Release doctor, content-addressed release manifest, dependency audits, CycloneDX SBOMs, and v1 release-candidate packaging

External DNS, cloud accounts, payment credentials, SMTP reputation, off-host backup storage, monitoring destinations, legal approval, and target-host rehearsals remain deployment responsibilities and are enforced as explicit release gates.
