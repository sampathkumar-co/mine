# Changelog

## 1.0.1 — 2026-07-24

Security and reliability hardening for the v1 launch candidate:

- Removed customer-accessible credit adjustments and made starter grants user-scoped
- Made all access tokens session-backed with atomic refresh rotation and secure cookies
- Added durable production jobs, duplicate suppression, queue isolation, and recovery dispatch
- Froze Alembic migrations and added schema-drift validation
- Hardened multipart completion, deletion retries, email-token retention, CSP, and containers
- Added resumable browser uploads with IndexedDB recovery and parallel part retries


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
