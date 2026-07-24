# Launch readiness and recovery

This document is the operational gate for a public Director OS deployment. Code paths are necessary but not sufficient: the external services, credentials, DNS, storage policies, and restore rehearsal must be completed in the target environment.

## Required production configuration

Production startup intentionally fails when these safety boundaries are not met:

- authentication is enabled and `DIRECTOR_AUTH_SECRET` is not the development value;
- automatic schema creation is disabled and Alembic migrations are used;
- Redis-backed rate limiting is enabled and configured to fail closed;
- Redis is required by the readiness check;
- metrics are protected with `DIRECTOR_METRICS_TOKEN`;
- Stripe keys exist whenever subscriptions are enabled.

Use long random values for authentication, metrics, database, SMTP, Stripe, and object-storage credentials. Keep `.env` outside version control and restrict it to the deployment operator.

## Health and observability

- `GET /api/v1/health/live` proves that the API process is running.
- `GET /api/v1/health/ready` checks PostgreSQL, Redis, and writable runtime storage. Production responses reveal component status without internal error details.
- `GET /api/v1/metrics` emits Prometheus text and requires a bearer token when configured.
- Request logs are structured JSON with request ID, normalized route, response class, duration, actor, and workspace where available.
- Authentication, webhook, and mutation routes carry fixed-window rate limits. Production uses Redis so limits remain shared across API replicas.

Suggested initial alerts:

- readiness returns `not_ready` for two consecutive probes;
- 5xx response ratio exceeds 2% for five minutes;
- request p95 exceeds the product SLO;
- Celery queue age exceeds the expected render start window;
- Redis or PostgreSQL storage exceeds 80%;
- backup age exceeds 26 hours;
- payment webhook failures or replay conflicts occur;
- privacy deletion jobs fail.

The in-process metrics registry is suitable for a single API process. Horizontal API scaling should move aggregation to a multi-process-aware metrics stack or OpenTelemetry collector.

## Rate-limit defaults

The defaults are deliberately conservative and configurable:

- authentication and recovery: 20 requests per minute per client;
- general mutations: 120 requests per minute per authenticated user or client;
- billing webhooks: 240 requests per minute per client.

Caddy is the trusted proxy in the provided production Compose stack. Set `DIRECTOR_TRUST_PROXY_HEADERS=true` only behind a proxy that overwrites untrusted forwarding headers.

## Workspace exports and deletion

Workspace administrators can request a signed ZIP export. It contains:

- workspace and member metadata;
- project contracts and asset manifests;
- analysis, edit graphs, revisions, camera decisions, and performance evidence;
- billing ledger, subscription state, and audit history.

Passwords, token hashes, provider upload identifiers, and private server paths are excluded. Large media binaries are not duplicated into the archive; asset hashes, names, sizes, and relationships are included. Exports expire and are removed automatically.

Only a workspace owner can schedule deletion. The request requires the exact workspace slug and a reason. Deletion is blocked while a production job is active or a paid subscription remains active. During the grace period, workspace mutations are locked and the owner may cancel the request. When due, Director OS deletes local/S3 objects, render outputs, cached media, and the workspace database graph.

This workflow is product data governance, not legal advice. Production operators must align retention, tax-record, backup, and privacy-request policies with the jurisdictions they serve.

## Backup policy

Run backups from a trusted host with Docker access:

```bash
bash ops/backup.sh
```

The backup contains:

- a PostgreSQL custom-format dump;
- the persistent Director media/output volume;
- Redis append-only state;
- a manifest and SHA-256 checksums.

Backups are sensitive because they contain customer content and account data. Store them encrypted, off-host, with access logging and retention enforcement. The local script removes backup directories older than `DIRECTOR_BACKUP_RETENTION_DAYS`, but off-site retention must be configured separately.

Recommended schedule:

- database and persistent data: daily;
- off-site copy: immediately after each successful backup;
- restore rehearsal: monthly and before every major migration;
- retention: at least 14 daily copies plus organization-specific longer-term snapshots.

## Restore rehearsal

Restoration is destructive and requires explicit confirmation:

```bash
bash ops/restore.sh /secure/path/to/backups/20260724T120000Z --confirm
```

The script verifies checksums before stopping services, restores PostgreSQL, Director data, and Redis, runs current migrations, and restarts the stack. Complete the rehearsal with:

```bash
DIRECTOR_METRICS_TOKEN=<token> bash ops/smoke.sh https://director.example.com
```

For a deeper authenticated smoke test, set `DIRECTOR_SMOKE_EMAIL` and `DIRECTOR_SMOKE_PASSWORD` for a dedicated low-privilege test account.

Record every rehearsal with backup timestamp, restore target, duration, checksum result, migration result, smoke-test result, and operator.

## Launch checklist

1. Point DNS to the deployment and verify automatic TLS renewal.
2. Replace every development secret and restrict the `.env` file.
3. Configure SMTP and confirm verification, reset, and invitation delivery.
4. Configure S3-compatible storage, CORS, lifecycle rules, and deletion permission.
5. Run Alembic against a restored production-sized backup.
6. Configure Stripe test mode, Price IDs, portal, and signed webhooks; replay duplicate events.
7. Configure monitoring for health, metrics, logs, queues, storage, backups, email, and webhooks.
8. Run backup, destructive restore rehearsal, and smoke verification.
9. Exercise account recovery, team roles, uploads, production, pickup capture, revisions, delivery, workspace export, and deletion cancellation.
10. Review privacy policy, terms, music/footage rights language, retention periods, support channel, and incident contacts.
11. Start with a limited cohort and a documented rollback decision-maker.

## Known external boundaries

The repository cannot complete these without deployment credentials or real infrastructure:

- DNS and public TLS issuance;
- SMTP reputation and deliverability;
- S3 bucket CORS, lifecycle, encryption, and IAM validation;
- Stripe test/live account configuration and tax settings;
- off-host encrypted backup storage;
- production monitoring destinations and alert routing;
- legal/privacy/terms review;
- a real-media end-to-end render rehearsal on the target host.
