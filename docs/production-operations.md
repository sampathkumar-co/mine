# Director OS Production Operations

This release replaces development-only shortcuts with explicit production boundaries for schema changes, account recovery, team permissions, storage, lifecycle cleanup, auditability, and usage credits.

## Database migrations

Alembic is now the authoritative production schema tool.

Fresh database:

```bash
cd backend
alembic upgrade head
```

Existing database created before Alembic:

```bash
cd backend
alembic stamp 20260724_0001
alembic upgrade head
```

Review a backup and rehearse this sequence against a restored database before production deployment. Production refuses `DIRECTOR_AUTO_CREATE_SCHEMA=true`.

The production Compose stack runs a one-shot `migrate` service. API, worker, and scheduler start only after migrations succeed.

## Session security

Login and registration first establish a short bootstrap access token. The browser immediately exchanges it for a server-recorded session containing:

- a short-lived signed access token;
- a random refresh token stored only as a SHA-256 digest;
- a session-family identifier;
- expiry, rotation, revocation, user-agent, and IP metadata.

Every refresh rotates the refresh token. Reuse of an already-rotated token revokes the entire session family. Password reset revokes every session for the user.

Recommended production values:

```dotenv
DIRECTOR_ACCESS_TOKEN_MINUTES=15
DIRECTOR_REFRESH_TOKEN_DAYS=30
DIRECTOR_AUTH_SECRET=<at-least-32-random-characters>
DIRECTOR_REQUIRE_VERIFIED_EMAIL=true
```

## Email delivery

The default `database` provider records messages in `email_outbox`; Celery Beat dispatches a delivery task once per minute. It is suitable for tests and inspecting local flows.

Production SMTP configuration:

```dotenv
DIRECTOR_EMAIL_PROVIDER=smtp
DIRECTOR_SMTP_HOST=smtp.example.com
DIRECTOR_SMTP_PORT=587
DIRECTOR_SMTP_STARTTLS=true
DIRECTOR_SMTP_USERNAME=director-os
DIRECTOR_SMTP_PASSWORD=<secret>
DIRECTOR_SMTP_FROM_EMAIL=Director OS <production@example.com>
DIRECTOR_PUBLIC_APP_URL=https://director.example.com
```

Verification, password-reset, and invitation tokens are random, single-use, hashed at rest, and automatically expire.

## Workspace roles

| Role | Read projects | Create/edit production | Manage members/invitations | Manage billing adjustments |
|---|---:|---:|---:|---:|
| Viewer | Yes | No | No | No |
| Editor | Yes | Yes | No | No |
| Admin | Yes | Yes | Yes | No |
| Owner | Yes | Yes | Yes | Yes |

A workspace must always retain at least one owner. Invites cannot grant ownership; an owner must explicitly transfer it.

## Multipart object storage

`DIRECTOR_OBJECT_STORAGE_PROVIDER=local` uses a provider-compatible multipart flow on the shared data volume. `s3` uses an S3-compatible multipart API and short-lived presigned upload-part URLs.

```dotenv
DIRECTOR_OBJECT_STORAGE_PROVIDER=s3
DIRECTOR_S3_BUCKET=director-production
DIRECTOR_S3_REGION=ap-south-1
DIRECTOR_S3_ENDPOINT_URL=
DIRECTOR_S3_ACCESS_KEY_ID=<secret>
DIRECTOR_S3_SECRET_ACCESS_KEY=<secret>
DIRECTOR_MULTIPART_PART_BYTES=8388608
DIRECTOR_MULTIPART_PRESIGN_MINUTES=15
```

The bucket CORS policy must permit the web origin to `PUT` parts and expose the `ETag` response header. Credentials never enter the browser. After S3 completion, the worker downloads and verifies a local cache so the existing FFmpeg pipeline continues to consume local file paths. `StoredObject` preserves provider, bucket, object key, cache path, and verification state.

## Lifecycle cleanup

Celery Beat schedules:

- queued email delivery every minute;
- expired upload/token/session/invitation/audit cleanup every hour.

Cleanup covers abandoned multipart uploads, old resumable `.part` files, expired account tokens, expired refresh-session records, stale invitations, and audit events beyond the configured retention.

## Audit events

Every mutating HTTP request receives an `X-Request-ID` and produces a best-effort append-only audit event containing:

- workspace and actor;
- request action;
- resource type and identifier;
- request ID;
- IP and user agent;
- response status metadata.

Audit persistence never turns an otherwise successful user action into a failure. Administrators can inspect recent workspace events in the Operations screen.

## Credit ledger

Each workspace has:

- total credit balance;
- currently reserved credits;
- available credits;
- an append-only, idempotent ledger.

Starting production estimates and reserves credits before queue acceptance. Queue rejection releases the reservation. A successful render settles it. A terminal worker failure releases it. The ledger is internal metering—not card charging or a payment processor.

Starter credits are granted once, using an idempotency key, when billing is first activated for a workspace.

## Production deployment

```bash
cp .env.example .env
# Fill the production database password, auth secret, domain, mail and storage settings.
docker compose -f compose.production.yml up --build -d
```

The public edge remains Caddy on ports 80/443. PostgreSQL, Redis, API, worker, scheduler, and migration services remain on the internal Compose network.
