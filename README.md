# Director OS

Director OS is an autonomous video production agent that turns raw footage, creative direction, reference videos, brand rules, and licensed music into publish-ready videos.

## Current system

The repository now contains a runnable authenticated platform and Tier 1 production engine:

- Next.js workspace client with account recovery, team administration, production library, revision chat, secure delivery, and guided browser capture
- FastAPI control plane with Director Contracts, Director Camera, Director Memory, immutable revisions, audit events, and credit metering
- PostgreSQL persistence with Alembic migrations
- Redis and Celery workers for durable production, revisions, email delivery, and lifecycle cleanup
- Local or S3-compatible multipart object storage
- FFmpeg rendering, scene analysis, subject framing, captions, audio cleanup, music ducking, semantic overlays, and final quality checks
- HTTPS production deployment through Caddy

### Editorial capabilities

- Multi-source transcription, scene detection, subject framing, quality scoring, and duplicate detection
- Narration-first story construction with explicit source provenance
- Claim-to-evidence matching and narration-preserving B-roll overlays
- Word-timed filler and silence cleanup
- Brand-aware captions, visual treatment, licensed music selection, fades, ducking, and loudness normalization
- Reference-video pace and visual fingerprints
- Editorial critic enforcing duration, must-include, and must-avoid rules
- Natural-language revisions with locked ranges, version comparison, activation, undo, redo, and partial rerendering
- Director Memory from accepted/rejected edits and weakly weighted performance evidence
- Director Camera readiness scoring, pickup missions, mission validation, continuity ghost frames, and automatic pickup insertion

The current semantic and revision layers are deliberately inspectable. They do not claim universal object understanding or unconstrained natural-language editing.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Web client: `http://localhost:3000`
- API health: `http://localhost:8000/api/v1/health`
- API docs: `http://localhost:8000/docs`

The development API runs `alembic upgrade head` before startup. Set `DIRECTOR_OPENAI_API_KEY` to enable timestamped speech transcription; conservative visual fallbacks remain available when transcription is optional.

## Authentication and workspaces

Register or sign in through the web client. The platform uses:

- salted PBKDF2 password hashes;
- short-lived signed access tokens;
- rotating refresh tokens stored as hashes;
- session-family revocation when refresh-token reuse is detected;
- single-use email verification, password-reset, and invitation tokens;
- owner, admin, editor, and viewer workspace roles.

Production can require verified email with:

```dotenv
DIRECTOR_REQUIRE_VERIFIED_EMAIL=true
```

## Uploads and storage

The client uses multipart uploads for new footage. Local development stores provider-compatible parts under the shared data volume. Production can use S3-compatible storage:

```dotenv
DIRECTOR_OBJECT_STORAGE_PROVIDER=s3
DIRECTOR_S3_BUCKET=director-production
DIRECTOR_S3_REGION=ap-south-1
DIRECTOR_S3_ACCESS_KEY_ID=<secret>
DIRECTOR_S3_SECRET_ACCESS_KEY=<secret>
```

The bucket must allow the configured web origin to upload parts and expose the `ETag` header. Director OS verifies the completed object and records storage provenance before promoting it into a normal project asset.

## Production flow

```text
Create Director Contract
→ multipart footage upload
→ reserve workspace credits
→ sensory analysis
→ Director Camera readiness audit
→ pickup missions when required
→ story and overlay planning
→ editorial critic
→ render and quality control
→ settle credits
→ secure preview/download
→ natural-language revisions
→ memory and performance learning
```

Production states include `created`, `uploading`, `ready_to_queue`, `queued`, `analyzing`, `needs_pickups`, `planning`, `rendering`, `quality_check`, `ready`, and `failed`.

## Billing ledger

Billing is currently an internal credit ledger—not card charging.

- Each workspace has total, reserved, and available credits.
- Starting production reserves an estimated amount before queue acceptance.
- Queue rejection or terminal failure releases the reservation.
- A successful render settles the reservation.
- Adjustments and lifecycle entries are append-only and idempotent.

## Email and cleanup workers

Celery Beat schedules:

- pending outbox delivery every minute;
- expired upload/token/session/invitation/audit cleanup every hour.

Local development uses the database outbox. Production can configure SMTP using the variables in `.env.example`.

## Production deployment

```bash
cp .env.example .env
# Fill production secrets, domain, database password, email, and storage settings.
docker compose -f compose.production.yml up --build -d
```

The production stack runs a one-shot migration service before API and worker startup. Only Caddy ports 80/443 are public.

For a database created before Alembic:

```bash
cd backend
alembic stamp 20260724_0001
alembic upgrade head
```

Rehearse that migration against a restored backup before deploying it to the live database.

## Documentation

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/director-camera.md`](docs/director-camera.md)
- [`docs/platform-workspaces.md`](docs/platform-workspaces.md)
- [`docs/production-operations.md`](docs/production-operations.md)
