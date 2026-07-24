# Director OS Platform Workspaces

This milestone turns the single-project client into an authenticated production workspace.

## Identity and workspaces

- Registration creates a user, a default workspace, and an owner membership in one transaction.
- Passwords are stored with salted PBKDF2-HMAC-SHA256 hashes.
- Access sessions are signed with `DIRECTOR_AUTH_SECRET` and expire after `DIRECTOR_AUTH_SESSION_MINUTES`.
- The browser stores the access token in session storage, so closing the tab session removes it.
- Every project belongs to a workspace. Project, asset, revision, Director Camera, and memory URLs are authorized before route logic runs.
- Cross-workspace project lookups return `404` to avoid confirming that another workspace's project exists.

The test suite may explicitly set `DIRECTOR_AUTH_REQUIRED=false`. Production startup rejects the built-in development secret.

## Resumable uploads

The client creates an upload session, then sends fixed-size chunks with the server-confirmed offset:

```text
POST  /projects/{project_id}/uploads
GET   /uploads/{upload_id}
PATCH /uploads/{upload_id}   Upload-Offset: <bytes>
```

A mismatched offset returns `409` and the authoritative `Upload-Offset` response header. When the declared byte count is complete, the backend calculates SHA-256, atomically moves the temporary file, creates the project asset, and advances the project to `ready_to_queue`.

Upload sessions never accept more than the Director OS project limit or the configured per-request chunk limit.

## Secure delivery

The database and API expose only `output_available`, never server file paths. An authenticated workspace member requests a short-lived delivery link:

```text
POST /projects/{project_id}/delivery
POST /projects/{project_id}/delivery?version=2&download=true
```

The returned link contains a signed purpose, project ID, optional revision version, disposition, and expiry. The public delivery endpoint validates the signature and expiry before resolving the current stored path.

## Revision chat

The project studio lists immutable graph versions and sends plain-language requests through the existing isolated revision engine. Failed revisions remain visible without replacing the active publishable output.

## Continuity ghost frames

Director Camera missions can request a frame generated from existing project footage:

```text
GET /projects/{project_id}/director-camera/missions/{mission_id}/ghost-frame
```

The authenticated endpoint extracts and caches a scaled JPEG with FFmpeg. The browser overlays it on the live camera preview. This is alignment guidance; submitted footage still passes backend duration, quality, semantic, audio, duplicate, and continuity validation.

## Production HTTPS

Use the standalone production stack:

```bash
cp .env.example .env
# Set a strong DIRECTOR_AUTH_SECRET, DIRECTOR_POSTGRES_PASSWORD,
# DIRECTOR_DOMAIN, and DIRECTOR_ACME_EMAIL.
docker compose -f compose.production.yml up -d --build
```

Caddy obtains and renews TLS certificates, routes `/api/*` to FastAPI, routes other traffic to Next.js, and applies HSTS, frame denial, content-type protection, a strict referrer policy, and camera/microphone permissions limited to the same origin.

## Migration boundary

The development stack still uses SQLAlchemy `create_all`. Adding `workspace_id` to an existing production database requires a real migration before deployment. Alembic migrations, password reset, email verification, invitations, SSO, refresh-token rotation, upload expiry cleanup, and object-storage multipart uploads remain subsequent platform work.
