# Director OS Architecture

## Product boundary

Director OS accepts footage plus a Director Contract and is designed to produce a publish-ready video through five durable stages:

1. Sensory analysis
2. Editorial planning
3. Edit Decision Graph generation
4. Deterministic rendering
5. Automated quality inspection and repair

## Implemented control plane

- **FastAPI API**: project creation, Director Contract validation, asset upload, queueing, and status reads
- **PostgreSQL**: durable project state, Director Contract data, task identifiers, assets, errors, and output availability
- **Streamed storage**: bounded chunks, server-generated filenames, content-type rules, SHA-256 hashes, and partial-file cleanup
- **Celery worker**: Redis-backed sequential processing with persisted stage transitions and bounded retries
- **FFmpeg boundary**: source probing, baseline vertical rendering, output probing, and technical validation
- **Docker Compose**: PostgreSQL and Redis readiness checks before API and worker startup

## Current baseline pipeline

```text
created
  -> uploading
  -> ready_to_queue
  -> queued
  -> analyzing
  -> planning
  -> rendering
  -> quality_check
  -> ready | failed
```

The current worker renders the first source video to a standard 1080x1920 deliverable. This proves upload, persistence, queueing, rendering, retry, and QC boundaries. It is not yet the autonomous editorial system.

## Next production milestones

1. Resumable multipart uploads and signed downloads
2. Alembic database migrations
3. Authentication and project ownership enforcement
4. Credit reservation and usage ledger
5. Transcription and word-level timing
6. Scene, face, speaker, motion, audio, and take-quality analysis
7. Director Contract compiler and conflict resolution
8. Edit Decision Graph with reversible decisions
9. Tier 1 autonomous cuts, captions, audio cleanup, music, reframing, and final QC
10. Director Camera pickup missions and continuity metadata

## Safety and reliability rules

- Never load full uploads into application memory.
- Never trust client filenames as storage paths.
- Reserve credits before queueing and settle only measured usage.
- Every processing stage must be idempotent and resumable.
- Source assets remain available through the configured revision window.
- Final deletion requires verified download or explicit retention expiry.
- AI providers sit behind replaceable adapters.
- API responses never expose private server storage paths.
- A failed queue submission restores the project to a retryable state.

## Planned modules

- `app/director`: Director Contract compiler, style compiler, editorial agents, and decision graph
- `app/sensory`: transcription, scenes, speakers, faces, objects, emotion, motion, and quality analysis
- `app/rendering`: deterministic timeline compilation and FFmpeg execution
- `app/quality`: technical, editorial, brand, and instruction-compliance checks
- `app/billing`: credit reservations, usage ledger, refunds, and cost telemetry
- `app/storage`: resumable uploads, signed downloads, retention, and purge jobs
- `app/director_camera`: missing-shot missions, capture guidance, and continuity verification
