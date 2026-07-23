# Director OS Architecture

## Product boundary

Director OS accepts footage plus a Director Contract and is designed to produce a publish-ready video through five durable stages:

1. Sensory analysis
2. Editorial planning
3. Edit Decision Graph generation
4. Deterministic rendering
5. Automated quality inspection and repair

## Implemented control plane

- **FastAPI API**: project creation, Director Contract validation, asset upload, queueing, status reads, and intelligence inspection
- **PostgreSQL**: durable project state, assets, analyses, Edit Decision Graphs, task identifiers, errors, and output availability
- **Streamed storage**: bounded chunks, server-generated filenames, content-type rules, SHA-256 hashes, and partial-file cleanup
- **Celery worker**: Redis-backed sequential processing with persisted stage transitions and bounded retries
- **Sensory boundary**: FFmpeg media probing, speech extraction, provider-backed timestamped transcription, and local scene detection
- **Director boundary**: deterministic Tier 1 scoring, target-duration selection, reasons, confidence, and versioned Edit Decision Graphs
- **FFmpeg boundary**: graph-driven segment trimming, vertical rendering, output probing, and duration/dimension validation
- **Docker Compose**: PostgreSQL and Redis readiness checks before API and worker startup

## Current Tier 1 pipeline

```text
created
  -> uploading
  -> ready_to_queue
  -> queued
  -> analyzing
       -> media probe
       -> scene boundaries
       -> optional timestamped transcript
       -> persisted analysis
  -> planning
       -> candidate scoring
       -> target-duration selection
       -> persisted Edit Decision Graph
  -> rendering
       -> graph-driven FFmpeg trim/concat
       -> 1080x1920 output
  -> quality_check
       -> dimension validation
       -> duration-to-graph validation
  -> ready | failed
```

When transcription credentials are absent and transcription is not mandatory, the planner uses conservative visual scene ranges. This keeps local development and non-speech footage runnable without pretending that transcript-level judgment occurred.

## Next production milestones

1. Alembic database migrations
2. Authentication and project ownership enforcement
3. Credit reservation and usage ledger
4. Word-precise silence, filler, repetition, and bad-take removal
5. Caption generation and safe-zone composition
6. Face-aware and product-aware reframing
7. Audio cleanup, music selection, ducking, and beat alignment
8. Reference-video style fingerprinting
9. Director Contract compilation across references, brand rules, and locked clips
10. Editorial and instruction-compliance critic passes
11. Resumable multipart uploads and signed downloads
12. Director Camera pickup missions and continuity metadata

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
- Analysis and edit decisions remain inspectable and versioned.
- A visual-only fallback must be labelled as lower-confidence behavior.

## Module boundaries

- `app/director`: Director Contract compiler, style compiler, editorial agents, and decision graphs
- `app/sensory`: transcription, scenes, speakers, faces, objects, emotion, motion, and quality analysis
- `app/rendering`: deterministic timeline compilation and FFmpeg execution
- `app/quality`: technical, editorial, brand, and instruction-compliance checks
- `app/billing`: credit reservations, usage ledger, refunds, and cost telemetry
- `app/storage`: resumable uploads, signed downloads, retention, and purge jobs
- `app/director_camera`: missing-shot missions, capture guidance, and continuity verification
