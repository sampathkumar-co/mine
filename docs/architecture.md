# Director OS Architecture

## Product boundary

Director OS accepts footage plus a Director Contract and produces a publish-ready video through five durable stages:

1. Sensory analysis
2. Editorial planning
3. Edit Decision Graph generation
4. Deterministic rendering
5. Automated quality inspection

## Initial services

- **FastAPI API**: authentication boundary, project intake, status, billing orchestration
- **Celery worker**: durable sequential processing with concurrency fixed to one for low-resource hosts
- **Redis**: task broker and short-lived job results
- **PostgreSQL**: projects, credits, assets, edit decisions, versions, and audit records
- **FFmpeg**: deterministic rendering and media inspection

## Safety rules

- Never load full uploads into application memory.
- Reserve credits before queueing and settle them only after successful processing.
- Every job stage must be idempotent and resumable.
- Source assets must remain available through the configured revision window.
- Final deletion requires a verified client download or explicit retention expiry.
- AI providers must sit behind replaceable adapters.

## Planned modules

- `app/director`: Director Contract, style compiler, editorial decision agents
- `app/sensory`: transcription, scene, speaker, object, and quality analysis
- `app/rendering`: Edit Decision Graph compiler and FFmpeg execution
- `app/quality`: output inspection and automatic repair
- `app/billing`: credit reservations, usage ledger, refunds, and cost telemetry
- `app/storage`: chunked uploads, signed downloads, retention, and purge jobs
- `app/director_camera`: missing-shot missions and continuity capture metadata
