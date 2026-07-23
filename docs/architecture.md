# Director OS Architecture

## Product boundary

Director OS accepts footage plus a Director Contract and produces a publish-ready video through five durable stages:

1. Sensory analysis
2. Editorial and style planning
3. Edit Decision Graph generation
4. Deterministic rendering
5. Automated quality inspection and repair

## Implemented control plane

- **FastAPI API**: project creation, Director Contract validation, asset upload, queueing, status reads, and intelligence inspection
- **PostgreSQL**: durable project state, assets, analyses, Edit Decision Graphs, task identifiers, errors, and output availability
- **Streamed storage**: bounded chunks, server-generated filenames, content-type rules, SHA-256 hashes, and partial-file cleanup
- **Celery worker**: Redis-backed sequential processing with persisted stage transitions and bounded retries
- **Sensory boundary**: source probing, timestamped transcription, scene detection, sampled face framing, reference fingerprints, and uploaded-music profiling
- **Director boundary**: segment scoring, word cleanup, target-duration selection, reference/brand style compilation, music energy matching, reasons, confidence, and versioned Edit Decision Graphs
- **Rendering boundary**: graph-driven trims, subject-aware vertical composition, brand captions, visual finishing, voice cleanup, music fades, speech-aware ducking, and final encoding
- **Docker Compose**: PostgreSQL and Redis readiness checks before API and worker startup

## Current Tier 1 pipeline

```text
created
  -> uploading
       -> source video
       -> optional reference video
       -> optional licensed music
       -> optional brand assets
  -> ready_to_queue
  -> queued
  -> analyzing
       -> source media probe
       -> scene boundaries
       -> optional timestamped transcript
       -> sampled subject framing
       -> optional reference pace/colour/motion fingerprint
       -> optional uploaded-music loudness and energy profiles
  -> planning
       -> candidate scoring and target-duration selection
       -> word-timed filler and silence cleanup
       -> brand/reference Production Style compilation
       -> licensed music energy matching
       -> caption cue generation
       -> persisted analysis and Edit Decision Graph
  -> rendering
       -> graph-driven FFmpeg trim/concat
       -> subject-aware 1080x1920 composition
       -> brand-aware caption burn-in
       -> voice cleanup and loudness normalization
       -> music loop, fade, sidechain ducking, and mix
  -> quality_check
       -> dimension validation
       -> duration-to-graph validation
       -> required audio-presence validation
  -> ready | failed
```

When transcription credentials are absent and transcription is not mandatory, the planner uses conservative visual scene ranges. Word cleanup and captions are then omitted rather than simulated.

Reference videos are converted into measurable editorial traits such as average shot duration, cut density, brightness, saturation, and motion. Protected assets, logos, music, and exact frames are not copied into the output.

Music is selected only from user-uploaded audio assets. The product does not silently download or attach third-party tracks, and users remain responsible for having appropriate usage rights.

## Next production milestones

1. Alembic database migrations
2. Authentication and project ownership enforcement
3. Credit reservation, usage ledger, and cost telemetry
4. Multiple source clips and take/duplicate detection
5. Product-aware framing and evidence-aware B-roll selection
6. Repetition and bad-take detection beyond vocal fillers
7. Beat-aware cuts and richer sound design
8. Brand logos, graphic templates, and reusable campaign kits
9. Revision isolation, segment locking, versions, and partial rerenders
10. Editorial, brand, and instruction-compliance critic passes
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
- Reference analysis may guide style but must not copy protected creative assets.
- Only user-supplied music may enter the render pipeline.
- Raw brand values are sanitized and bounded before reaching FFmpeg or ASS rendering.

## Module boundaries

- `app/director`: Director Contract compiler, production-style compiler, editorial agents, cleanup, and decision graphs
- `app/sensory`: transcription, scenes, framing, reference fingerprints, music profiles, speakers, objects, emotion, motion, and quality analysis
- `app/rendering`: captions, deterministic timeline compilation, audio mixing, and FFmpeg execution
- `app/quality`: technical, editorial, brand, and instruction-compliance checks
- `app/billing`: credit reservations, usage ledger, refunds, and cost telemetry
- `app/storage`: resumable uploads, signed downloads, retention, and purge jobs
- `app/director_camera`: missing-shot missions, capture guidance, and continuity verification
