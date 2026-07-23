# Director OS

Director OS is an autonomous video production agent that turns raw footage, creative directions, reference videos, and brand assets into publish-ready videos.

## Current implementation

The repository now contains a runnable Tier 1 backend path with:

- FastAPI project, upload, queue, status, and intelligence APIs
- Typed Director Contract with six-tier validation
- PostgreSQL persistence for projects, assets, analyses, and Edit Decision Graphs
- Streamed uploads with size, type, and SHA-256 validation
- Durable Celery + Redis processing with sequential worker execution
- FFmpeg media probing, local scene-boundary detection, audio extraction, and 9:16 rendering
- Provider-backed transcription with word and segment timestamps
- Explainable Tier 1 segment scoring with reasons and confidence
- Word-timed removal of vocal fillers and long internal pauses
- Animated burned-in captions positioned above common social-platform UI zones
- Sampled face detection for stable subject-aware vertical crops with a centre fallback
- Voice cleanup, noise reduction, and social-video loudness normalization
- FFmpeg rendering driven by the stored Edit Decision Graph
- Persisted processing states and retry/failure reporting
- Automated output dimension, duration, and audio-presence checks
- Docker Compose development stack
- Backend linting and tests in GitHub Actions

This is a deterministic Tier 1 production engine, not yet the finished autonomous director. It can create a traceable cleaned talking-head cut with captions, framing, and finished audio. Music selection, reference-style compilation, brand graphics, advanced caption design, revisions, billing, and Director Camera remain upcoming milestones.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at `http://localhost:8000`.

```bash
curl http://localhost:8000/api/v1/health
```

To enable speech transcription, set `DIRECTOR_OPENAI_API_KEY` in `.env`. With `DIRECTOR_REQUIRE_TRANSCRIPTION=false`, footage can still use the conservative scene-based fallback when credentials are absent. Word cleanup and captions require timestamped transcription; subject framing can operate locally.

## Project workflow

### 1. Create the Director Contract

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "9afc424f-91af-4f13-b917-44f778f18b9d",
    "contract": {
      "objective": "Create a polished business reel",
      "target_audience": "small-business owners",
      "tier": 1,
      "target_duration_seconds": 45,
      "must_avoid": ["emojis"],
      "creative_freedom": 0.6
    }
  }'
```

### 2. Upload source footage

Replace `<PROJECT_ID>` with the returned project ID.

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/assets \
  -F 'kind=source_video' \
  -F 'file=@./raw-footage.mp4;type=video/mp4'
```

Supported asset kinds are `source_video`, `reference_video`, `logo`, and `brand_asset`.

### 3. Queue production

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/start
```

### 4. Read durable project status

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>
```

### 5. Inspect what the director understood and selected

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>/intelligence
```

The response includes media metadata, transcript and scene analysis, subject-framing confidence, and the versioned Edit Decision Graph used for rendering.

Project states include `created`, `uploading`, `ready_to_queue`, `queued`, `analyzing`, `planning`, `rendering`, `quality_check`, `ready`, and `failed`.

See [`docs/architecture.md`](docs/architecture.md) for system boundaries and the next production milestones.
