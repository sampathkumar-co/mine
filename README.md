# Director OS

Director OS is an autonomous video production agent that turns raw footage, creative directions, reference videos, brand rules, and licensed music into publish-ready videos.

## Current implementation

The repository contains a runnable Tier 1 backend path with:

- FastAPI project, upload, queue, status, and intelligence APIs
- Typed Director Contract with six-tier validation
- PostgreSQL persistence for projects, assets, analyses, and Edit Decision Graphs
- Streamed uploads with size, type, and SHA-256 validation
- Durable Celery + Redis processing with sequential worker execution
- Provider-backed transcription with word and segment timestamps
- Local scene detection and sampled subject framing
- Explainable segment scoring, filler removal, and pause cleanup
- Reference-video fingerprints for pace, brightness, saturation, and motion
- Safe compilation of brand caption, visual, and music rules
- Selection among user-uploaded licensed music tracks by energy fit
- Music fades, speech-aware ducking, and final loudness normalization
- Brand-aware animated captions in social-platform safe zones
- FFmpeg rendering driven by the stored Edit Decision Graph
- Automated output dimension, duration, and audio-presence checks
- Docker Compose development stack and GitHub Actions tests

This is a deterministic Tier 1 production engine, not yet the finished autonomous director. It creates a traceable talking-head cut with cleanup, captions, framing, reference-informed styling, and optional licensed music. Brand graphics, revisions, billing, performance learning, and Director Camera remain future milestones.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at `http://localhost:8000`.

```bash
curl http://localhost:8000/api/v1/health
```

To enable speech transcription, set `DIRECTOR_OPENAI_API_KEY` in `.env`. With `DIRECTOR_REQUIRE_TRANSCRIPTION=false`, footage can still use the conservative scene-based fallback. Word cleanup and captions require timestamped transcription; reference analysis, subject framing, and uploaded-music analysis run locally.

## Project workflow

### 1. Create the Director Contract

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "9afc424f-91af-4f13-b917-44f778f18b9d",
    "contract": {
      "objective": "Create a calm luxury property reel",
      "target_audience": "first-time buyers",
      "tier": 1,
      "target_duration_seconds": 45,
      "must_avoid": ["emojis"],
      "brand_rules": {
        "caption_font": "Inter",
        "caption_primary_color": "#FFFFFF",
        "caption_accent_color": "#D4AF37",
        "caption_position": "lower",
        "caption_all_caps": false,
        "music_energy": "calm",
        "music_volume": 0.14
      },
      "creative_freedom": 0.6
    }
  }'
```

### 2. Upload assets

Replace `<PROJECT_ID>` with the returned project ID.

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/assets \
  -F 'kind=source_video' \
  -F 'file=@./raw-footage.mp4;type=video/mp4'
```

Optional reference video:

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/assets \
  -F 'kind=reference_video' \
  -F 'file=@./reference.mp4;type=video/mp4'
```

Optional licensed music:

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/assets \
  -F 'kind=music' \
  -F 'file=@./licensed-track.mp3;type=audio/mpeg'
```

Supported asset kinds are `source_video`, `reference_video`, `logo`, `brand_asset`, and `music`. Music must be supplied by the user and must have appropriate usage rights.

### 3. Queue production

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/start
```

### 4. Read durable project status

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>
```

### 5. Inspect director intelligence

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>/intelligence
```

The response includes source analysis, transcript and scenes, subject-framing confidence, reference fingerprint, music profiles, compiled production style, selected music asset ID, and the versioned Edit Decision Graph used for rendering.

Project states include `created`, `uploading`, `ready_to_queue`, `queued`, `analyzing`, `planning`, `rendering`, `quality_check`, `ready`, and `failed`.

See [`docs/architecture.md`](docs/architecture.md) for system boundaries and upcoming milestones.
