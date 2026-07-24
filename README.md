# Director OS

Director OS is an autonomous video production agent that turns raw footage, creative directions, reference videos, brand rules, and licensed music into publish-ready videos.

## Current implementation

The repository contains a runnable Tier 1 backend path with:

- FastAPI project, upload, queue, status, intelligence, and revision APIs
- Typed Director Contract with six-tier validation
- PostgreSQL persistence for projects, assets, analyses, active graphs, and append-only graph revisions
- Streamed uploads with size, type, and SHA-256 validation
- Durable Celery + Redis processing with sequential worker execution
- Per-source probing, transcription, scene detection, and subject framing
- Exact and perceptual duplicate detection across uploaded clips
- Clip-role classification for primary speech, B-roll, evidence, and rejected takes
- Quality scoring and explainable rejection reasons for every source clip
- Local semantic tags from filenames, transcripts, composition, motion, light, colour, and subject framing
- Narration-first cross-clip story construction with explicit source asset IDs
- Claim-to-evidence matching for timed B-roll and evidence overlays
- Continuity scoring across adjacent narration sources
- A pre-render editorial critic with must-include and must-avoid enforcement
- Two-pass FFmpeg rendering that preserves narration while placing full-frame visual overlays
- Word-timed filler and pause cleanup across multiple source transcripts
- Multi-source brand-aware captions rendered above visual overlays
- Reference-video fingerprints for pace, brightness, saturation, and motion
- Safe compilation of brand caption, visual, and music rules
- Selection among user-uploaded licensed music tracks by energy fit
- Music fades, speech-aware ducking, and final loudness normalization
- Natural-language deterministic revisions with locked output ranges
- Version comparison, activation, undo, and redo through immutable revision history
- Component-isolated rerendering that reuses cached narration for caption/B-roll-only changes
- Automated output dimension, duration, and audio-presence checks
- Docker Compose development stack and GitHub Actions tests

This remains a deterministic Tier 1 production engine, not the finished autonomous director. Semantic tags and revision interpretation are inspectable rules rather than opaque claims of universal understanding. Model-backed revision interpretation, segment-level narration caches, advanced object tracking, optical continuity repair, billing, performance learning, and Director Camera remain future milestones.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at `http://localhost:8000`.

```bash
curl http://localhost:8000/api/v1/health
```

To enable speech transcription, set `DIRECTOR_OPENAI_API_KEY` in `.env`. With `DIRECTOR_REQUIRE_TRANSCRIPTION=false`, footage can still use conservative visual-scene fallbacks. Word cleanup and captions require timestamped transcription; reference analysis, semantic frame sampling, subject framing, duplicate fingerprints, and uploaded-music analysis run locally.

## Project workflow

### 1. Create the Director Contract

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "9afc424f-91af-4f13-b917-44f778f18b9d",
    "contract": {
      "objective": "Explain the result and show proof from the dashboard",
      "target_audience": "small-business owners",
      "tier": 1,
      "target_duration_seconds": 45,
      "must_include": ["dashboard proof"],
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

### 2. Upload source clips and optional assets

Replace `<PROJECT_ID>` with the returned project ID. Upload `source_video` more than once to create a multi-clip project. The default VPS-safe limit is eight source clips.

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/assets \
  -F 'kind=source_video' \
  -F 'file=@./talking-head.mp4;type=video/mp4'

curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/assets \
  -F 'kind=source_video' \
  -F 'file=@./dashboard-proof.mp4;type=video/mp4'
```

Optional reference video and user-licensed music use the same endpoint with `kind=reference_video` or `kind=music`.

### 3. Queue production

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/start
```

### 4. Inspect status and director intelligence

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>/intelligence
```

## Revision workflow

### Create a natural-language revision

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/revisions \
  -H 'Content-Type: application/json' \
  -d '{
    "instruction": "Shorten it to 30 seconds, use larger all-caps captions, and remove B-roll",
    "locked_ranges": [
      {"start": 0, "end": 3.5, "label": "Keep approved hook"}
    ]
  }'
```

The deterministic revision compiler currently understands duration tightening, intro/outro trims, quoted phrase removal, caption visibility/case/size, music enable/disable, and B-roll removal/reduction. Unrecognized instructions are preserved as warnings rather than silently guessed.

### List and inspect revisions

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>/revisions
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>/revisions/2
```

### Compare two versions

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>/revisions/1/compare/2
```

The comparison reports changed components, changed output ranges, reusable ranges, graph fingerprints, and whether the cached narration master can be reused.

### Undo or redo by activating a completed version

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/revisions/1/activate
```

Revision rows and outputs are immutable. Activation only changes the project's active graph and output pointer, so any ready version can be restored without losing newer versions.

Project states include `created`, `uploading`, `ready_to_queue`, `queued`, `analyzing`, `planning`, `rendering`, `quality_check`, `ready`, and `failed`. Revision states include `queued`, `planning`, `rendering`, `quality_check`, `ready`, and `failed`.

See [`docs/architecture.md`](docs/architecture.md) for system boundaries and upcoming milestones.
