# Director OS

Director OS is an autonomous video production agent that turns raw footage, creative directions, reference videos, and brand assets into publish-ready videos.

## Current implementation

The repository now contains a runnable backend control plane with:

- FastAPI project and upload APIs
- Typed Director Contract with six-tier validation
- PostgreSQL persistence for projects and assets
- Streamed uploads with size, type, and SHA-256 validation
- Durable Celery + Redis processing with sequential worker execution
- FFmpeg media inspection and a baseline 9:16 render
- Persisted processing states and retry/failure reporting
- Automated output dimension and duration checks
- Docker Compose development stack
- Backend linting and tests in GitHub Actions

The current FFmpeg renderer is deliberately a deterministic baseline. Transcription, multimodal footage understanding, editorial planning, captions, intelligent cuts, reference-style compilation, Director Camera, billing, and autonomous repair remain upcoming milestones.

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at `http://localhost:8000`.

```bash
curl http://localhost:8000/api/v1/health
```

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

Supported asset kinds are:

- `source_video`
- `reference_video`
- `logo`
- `brand_asset`

### 3. Queue production

```bash
curl -X POST http://localhost:8000/api/v1/projects/<PROJECT_ID>/start
```

### 4. Read durable project status

```bash
curl http://localhost:8000/api/v1/projects/<PROJECT_ID>
```

Project states currently include `created`, `uploading`, `ready_to_queue`, `queued`, `analyzing`, `planning`, `rendering`, `quality_check`, `ready`, and `failed`.

See [`docs/architecture.md`](docs/architecture.md) for system boundaries and the next production milestones.
