# Director OS

Director OS is an autonomous video production agent that turns raw footage, creative directions, reference videos, and brand assets into publish-ready videos.

## Foundation included

- FastAPI control plane
- Typed Director Contract and six-tier project intake
- Durable Celery + Redis worker configured for sequential processing
- PostgreSQL service for future project and credit ledgers
- FFmpeg-enabled backend container
- Automated backend linting and tests
- Architecture and safety documentation

## Run locally

```bash
cp .env.example .env
docker compose up --build
```

The API will be available at `http://localhost:8000`.

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Create a project:

```bash
curl -X POST http://localhost:8000/api/v1/projects \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "9afc424f-91af-4f13-b917-44f778f18b9d",
    "contract": {
      "objective": "Create a polished business reel",
      "tier": 1,
      "target_duration_seconds": 45,
      "must_avoid": ["emojis"]
    }
  }'
```

See [`docs/architecture.md`](docs/architecture.md) for the system boundaries and planned modules.
