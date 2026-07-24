# Release qualification

Director OS uses a separate release-qualification workflow for checks that are slower or more infrastructure-heavy than the normal backend and frontend suites.

## Automated gates

Every qualifying pull request runs two independent jobs.

### Real media render

The media job installs the operating-system FFmpeg package and generates temporary synthetic sources at runtime. It then exercises the actual production renderer with:

- two video inputs with different orientations;
- one source with narration audio and one source without audio;
- source-specific 9:16 reframing;
- multi-segment concat;
- ASS caption rendering through libass;
- music looping, fades, sidechain ducking, mixing, and loudness normalization;
- H.264/AAC output;
- vertical-output quality checks; and
- fast-start MP4 atom ordering.

This test exists specifically to catch missing FFmpeg filters, codec support, channel-layout problems, and invalid filter graphs that ordinary unit tests cannot detect.

### Full-stack rehearsal

The stack job builds and starts PostgreSQL, Redis, FastAPI, Celery, Next.js, and the shared storage volume using the development Compose topology. It then verifies:

- dependency readiness;
- public liveness;
- protected metrics;
- Celery worker connectivity;
- frontend and API container health;
- a bounded concurrent health probe; and
- service logs as a retained workflow artifact.

The concurrency probe is deliberately small. It is a release regression signal, not a substitute for capacity planning on the target host.

## Running locally

Real media qualification:

```bash
cd backend
DIRECTOR_RUN_MEDIA_SMOKE=1 pytest -q tests/test_ffmpeg_integration.py
```

Full-stack qualification:

```bash
cp .env.example .env
docker compose up --build -d postgres redis api worker frontend
bash ops/smoke.sh http://127.0.0.1:8000
python3 ops/load_probe.py \
  --url http://127.0.0.1:8000/api/v1/health/live \
  --requests 240 \
  --concurrency 24 \
  --p95-ms 1000
docker compose down -v --remove-orphans
```

## Release decision

A code revision is eligible for deployment only when all of these are green:

1. Backend CI
2. Frontend CI
3. Release Qualification / media-render
4. Release Qualification / stack-rehearsal
5. Target-environment checks in `docs/launch-readiness.md`

The repository checks prove the code and container topology. They do not prove DNS, certificate issuance, external SMTP reputation, cloud IAM, Stripe account configuration, encrypted off-host backups, or target-host capacity.