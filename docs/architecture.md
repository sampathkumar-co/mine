# Director OS Architecture

## Product boundary

Director OS accepts footage plus a Director Contract and produces a publish-ready video through five durable stages:

1. Sensory analysis
2. Editorial and style planning
3. Edit Decision Graph generation
4. Deterministic rendering
5. Automated quality inspection and repair

The production engine sits behind an authenticated workspace platform with subscription entitlements, credit reservation, object storage, auditability, and durable background processing.

## Implemented control plane

- **Next.js client**: authentication, recovery, workspaces, team administration, subscription controls, production library, capture, revisions, and delivery
- **FastAPI API**: project creation, Director Contract validation, asset upload, queueing, status reads, intelligence inspection, subscription checkout, customer portal, and signed billing webhooks
- **PostgreSQL**: durable users, sessions, workspaces, projects, assets, analyses, edit graphs, subscriptions, webhook idempotency, billing ledger, audits, and output availability
- **Alembic**: ordered production schema migrations and startup migration gates
- **Storage boundary**: local or S3-compatible multipart uploads, provider-generated object keys, checksums, provenance, and lifecycle cleanup
- **Celery workers**: Redis-backed production, revision, email, and cleanup tasks with persisted stage transitions and bounded retries
- **Sensory boundary**: multi-source probing, timestamped transcription, scene detection, duplicate analysis, sampled subject framing, reference fingerprints, and uploaded-music profiling
- **Director boundary**: cross-clip story selection, evidence matching, word cleanup, target-duration selection, reference/brand style compilation, music energy matching, reasons, confidence, and versioned Edit Decision Graphs
- **Rendering boundary**: graph-driven trims, overlays, subject-aware vertical composition, brand captions, visual finishing, voice cleanup, music fades, speech-aware ducking, and final encoding
- **Commercial boundary**: Stripe behind a replaceable adapter, verified webhook state, plan entitlements, invoice-triggered credit grants, and the internal append-only usage ledger
- **Deployment boundary**: development and production Compose stacks, one-shot migrations, Caddy HTTPS termination, worker health dependencies, and scheduled lifecycle jobs

## Current production pipeline

```text
authenticated workspace
  -> resolve active plan
       -> duration entitlement
       -> Director tier entitlement
       -> workspace seat entitlement
  -> created
  -> uploading
       -> source and pickup clip limit
       -> optional reference video
       -> optional licensed music
       -> optional brand assets
  -> ready_to_queue
  -> reserve usage credits
  -> queued
  -> analyzing
       -> per-source media probe
       -> scene boundaries
       -> optional timestamped transcript
       -> sampled subject framing
       -> duplicate and bad-take signals
       -> semantic and evidence tags
       -> optional reference pace/colour/motion fingerprint
       -> optional uploaded-music loudness and energy profiles
  -> Director Camera readiness audit
       -> ready to edit
       -> or needs_pickups with capture missions
  -> planning
       -> cross-clip narration story
       -> evidence and B-roll overlays
       -> word-timed filler and silence cleanup
       -> brand/reference Production Style compilation
       -> licensed music energy matching
       -> editorial critic and repair
       -> caption cue generation
       -> persisted analysis and Edit Decision Graph
  -> rendering
       -> graph-driven multi-input FFmpeg render
       -> subject-aware 1080x1920 composition
       -> brand-aware caption burn-in
       -> voice cleanup and loudness normalization
       -> music loop, fade, sidechain ducking, and mix
  -> quality_check
       -> dimension validation
       -> duration-to-graph validation
       -> required audio-presence validation
  -> settle reserved credits
  -> ready | failed and release reservation
```

When transcription credentials are absent and transcription is not mandatory, the planner uses conservative visual scene ranges. Word cleanup and captions are then omitted rather than simulated.

Reference videos are converted into measurable editorial traits such as average shot duration, cut density, brightness, saturation, and motion. Protected assets, logos, music, and exact frames are not copied into the output.

Music is selected only from user-uploaded audio assets. The product does not silently download or attach third-party tracks, and users remain responsible for having appropriate usage rights.

## Subscription and entitlement flow

```text
workspace owner requests paid plan
  -> server creates hosted Checkout Session
  -> browser completes Checkout
  -> verified subscription webhook updates plan state
  -> verified paid-invoice webhook grants plan credits once
  -> project and team operations read active entitlements
  -> usage ledger reserves and settles production cost
```

Browser redirects never activate entitlements or grant credits. Provider event IDs and invoice IDs make webhook processing and credit grants idempotent. Cancellation returns new activity to starter limits without deleting existing workspace data.

## Remaining product milestones

1. Beat-aware cuts, musical phrase alignment, and richer sound design
2. Brand logos, motion-graphic templates, lower thirds, and reusable campaign kits
3. Stronger bad-take, repetition, blur, exposure, and clipping analysis
4. Model-backed semantic understanding and revision interpretation behind inspectable adapters
5. Advanced subject/object tracking, optical continuity repair, and more capable live Director Camera guidance
6. Real payment-provider launch rehearsal, tax configuration, revenue reconciliation, and support operations
7. Burst rendering infrastructure, queue autoscaling, and workload-aware cost telemetry
8. Cross-account anonymized benchmark learning with explicit privacy controls

## Safety and reliability rules

- Never load full uploads into application memory.
- Never trust client filenames as storage paths.
- Verify payment-provider webhook signatures against the raw request body.
- Never activate a paid plan from a browser success redirect.
- Keep payment-provider state separate from the internal usage ledger.
- Make provider events and invoice credit grants idempotent.
- Reserve credits before queueing and settle only measured usage.
- Recheck entitlements before costly work begins.
- Every processing stage must be idempotent and resumable.
- Source assets remain available through the configured revision window.
- Final deletion requires verified download or explicit retention expiry.
- AI and payment providers sit behind replaceable adapters.
- API responses never expose private server storage paths or provider secrets.
- A failed queue submission restores the project to a retryable state.
- Analysis and edit decisions remain inspectable and versioned.
- A visual-only fallback must be labelled as lower-confidence behavior.
- Reference analysis may guide style but must not copy protected creative assets.
- Only user-supplied music may enter the render pipeline.
- Raw brand values are sanitized and bounded before reaching FFmpeg or ASS rendering.
- Downgrades preserve workspace data and block only new over-limit activity.

## Module boundaries

- `app/director`: Director Contract compiler, production-style compiler, editorial agents, cleanup, and decision graphs
- `app/sensory`: transcription, scenes, framing, reference fingerprints, music profiles, objects, motion, quality, and multi-clip analysis
- `app/rendering`: captions, deterministic timeline compilation, overlays, audio mixing, and FFmpeg execution
- `app/quality`: technical, editorial, brand, and instruction-compliance checks
- `app/services/billing.py`: credit reservations, usage ledger, releases, and settlements
- `app/services/subscriptions.py`: hosted billing sessions, signed provider events, subscription state, and recurring credit grants
- `app/services/entitlements.py`: plan catalog and duration, tier, clip, and seat enforcement
- `app/services/storage.py`: multipart providers, object verification, provenance, and purge jobs
- `app/director/camera.py`: missing-shot missions, capture guidance, readiness, and continuity verification
