# Director Camera and Production Readiness

Director Camera audits whether the uploaded footage is sufficient for the Director Contract before the final edit is rendered.

## Contract modes

```json
{
  "director_camera_mode": "required",
  "production_readiness_threshold": 0.72
}
```

- `off` skips the camera audit.
- `advisory` records readiness and pickup missions but allows rendering to continue.
- `required` pauses the project in `needs_pickups` when readiness is below the threshold or a blocking requirement is missing.

Explicit current-project requirements remain higher priority than Director Memory or camera defaults.

## Readiness dimensions

The audit produces a weighted score from:

- narrative coverage and opening clarity
- usable primary audio
- visual quality and subject framing
- footage duration and B-roll coverage
- evidence/proof coverage required by the objective
- cross-clip lighting, framing, colour, and motion continuity

Every dimension includes findings and a blocking flag. The report and all pickup missions are stored in PostgreSQL and exposed through:

```text
GET /api/v1/projects/{project_id}/director-camera
```

## Pickup missions

Director Camera can request:

- a direct opening hook
- a clear call to action
- visible evidence or product proof
- supporting B-roll
- a clean primary audio retake
- a continuity bridge shot

Each mission includes target terms, duration range, whether audio is required, framing, light, stability, safe-zone, continuity, and privacy guidance.

Upload a take against its mission:

```bash
curl -X POST \
  http://localhost:8000/api/v1/projects/<PROJECT_ID>/director-camera/missions/<MISSION_ID>/submit \
  -F 'file=@./pickup.mp4;type=video/mp4'
```

Then resume validation and production:

```bash
curl -X POST \
  http://localhost:8000/api/v1/projects/<PROJECT_ID>/director-camera/resume
```

## Validation

A submitted pickup is checked for:

- requested duration
- duplicate footage
- minimum clip quality
- required audio stream
- transcript availability when speech is expected
- semantic match to the mission
- continuity with existing footage

The validation result stores its score, blocking reasons, warnings, matched terms, and continuity score. Rejected missions can receive a new submission.

## Automatic insertion

Accepted pickups are promoted into the production analysis:

- hook pickups are prepended to the narration story when not naturally selected
- CTA pickups are appended
- evidence, B-roll, and continuity pickups are scheduled as narration-preserving overlays
- clean primary retakes become eligible primary-speech sources

Every insertion is recorded in the Edit Decision Graph notes and is rechecked by the editorial critic before rendering.

## Advisory improvement pass

Advisory missions remain actionable after a project is ready. Submitting an optional pickup moves the project to `needs_pickups`; resuming creates a new production pass while the previous output path remains stored until the replacement succeeds.

## Override

A required-mode gate can be explicitly overridden:

```bash
curl -X POST \
  http://localhost:8000/api/v1/projects/<PROJECT_ID>/director-camera/override \
  -H 'Content-Type: application/json' \
  -d '{"reason":"Publish the current cut for the scheduled campaign."}'
```

The reason and timestamp are stored inside the Director Contract and written into the graph notes. An override never deletes the audit or pickup missions.
