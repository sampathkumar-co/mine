# Director OS Web Client

The web client is a responsive Next.js application for the first complete operator workflow:

1. Create a Director Contract.
2. Select Director Camera mode and readiness threshold.
3. Upload one or more source videos.
4. Start autonomous production.
5. Poll project and camera state without blocking the browser.
6. Inspect Production Readiness dimensions and pickup history.
7. Open a pickup mission in guided capture.
8. Record, review, retake, and submit footage.
9. Resume validation or explicitly override a required gate.

## Local development

The full stack runs with:

```bash
cp .env.example .env
docker compose up --build
```

Open `http://localhost:3000`. The frontend calls `http://localhost:8000/api/v1` by default. A standalone frontend session can use:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

## Guided capture

The capture client uses browser-native APIs rather than a proprietary camera SDK:

- `navigator.mediaDevices.getUserMedia` for camera and microphone access
- `MediaRecorder` for WebM pickup recording
- `AudioContext` and `AnalyserNode` for a live speech-level indicator
- downsampled canvas frames for an approximate light and stability signal
- `DeviceOrientationEvent` for a best-effort horizon/level signal
- a frozen camera frame for manual ghost-frame alignment
- CSS overlays for 9:16 safe zones, thirds, and eye-line guidance

The browser guidance is advisory. The backend remains authoritative and validates the submitted file against the mission after upload.

## Security requirements

Camera and microphone access require a secure browser context. `localhost` is treated as secure by modern browsers during local development. A deployed client must use HTTPS. Permission denial is handled in the capture modal and does not alter the project.

`NEXT_PUBLIC_DIRECTOR_API_URL` is compiled into the browser bundle. Set it to the externally reachable API URL during the image build. Do not place secrets in any `NEXT_PUBLIC_*` variable.

## Current limitations

- The ghost frame is captured manually from the live camera; the backend does not yet return a frame from the continuity reference clip.
- Light and stability meters are practical heuristics, not calibrated photometry or optical stabilization analysis.
- Device orientation is unavailable on some desktops and may require an additional permission gesture on iOS.
- Browser recording format depends on `MediaRecorder` support. The backend accepts the resulting video MIME type and FFmpeg handles normalization.
- Authentication and multi-account project lists remain future milestones.
