from __future__ import annotations

from pathlib import Path
from statistics import median

from app.sensory.models import SubjectFraming


def detect_primary_subject(
    source_path: str | Path,
    *,
    max_samples: int = 24,
) -> SubjectFraming:
    """Estimate a stable horizontal subject position from sampled face detections.

    Detection failures intentionally return a centred, low-confidence fallback so
    rendering remains deterministic on footage without visible faces.
    """
    try:
        import cv2
    except ImportError:
        return SubjectFraming(detector="center_fallback")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        return SubjectFraming(detector="center_fallback")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        if frame_count <= 0 or fps <= 0:
            return SubjectFraming(detector="center_fallback")

        sample_count = max(1, min(max_samples, frame_count))
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return SubjectFraming(detector="center_fallback")

        centres: list[float] = []
        attempted = 0
        for index in range(sample_count):
            frame_index = round(index * (frame_count - 1) / max(1, sample_count - 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            attempted += 1
            height, width = frame.shape[:2]
            if width <= 0 or height <= 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            if len(faces) == 0:
                continue
            x, _y, face_width, _face_height = max(
                faces,
                key=lambda face: int(face[2]) * int(face[3]),
            )
            centres.append((float(x) + float(face_width) / 2) / float(width))

        if not centres or attempted == 0:
            return SubjectFraming(detector="center_fallback")

        confidence = min(1.0, len(centres) / attempted)
        return SubjectFraming(
            normalized_center_x=min(0.95, max(0.05, float(median(centres)))),
            confidence=round(confidence, 3),
            detector="opencv_haar_face",
            sampled_frames=attempted,
            detected_frames=len(centres),
        )
    finally:
        capture.release()
