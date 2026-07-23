from __future__ import annotations

import re
from pathlib import Path

from app.sensory.models import ClipAnalysis, SceneRange, SubjectFraming, TranscriptResult

EVIDENCE_HINTS = {
    "before",
    "after",
    "chart",
    "dashboard",
    "demo",
    "evidence",
    "proof",
    "product",
    "result",
    "screen",
    "screenshot",
}
BROLL_HINTS = {"broll", "b-roll", "detail", "establishing", "insert", "room", "wide"}


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", value.casefold()))


def compute_perceptual_hash(path: str | Path) -> str | None:
    """Return a 64-bit difference hash from the middle readable frame."""
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return None
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count > 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        bits = resized[:, 1:] > resized[:, :-1]
        value = 0
        for bit in bits.flatten():
            value = (value << 1) | int(bit)
        return f"{value:016x}"
    except Exception:
        return None


def hamming_distance(left: str | None, right: str | None) -> int | None:
    if not left or not right or len(left) != len(right):
        return None
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return None


def analyze_source_clip(
    *,
    asset_id: str,
    filename: str,
    sha256: str,
    media: dict[str, object],
    transcript: TranscriptResult | None,
    scenes: list[SceneRange],
    subject_framing: SubjectFraming,
    perceptual_hash: str | None,
) -> ClipAnalysis:
    duration = float(media.get("duration_seconds", 0) or 0)
    width = int(media.get("width", 0) or 0)
    height = int(media.get("height", 0) or 0)
    has_audio = bool(media.get("has_audio", False))
    transcript_words = len(transcript.words) if transcript else 0
    speech_density = transcript_words / max(duration, 1.0)

    filename_terms = _terms(Path(filename).stem)
    transcript_terms = _terms(transcript.text if transcript else "")
    evidence_terms = sorted((filename_terms | transcript_terms) & EVIDENCE_HINTS)

    role = "primary_speech"
    if filename_terms & EVIDENCE_HINTS:
        role = "evidence"
    elif filename_terms & BROLL_HINTS or not has_audio:
        role = "b_roll"
    elif transcript is None or speech_density < 0.25:
        role = "b_roll"

    score = 0.48
    rejection_reasons: list[str] = []
    if duration >= 1.0:
        score += 0.08
    else:
        score -= 0.28
        rejection_reasons.append("Clip is shorter than one second.")
    if width >= 720 and height >= 720:
        score += 0.1
    elif width < 480 or height < 480:
        score -= 0.2
        rejection_reasons.append("Clip resolution is too low for a clean vertical output.")
    if has_audio and transcript_words:
        score += min(0.2, speech_density * 0.08)
    if subject_framing.confidence >= 0.35:
        score += 0.06
    if len(scenes) > 1:
        score += min(0.06, (len(scenes) - 1) * 0.01)
    if role in {"b_roll", "evidence"}:
        score += 0.04
    if duration <= 0:
        rejection_reasons.append("Clip has no measurable duration.")
        score = 0

    score = min(1.0, max(0.0, score))
    if score < 0.2 or duration <= 0:
        role = "rejected"

    return ClipAnalysis(
        asset_id=asset_id,
        filename=filename,
        sha256=sha256,
        media=media,
        transcript=transcript,
        scenes=scenes,
        subject_framing=subject_framing,
        perceptual_hash=perceptual_hash,
        role=role,
        quality_score=round(score, 3),
        rejection_reasons=rejection_reasons,
        evidence_terms=evidence_terms,
    )


def mark_duplicate_clips(
    clips: list[ClipAnalysis],
    *,
    perceptual_distance_threshold: int = 6,
) -> list[ClipAnalysis]:
    accepted: list[ClipAnalysis] = []
    result: list[ClipAnalysis] = []
    for clip in clips:
        duplicate_of: str | None = None
        for previous in accepted:
            exact_match = bool(clip.sha256 and clip.sha256 == previous.sha256)
            distance = hamming_distance(clip.perceptual_hash, previous.perceptual_hash)
            visual_match = distance is not None and distance <= perceptual_distance_threshold
            if exact_match or visual_match:
                duplicate_of = previous.asset_id
                break

        if duplicate_of:
            result.append(
                clip.model_copy(
                    update={
                        "duplicate_of_asset_id": duplicate_of,
                        "role": "rejected",
                        "quality_score": min(clip.quality_score, 0.1),
                        "rejection_reasons": [
                            *clip.rejection_reasons,
                            f"Duplicate or near-duplicate of source asset {duplicate_of}.",
                        ],
                    }
                )
            )
        else:
            accepted.append(clip)
            result.append(clip)
    return result
