from __future__ import annotations

import re
from pathlib import Path

from app.sensory.models import (
    ContinuityProfile,
    SemanticTag,
    SubjectFraming,
    TranscriptResult,
)

SEMANTIC_GROUPS: dict[str, set[str]] = {
    "product": {"product", "package", "device", "phone", "laptop", "bottle", "box", "item"},
    "screen": {"screen", "screenshot", "dashboard", "website", "app", "software", "analytics"},
    "document": {"document", "report", "contract", "paper", "invoice", "certificate", "receipt"},
    "chart": {"chart", "graph", "metric", "growth", "revenue", "result", "percentage"},
    "before_after": {"before", "after", "transformation", "comparison"},
    "demo": {"demo", "demonstration", "show", "showing", "workflow", "process", "steps"},
    "location": {"room", "office", "house", "property", "kitchen", "bedroom", "street", "store"},
    "testimonial": {"testimonial", "review", "customer", "client", "feedback"},
    "proof": {"proof", "evidence", "verified", "data", "result", "results"},
}


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", value.casefold()))


def _deduplicate(tags: list[SemanticTag]) -> list[SemanticTag]:
    best: dict[str, SemanticTag] = {}
    for tag in tags:
        existing = best.get(tag.label)
        if existing is None or tag.confidence > existing.confidence:
            best[tag.label] = tag
    return sorted(best.values(), key=lambda item: (-item.confidence, item.label))


def extract_text_semantic_tags(
    filename: str,
    transcript: TranscriptResult | None,
) -> list[SemanticTag]:
    filename_tokens = _tokens(Path(filename).stem)
    transcript_tokens = _tokens(transcript.text if transcript else "")
    tags: list[SemanticTag] = []

    for label, hints in SEMANTIC_GROUPS.items():
        filename_hits = sorted(filename_tokens & hints)
        transcript_hits = sorted(transcript_tokens & hints)
        if filename_hits:
            tags.append(
                SemanticTag(
                    label=label,
                    confidence=min(0.94, 0.72 + len(filename_hits) * 0.06),
                    source="filename",
                    evidence=", ".join(filename_hits[:4]),
                )
            )
        if transcript_hits:
            tags.append(
                SemanticTag(
                    label=label,
                    confidence=min(0.9, 0.58 + len(transcript_hits) * 0.05),
                    source="transcript",
                    evidence=", ".join(transcript_hits[:4]),
                )
            )

    transcript_text = transcript.text if transcript else ""
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|x|times|₹|\$|€)?\b", transcript_text):
        tags.append(
            SemanticTag(
                label="measurable_claim",
                confidence=0.78,
                source="transcript",
                evidence="Transcript contains a measurable value.",
            )
        )
    return _deduplicate(tags)


def analyze_visual_semantics(
    path: str | Path,
    *,
    media: dict[str, object],
    subject_framing: SubjectFraming,
    max_samples: int = 10,
) -> tuple[list[SemanticTag], ContinuityProfile]:
    width = int(media.get("width", 0) or 0)
    height = int(media.get("height", 0) or 0)
    aspect_ratio = width / height if width > 0 and height > 0 else 9 / 16
    tags: list[SemanticTag] = []
    brightness_values: list[float] = []
    saturation_values: list[float] = []
    motion_values: list[float] = []
    edge_values: list[float] = []

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_count = min(max_samples, max(1, frame_count))
        indices = [round(index * max(0, frame_count - 1) / max(1, sample_count - 1)) for index in range(sample_count)]
        previous_gray = None
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            brightness_values.append(float(hsv[:, :, 2].mean() / 255))
            saturation_values.append(float(hsv[:, :, 1].mean() / 255))
            edges = cv2.Canny(gray, 70, 150)
            edge_values.append(float((edges > 0).mean()))
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                motion_values.append(float(difference.mean() / 255))
            previous_gray = gray
        capture.release()
    except Exception:
        pass

    brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0.5
    saturation = sum(saturation_values) / len(saturation_values) if saturation_values else 0.5
    motion = sum(motion_values) / len(motion_values) if motion_values else 0.15
    edge_density = sum(edge_values) / len(edge_values) if edge_values else 0.04

    if subject_framing.confidence >= 0.25:
        tags.append(
            SemanticTag(
                label="person",
                confidence=min(0.92, 0.55 + subject_framing.confidence * 0.4),
                source="vision_heuristic",
                evidence="A stable face/subject was detected across sampled frames.",
            )
        )
    if saturation < 0.35 and edge_density > 0.08:
        tags.append(
            SemanticTag(
                label="screen_or_document",
                confidence=min(0.82, 0.55 + edge_density),
                source="vision_heuristic",
                evidence="Low saturation and dense rectangular edges suggest a screen or document.",
            )
        )
    if aspect_ratio > 1.25 and subject_framing.confidence < 0.25:
        tags.append(
            SemanticTag(
                label="environment",
                confidence=0.64,
                source="vision_heuristic",
                evidence="Wide composition without a stable face suggests environmental coverage.",
            )
        )
    if motion > 0.08:
        tags.append(
            SemanticTag(
                label="motion_demo",
                confidence=min(0.86, 0.55 + motion * 2),
                source="vision_heuristic",
                evidence="Sampled frames contain sustained visual movement.",
            )
        )
    if edge_density > 0.12 and subject_framing.confidence < 0.2:
        tags.append(
            SemanticTag(
                label="detail_or_product",
                confidence=min(0.78, 0.5 + edge_density),
                source="vision_heuristic",
                evidence="Dense local detail without a dominant face suggests an insert or product shot.",
            )
        )

    profile = ContinuityProfile(
        brightness=round(min(1.0, max(0.0, brightness)), 3),
        saturation=round(min(1.0, max(0.0, saturation)), 3),
        motion_energy=round(min(1.0, max(0.0, motion * 3)), 3),
        subject_center_x=subject_framing.normalized_center_x,
        aspect_ratio=round(max(0.01, aspect_ratio), 3),
        sampled_frames=len(brightness_values),
    )
    return _deduplicate(tags), profile


def semantic_terms(tags: list[SemanticTag]) -> set[str]:
    terms: set[str] = set()
    for tag in tags:
        terms.update(_tokens(tag.label.replace("_", " ")))
        if tag.evidence:
            terms.update(_tokens(tag.evidence))
    return terms


def continuity_similarity(left: ContinuityProfile, right: ContinuityProfile) -> float:
    brightness = 1 - min(1.0, abs(left.brightness - right.brightness) / 0.6)
    saturation = 1 - min(1.0, abs(left.saturation - right.saturation) / 0.7)
    subject = 1 - min(1.0, abs(left.subject_center_x - right.subject_center_x) / 0.55)
    aspect = 1 - min(1.0, abs(left.aspect_ratio - right.aspect_ratio) / 1.2)
    motion = 1 - min(1.0, abs(left.motion_energy - right.motion_energy) / 0.8)
    score = brightness * 0.25 + saturation * 0.2 + subject * 0.25 + aspect * 0.15 + motion * 0.15
    return round(min(1.0, max(0.0, score)), 3)
