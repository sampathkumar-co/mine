from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings
from app.sensory.models import TranscriptResult, TranscriptSegment, TranscriptWord


class TranscriptionError(RuntimeError):
    pass


def parse_verbose_transcription(
    payload: dict[str, Any], *, provider: str, model: str
) -> TranscriptResult:
    words = [
        TranscriptWord(
            word=str(item.get("word", "")).strip(),
            start=max(0.0, float(item.get("start", 0))),
            end=max(0.0, float(item.get("end", 0))),
        )
        for item in payload.get("words", [])
        if str(item.get("word", "")).strip()
    ]
    segments = [
        TranscriptSegment(
            start=max(0.0, float(item.get("start", 0))),
            end=max(0.0, float(item.get("end", 0))),
            text=str(item.get("text", "")).strip(),
            confidence=0.8,
        )
        for item in payload.get("segments", [])
        if str(item.get("text", "")).strip()
    ]
    duration = float(payload.get("duration", 0) or 0)
    if duration <= 0 and words:
        duration = max(word.end for word in words)
    if duration <= 0 and segments:
        duration = max(segment.end for segment in segments)

    return TranscriptResult(
        text=str(payload.get("text", "")).strip(),
        language=payload.get("language"),
        duration_seconds=max(0.0, duration),
        provider=provider,
        model=model,
        words=words,
        segments=segments,
    )


def transcribe_audio(audio_path: str | Path, settings: Settings) -> TranscriptResult:
    provider = settings.transcription_provider.strip().casefold()
    if provider != "openai":
        raise TranscriptionError(f"Unsupported transcription provider: {provider}")
    if not settings.openai_api_key:
        raise TranscriptionError("DIRECTOR_OPENAI_API_KEY is required for transcription")

    endpoint = f"{settings.openai_base_url.rstrip('/')}/audio/transcriptions"
    with Path(audio_path).open("rb") as audio_file:
        files = {"file": (Path(audio_path).name, audio_file, "audio/wav")}
        data = {
            "model": settings.transcription_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
            "temperature": "0",
        }
        try:
            response = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                files=files,
                data=data,
                timeout=settings.transcription_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            error_response = getattr(exc, "response", None)
            detail = getattr(error_response, "text", "") or str(exc)
            raise TranscriptionError(detail[-2_000:]) from exc

    payload = response.json()
    if not isinstance(payload, dict):
        raise TranscriptionError("Transcription provider returned an invalid response")
    return parse_verbose_transcription(
        payload,
        provider=provider,
        model=settings.transcription_model,
    )
