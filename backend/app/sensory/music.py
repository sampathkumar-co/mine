from __future__ import annotations

import json
import re
import subprocess
import sys
from array import array
from pathlib import Path

from app.core.config import Settings
from app.director.rhythm import estimate_rhythm_from_envelope, pcm_energy_envelope
from app.sensory.models import MusicProfile

MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB")
PEAK_VOLUME = re.compile(r"max_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB")
_RHYTHM_SAMPLE_RATE = 8_000
_RHYTHM_FRAMES_PER_SECOND = 50
_RHYTHM_ANALYSIS_SECONDS = 600
_PROFILE_CACHE: dict[tuple[str, int, int, str, str], MusicProfile] = {}


class MusicAnalysisError(RuntimeError):
    pass


def _run(command: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", None) or str(exc)
        raise MusicAnalysisError(stderr[-2_000:]) from exc


def _run_bytes(command: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", None)
        if isinstance(stderr, bytes):
            message = stderr.decode("utf-8", errors="replace")
        else:
            message = str(stderr or exc)
        raise MusicAnalysisError(message[-2_000:]) from exc


def _energy_from_volume(mean_volume_db: float, filename: str) -> float:
    energy = (mean_volume_db + 34) / 24
    normalized_name = filename.casefold()
    if any(term in normalized_name for term in {"calm", "ambient", "soft", "piano"}):
        energy -= 0.18
    if any(term in normalized_name for term in {"upbeat", "energetic", "dance", "rock", "fast"}):
        energy += 0.18
    return min(1.0, max(0.0, energy))


def _cache_key(path: Path, *, asset_id: str, filename: str) -> tuple[str, int, int, str, str]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns, asset_id, filename


def _estimate_music_rhythm(
    path: Path,
    *,
    duration_seconds: float,
    settings: Settings,
):
    analysis_duration = min(max(0.0, duration_seconds), _RHYTHM_ANALYSIS_SECONDS)
    if analysis_duration < 4:
        return estimate_rhythm_from_envelope(
            [],
            frames_per_second=_RHYTHM_FRAMES_PER_SECOND,
            duration_seconds=duration_seconds,
        )
    result = _run_bytes(
        [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-t",
            f"{analysis_duration:.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(_RHYTHM_SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        ],
        timeout_seconds=min(settings.render_timeout_seconds, 900),
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    envelope = pcm_energy_envelope(
        samples.tolist(),
        sample_rate=_RHYTHM_SAMPLE_RATE,
        frames_per_second=_RHYTHM_FRAMES_PER_SECOND,
    )
    return estimate_rhythm_from_envelope(
        envelope,
        frames_per_second=_RHYTHM_FRAMES_PER_SECOND,
        duration_seconds=duration_seconds,
    )


def analyze_music(
    path: str | Path,
    *,
    asset_id: str,
    filename: str,
    settings: Settings,
) -> MusicProfile:
    music_path = Path(path)
    cache_key = _cache_key(music_path, asset_id=asset_id, filename=filename)
    cached = _PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached.model_copy(deep=True)

    probe = _run(
        [
            settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(music_path),
        ],
        timeout_seconds=min(settings.render_timeout_seconds, 120),
    )
    payload = json.loads(probe.stdout)
    streams = payload.get("streams", [])
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise MusicAnalysisError("Uploaded music asset has no audio stream")
    duration = float(payload.get("format", {}).get("duration", 0) or 0)

    volume = _run(
        [
            settings.ffmpeg_binary,
            "-hide_banner",
            "-i",
            str(music_path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        timeout_seconds=min(settings.render_timeout_seconds, 600),
    )
    mean_match = MEAN_VOLUME.search(volume.stderr)
    peak_match = PEAK_VOLUME.search(volume.stderr)
    mean_volume = float(mean_match.group(1)) if mean_match else -24.0
    peak_volume = float(peak_match.group(1)) if peak_match else -6.0

    try:
        rhythm = _estimate_music_rhythm(
            music_path,
            duration_seconds=duration,
            settings=settings,
        )
    except MusicAnalysisError:
        rhythm = estimate_rhythm_from_envelope(
            [],
            frames_per_second=_RHYTHM_FRAMES_PER_SECOND,
            duration_seconds=duration,
        )

    profile = MusicProfile(
        asset_id=asset_id,
        filename=filename,
        duration_seconds=max(0.0, duration),
        mean_volume_db=mean_volume,
        peak_volume_db=peak_volume,
        energy=round(_energy_from_volume(mean_volume, filename), 3),
        tempo_bpm=rhythm.tempo_bpm,
        beat_interval_seconds=rhythm.beat_interval_seconds,
        beat_offset_seconds=rhythm.beat_offset_seconds,
        beat_times=list(rhythm.beat_times),
        phrase_times=list(rhythm.phrase_times),
        rhythm_confidence=rhythm.confidence,
    )
    _PROFILE_CACHE[cache_key] = profile
    return profile.model_copy(deep=True)


def choose_music(
    profiles: list[MusicProfile],
    *,
    desired_energy: float,
) -> MusicProfile | None:
    if not profiles:
        return None
    desired = min(1.0, max(0.0, desired_energy))
    return min(
        profiles,
        key=lambda profile: (
            abs(profile.energy - desired),
            -profile.rhythm_confidence,
            -profile.duration_seconds,
            profile.filename.casefold(),
        ),
    )
