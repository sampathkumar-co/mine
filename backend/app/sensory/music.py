from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from app.core.config import Settings
from app.sensory.models import MusicProfile

MEAN_VOLUME = re.compile(r"mean_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB")
PEAK_VOLUME = re.compile(r"max_volume:\s*(-?[0-9]+(?:\.[0-9]+)?)\s*dB")


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


def _energy_from_volume(mean_volume_db: float, filename: str) -> float:
    energy = (mean_volume_db + 34) / 24
    normalized_name = filename.casefold()
    if any(term in normalized_name for term in {"calm", "ambient", "soft", "piano"}):
        energy -= 0.18
    if any(term in normalized_name for term in {"upbeat", "energetic", "dance", "rock", "fast"}):
        energy += 0.18
    return min(1.0, max(0.0, energy))


def analyze_music(
    path: str | Path,
    *,
    asset_id: str,
    filename: str,
    settings: Settings,
) -> MusicProfile:
    probe = _run(
        [
            settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            str(path),
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
            str(path),
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

    return MusicProfile(
        asset_id=asset_id,
        filename=filename,
        duration_seconds=max(0.0, duration),
        mean_volume_db=mean_volume,
        peak_volume_db=peak_volume,
        energy=round(_energy_from_volume(mean_volume, filename), 3),
    )


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
            -profile.duration_seconds,
            profile.filename.casefold(),
        ),
    )
