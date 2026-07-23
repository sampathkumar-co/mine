import re
import subprocess
from pathlib import Path

from app.core.config import Settings
from app.sensory.models import SceneRange

PTS_TIME_PATTERN = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


class SceneDetectionError(RuntimeError):
    pass


def detect_scenes(
    source_path: str | Path,
    *,
    duration_seconds: float,
    settings: Settings,
) -> list[SceneRange]:
    if duration_seconds <= 0:
        return []

    command = [
        settings.ffmpeg_binary,
        "-hide_banner",
        "-i",
        str(source_path),
        "-filter:v",
        f"select='gt(scene,{settings.scene_detection_threshold})',showinfo",
        "-an",
        "-vsync",
        "vfr",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=min(settings.render_timeout_seconds, 600),
        )
    except subprocess.TimeoutExpired as exc:
        raise SceneDetectionError("Scene detection timed out") from exc

    if result.returncode not in {0, 255}:
        raise SceneDetectionError(result.stderr[-4_000:])

    boundaries = [0.0]
    for match in PTS_TIME_PATTERN.finditer(result.stderr):
        timestamp = float(match.group(1))
        if 0.25 < timestamp < duration_seconds - 0.25:
            boundaries.append(timestamp)
    boundaries.append(duration_seconds)
    boundaries = sorted(set(round(value, 3) for value in boundaries))

    scenes: list[SceneRange] = []
    for start, end in zip(boundaries, boundaries[1:], strict=True):
        if end - start >= settings.minimum_scene_seconds:
            scenes.append(SceneRange(start=start, end=end, confidence=0.72))

    if not scenes:
        scenes.append(SceneRange(start=0, end=duration_seconds, confidence=0.5))
    return scenes
