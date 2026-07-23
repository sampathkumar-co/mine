import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings


class MediaCommandError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MediaProbe:
    duration_seconds: float
    width: int
    height: int
    video_codec: str
    has_audio: bool


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
        raise MediaCommandError(stderr[-4_000:]) from exc


def probe_media(path: str | Path, settings: Settings) -> MediaProbe:
    result = _run(
        [
            settings.ffprobe_binary,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            str(path),
        ],
        timeout_seconds=min(settings.render_timeout_seconds, 120),
    )
    payload: dict[str, Any] = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise MediaCommandError("No video stream found")

    return MediaProbe(
        duration_seconds=float(payload.get("format", {}).get("duration", 0)),
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        video_codec=str(video_stream.get("codec_name", "unknown")),
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
    )


def render_vertical_baseline(
    source_path: str | Path,
    output_path: str | Path,
    settings: Settings,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            settings.ffmpeg_binary,
            "-y",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-vf",
            (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
            ),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            str(output),
        ],
        timeout_seconds=settings.render_timeout_seconds,
    )


def validate_vertical_output(probe: MediaProbe) -> None:
    if probe.duration_seconds <= 0:
        raise MediaCommandError("Rendered output has no measurable duration")
    if (probe.width, probe.height) != (1080, 1920):
        raise MediaCommandError(
            f"Rendered output dimensions are {probe.width}x{probe.height}, expected 1080x1920"
        )
