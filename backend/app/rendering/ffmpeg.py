import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.director.edit_graph import EditDecisionGraph


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


def extract_transcription_audio(
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
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        timeout_seconds=settings.render_timeout_seconds,
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


def render_edit_decision_graph(
    source_path: str | Path,
    output_path: str | Path,
    graph: EditDecisionGraph,
    *,
    has_audio: bool,
    settings: Settings,
) -> None:
    if not graph.segments:
        raise MediaCommandError("Edit Decision Graph contains no renderable segments")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    concat_inputs: list[str] = []
    scale = (
        "scale=1080:1920:force_original_aspect_ratio=decrease,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )

    for index, segment in enumerate(graph.segments):
        filters.append(
            f"[0:v]trim=start={segment.source_start}:end={segment.source_end},"
            f"setpts=PTS-STARTPTS,{scale}[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if has_audio:
            filters.append(
                f"[0:a]atrim=start={segment.source_start}:end={segment.source_end},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")

    audio_count = 1 if has_audio else 0
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(graph.segments)}:v=1:a={audio_count}[vout]"
        + ("[aout]" if has_audio else "")
    )
    command = [
        settings.ffmpeg_binary,
        "-y",
        "-i",
        str(source_path),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
    ]
    if has_audio:
        command.extend(["-map", "[aout]", "-c:a", "aac", "-b:a", "160k"])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run(command, timeout_seconds=settings.render_timeout_seconds)


def validate_vertical_output(probe: MediaProbe) -> None:
    if probe.duration_seconds <= 0:
        raise MediaCommandError("Rendered output has no measurable duration")
    if (probe.width, probe.height) != (1080, 1920):
        raise MediaCommandError(
            f"Rendered output dimensions are {probe.width}x{probe.height}, expected 1080x1920"
        )
