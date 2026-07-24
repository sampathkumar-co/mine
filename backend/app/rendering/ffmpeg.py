import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.director.edit_graph import EditDecisionGraph
from app.director.style import ProductionStyle


class MediaCommandError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MediaProbe:
    duration_seconds: float
    width: int
    height: int
    video_codec: str
    has_audio: bool


@dataclass(frozen=True, slots=True)
class RenderSource:
    asset_id: str
    path: str
    probe: MediaProbe
    subject_center_x: float = 0.5


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


def _vertical_filter(probe: MediaProbe, subject_center_x: float) -> str:
    if probe.width <= 0 or probe.height <= 0:
        raise MediaCommandError("Source dimensions are unavailable")

    target_ratio = 9 / 16
    source_ratio = probe.width / probe.height
    if source_ratio > target_ratio:
        crop_width = min(probe.width, round(probe.height * target_ratio))
        crop_width -= crop_width % 2
        maximum_x = max(0, probe.width - crop_width)
        crop_x = round(subject_center_x * probe.width - crop_width / 2)
        crop_x = min(maximum_x, max(0, crop_x))
        crop_x -= crop_x % 2
        return (
            f"crop={crop_width}:{probe.height}:{crop_x}:0,"
            "scale=1080:1920:flags=lanczos,setsar=1"
        )

    return (
        "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
        "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1"
    )


def _escape_filter_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


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


def _music_filter(
    *,
    input_index: int,
    duration_seconds: float,
    style: ProductionStyle,
) -> str:
    filters = [
        f"[{input_index}:a]atrim=duration={duration_seconds:.3f}",
        "asetpts=PTS-STARTPTS",
    ]
    fade = min(style.music.fade_seconds, max(0.0, duration_seconds / 3))
    if fade > 0.01:
        filters.append(f"afade=t=in:st=0:d={fade:.3f}")
        fade_out_start = max(0.0, duration_seconds - fade)
        filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade:.3f}")
    filters.append(f"volume={style.music.volume:.4f}[musicbed]")
    return ",".join(filters)


def _source_for_segment(
    segment,
    sources: list[RenderSource],
) -> tuple[int, RenderSource]:
    by_asset_id = {source.asset_id: (index, source) for index, source in enumerate(sources)}
    if segment.source_asset_id and segment.source_asset_id in by_asset_id:
        return by_asset_id[segment.source_asset_id]
    if 0 <= segment.source_index < len(sources):
        return segment.source_index, sources[segment.source_index]
    return 0, sources[0]


def render_multiclip_edit_decision_graph(
    sources: list[RenderSource],
    output_path: str | Path,
    graph: EditDecisionGraph,
    *,
    caption_path: str | Path | None = None,
    music_path: str | Path | None = None,
    style: ProductionStyle | None = None,
    settings: Settings,
) -> None:
    if not sources:
        raise MediaCommandError("No render sources were supplied")
    if not graph.segments:
        raise MediaCommandError("Edit Decision Graph contains no renderable segments")

    production_style = style or ProductionStyle()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    filters: list[str] = []
    concat_inputs: list[str] = []
    visual_finish = (
        f"eq=contrast={production_style.visual.contrast:.4f}:"
        f"saturation={production_style.visual.saturation:.4f}:"
        f"brightness={production_style.visual.brightness:.4f}"
    )

    for index, segment in enumerate(graph.segments):
        source_index, source = _source_for_segment(segment, sources)
        vertical_filter = _vertical_filter(source.probe, source.subject_center_x)
        duration = max(0.01, segment.source_end - segment.source_start)
        filters.append(
            f"[{source_index}:v]trim=start={segment.source_start}:end={segment.source_end},"
            f"setpts=PTS-STARTPTS,{vertical_filter},{visual_finish}[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if source.probe.has_audio:
            filters.append(
                f"[{source_index}:a]atrim=start={segment.source_start}:end={segment.source_end},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        else:
            filters.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        concat_inputs.append(f"[a{index}]")

    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(graph.segments)}:v=1:a=1[vconcat][aconcat]"
    )

    if caption_path is not None and Path(caption_path).exists():
        escaped_caption_path = _escape_filter_path(caption_path)
        filters.append(f"[vconcat]ass=filename='{escaped_caption_path}'[vout]")
    else:
        filters.append("[vconcat]null[vout]")

    command = [settings.ffmpeg_binary, "-y"]
    for source in sources:
        command.extend(["-i", source.path])

    has_music = music_path is not None and production_style.music.enabled
    music_input_index: int | None = None
    if has_music:
        music_input_index = len(sources)
        command.extend(["-stream_loop", "-1", "-i", str(music_path)])

    filters.append(
        "[aconcat]highpass=f=80,lowpass=f=12000,afftdn=nf=-25,"
        "loudnorm=I=-16:TP=-1.5:LRA=11[voiceclean]"
    )

    if has_music and music_input_index is not None:
        filters.append(
            _music_filter(
                input_index=music_input_index,
                duration_seconds=graph.selected_duration_seconds,
                style=production_style,
            )
        )
        filters.extend(
            [
                "[voiceclean]asplit=2[voicekey][voicemix]",
                (
                    "[musicbed][voicekey]sidechaincompress="
                    f"threshold={production_style.music.ducking_threshold:.4f}:"
                    "ratio=10:attack=20:release=350[ducked]"
                ),
                (
                    "[voicemix][ducked]amix=inputs=2:duration=first:normalize=0,"
                    "loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
                ),
            ]
        )
    else:
        filters.append("[voiceclean]anull[aout]")

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-t",
            f"{graph.selected_duration_seconds:.3f}",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run(command, timeout_seconds=settings.render_timeout_seconds)


def render_edit_decision_graph(
    source_path: str | Path,
    output_path: str | Path,
    graph: EditDecisionGraph,
    *,
    media_probe: MediaProbe,
    subject_center_x: float = 0.5,
    caption_path: str | Path | None = None,
    music_path: str | Path | None = None,
    style: ProductionStyle | None = None,
    settings: Settings,
) -> None:
    source_asset_id = graph.segments[0].source_asset_id or "source-0"
    render_multiclip_edit_decision_graph(
        [
            RenderSource(
                asset_id=source_asset_id,
                path=str(source_path),
                probe=media_probe,
                subject_center_x=subject_center_x,
            )
        ],
        output_path,
        graph,
        caption_path=caption_path,
        music_path=music_path,
        style=style,
        settings=settings,
    )


def validate_vertical_output(probe: MediaProbe, *, expect_audio: bool | None = None) -> None:
    if probe.duration_seconds <= 0:
        raise MediaCommandError("Rendered output has no measurable duration")
    if (probe.width, probe.height) != (1080, 1920):
        raise MediaCommandError(
            f"Rendered output dimensions are {probe.width}x{probe.height}, expected 1080x1920"
        )
    if expect_audio is True and not probe.has_audio:
        raise MediaCommandError("Rendered output unexpectedly lost its audio stream")
