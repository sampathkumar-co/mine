from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.director.edit_graph import EditDecisionGraph, EditSegment
from app.director.style import CaptionStyle, hex_to_ass
from app.sensory.models import TranscriptResult, TranscriptWord

TranscriptCollection = TranscriptResult | Mapping[str, TranscriptResult] | None


@dataclass(frozen=True, slots=True)
class CaptionCue:
    start: float
    end: float
    text: str


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{cs:02d}"


def _escape_ass_text(value: str) -> str:
    line_break_token = "__DIRECTOR_LINE_BREAK__"
    escaped = value.replace(r"\N", line_break_token)
    escaped = escaped.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return escaped.replace(line_break_token, r"\N")


def _resolve_transcript(
    segment: EditSegment,
    transcripts: TranscriptCollection,
) -> TranscriptResult | None:
    if transcripts is None or isinstance(transcripts, TranscriptResult):
        return transcripts
    if segment.source_asset_id is None:
        return next(iter(transcripts.values()), None)
    return transcripts.get(segment.source_asset_id)


def _words_for_segment(
    transcript: TranscriptResult,
    *,
    source_start: float,
    source_end: float,
) -> list[TranscriptWord]:
    return [
        word
        for word in transcript.words
        if word.end > source_start and word.start < source_end and word.end > word.start
    ]


def build_caption_cues(
    graph: EditDecisionGraph,
    transcript: TranscriptCollection,
    *,
    max_words: int = 5,
    max_cue_seconds: float = 2.2,
) -> list[CaptionCue]:
    if transcript is None:
        return []

    cues: list[CaptionCue] = []
    for segment in graph.segments:
        segment_transcript = _resolve_transcript(segment, transcript)
        if segment_transcript is None or not segment_transcript.words:
            continue
        words = _words_for_segment(
            segment_transcript,
            source_start=segment.source_start,
            source_end=segment.source_end,
        )
        if not words:
            continue

        group: list[TranscriptWord] = []
        for word in words:
            if group:
                projected_duration = word.end - group[0].start
                if len(group) >= max_words or projected_duration > max_cue_seconds:
                    cues.append(_cue_from_words(group, segment))
                    group = []
            group.append(word)
        if group:
            cues.append(_cue_from_words(group, segment))

    return [cue for cue in cues if cue.end - cue.start >= 0.12 and cue.text]


def _cue_from_words(words: list[TranscriptWord], segment: EditSegment) -> CaptionCue:
    start = segment.output_start + max(0.0, words[0].start - segment.source_start)
    end = segment.output_start + max(0.0, words[-1].end - segment.source_start)
    end = min(end, segment.output_end)
    text_words = [word.word.strip() for word in words if word.word.strip()]
    if len(text_words) > 3:
        split_at = (len(text_words) + 1) // 2
        text = " ".join(text_words[:split_at]) + r"\N" + " ".join(text_words[split_at:])
    else:
        text = " ".join(text_words)
    return CaptionCue(start=round(start, 3), end=round(end, 3), text=text)


def _animation(style: CaptionStyle) -> str:
    alignment = {"lower": 2, "center": 5, "upper": 8}[style.position]
    if style.animation == "none":
        return rf"{{\an{alignment}}}"
    if style.animation == "fade":
        return rf"{{\an{alignment}\fad(100,120)}}"
    return rf"{{\an{alignment}\fad(60,80)\fscx108\fscy108\t(0,120,\fscx100\fscy100)}}"


def write_ass_captions(
    path: str | Path,
    graph: EditDecisionGraph,
    transcript: TranscriptCollection,
    *,
    style: CaptionStyle | None = None,
    max_words: int | None = None,
    margin_vertical: int | None = None,
) -> int:
    caption_style = style or CaptionStyle()
    output = Path(path)
    if not caption_style.enabled:
        output.unlink(missing_ok=True)
        return 0
    if max_words is not None or margin_vertical is not None:
        caption_style = caption_style.model_copy(
            update={
                "max_words": max_words if max_words is not None else caption_style.max_words,
                "margin_vertical": (
                    margin_vertical if margin_vertical is not None else caption_style.margin_vertical
                ),
            }
        )

    cues = build_caption_cues(graph, transcript, max_words=caption_style.max_words)
    output.parent.mkdir(parents=True, exist_ok=True)

    primary = hex_to_ass(caption_style.primary_color)
    accent = hex_to_ass(caption_style.accent_color)
    outline = hex_to_ass(caption_style.outline_color)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Director,{caption_style.font_name},{caption_style.font_size},{primary},{accent},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,6,0,2,90,90,{caption_style.margin_vertical},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    animation = _animation(caption_style)
    event_lines = []
    for cue in cues:
        text = cue.text.upper() if caption_style.all_caps else cue.text
        event_lines.append(
            "Dialogue: 0,"
            f"{_ass_time(cue.start)},{_ass_time(cue.end)},Director,,0,0,0,,"
            f"{animation}{_escape_ass_text(text)}"
        )

    output.write_text(header + "\n".join(event_lines) + ("\n" if event_lines else ""), encoding="utf-8")
    return len(cues)
