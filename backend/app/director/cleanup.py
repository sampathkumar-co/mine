from __future__ import annotations

import re

from app.director.edit_graph import EditDecisionGraph, EditSegment
from app.sensory.models import TranscriptResult, TranscriptWord

VOCAL_FILLERS = {"ah", "erm", "hmm", "uh", "um"}


def _normalise_word(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", value.casefold())


def _is_filler(word: TranscriptWord) -> bool:
    return _normalise_word(word.word) in VOCAL_FILLERS


def _words_inside(
    transcript: TranscriptResult,
    *,
    start: float,
    end: float,
) -> list[TranscriptWord]:
    return [
        word
        for word in transcript.words
        if word.end > start and word.start < end and word.end > word.start
    ]


def _speech_runs(
    words: list[TranscriptWord],
    *,
    silence_threshold_seconds: float,
) -> list[list[TranscriptWord]]:
    runs: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []

    for word in words:
        if _is_filler(word):
            if current:
                runs.append(current)
                current = []
            continue

        if current and word.start - current[-1].end >= silence_threshold_seconds:
            runs.append(current)
            current = []
        current.append(word)

    if current:
        runs.append(current)
    return runs


def refine_graph_with_word_timings(
    graph: EditDecisionGraph,
    transcript: TranscriptResult | None,
    *,
    silence_threshold_seconds: float = 0.55,
    speech_padding_seconds: float = 0.08,
    minimum_clip_seconds: float = 0.28,
) -> EditDecisionGraph:
    """Remove vocal fillers and long internal pauses using word timestamps.

    The operation is conservative: when word timings are unavailable or a selected
    segment cannot produce a stable speech run, the original graph is preserved.
    """
    if transcript is None or not transcript.words:
        return graph

    output_cursor = 0.0
    refined: list[EditSegment] = []
    removed_regions = 0

    for selected in graph.segments:
        source_words = _words_inside(
            transcript,
            start=selected.source_start,
            end=selected.source_end,
        )
        runs = _speech_runs(
            source_words,
            silence_threshold_seconds=silence_threshold_seconds,
        )
        if not runs:
            duration = selected.source_end - selected.source_start
            selected_copy = selected.model_copy(
                update={
                    "output_start": round(output_cursor, 3),
                    "output_end": round(output_cursor + duration, 3),
                }
            )
            refined.append(selected_copy)
            output_cursor += duration
            continue

        if len(runs) > 1 or any(_is_filler(word) for word in source_words):
            removed_regions += 1

        segment_added = False
        for run in runs:
            start = max(selected.source_start, run[0].start - speech_padding_seconds)
            end = min(selected.source_end, run[-1].end + speech_padding_seconds)
            duration = end - start
            if duration < minimum_clip_seconds:
                continue

            text = " ".join(word.word.strip() for word in run if word.word.strip()).strip()
            refined.append(
                EditSegment(
                    source_start=round(start, 3),
                    source_end=round(end, 3),
                    output_start=round(output_cursor, 3),
                    output_end=round(output_cursor + duration, 3),
                    score=selected.score,
                    confidence=min(selected.confidence, 0.92),
                    transition="cut",
                    reason=(
                        f"{selected.reason}; tightened with word-level filler and silence cleanup"
                    ),
                    transcript_text=text or selected.transcript_text,
                )
            )
            output_cursor += duration
            segment_added = True

        if not segment_added:
            duration = selected.source_end - selected.source_start
            refined.append(
                selected.model_copy(
                    update={
                        "output_start": round(output_cursor, 3),
                        "output_end": round(output_cursor + duration, 3),
                    }
                )
            )
            output_cursor += duration

    if not refined:
        return graph

    notes = list(graph.notes)
    if removed_regions:
        notes.append(
            f"Word-level cleanup tightened {removed_regions} selected region(s) by removing vocal fillers or long pauses."
        )

    return graph.model_copy(
        update={
            "strategy": "tier1_retention_cleanup_with_word_precision",
            "selected_duration_seconds": round(output_cursor, 3),
            "segments": refined,
            "notes": notes,
        }
    )
