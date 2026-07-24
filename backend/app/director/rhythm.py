from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from app.core.config import Settings
from app.director.edit_graph import EditDecisionGraph
from app.sensory.models import MusicProfile


@dataclass(frozen=True, slots=True)
class RhythmEstimate:
    tempo_bpm: float | None
    beat_interval_seconds: float | None
    beat_offset_seconds: float
    beat_times: tuple[float, ...]
    phrase_times: tuple[float, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class MusicTimingPlan:
    usable: bool
    start_offset_seconds: float
    tempo_bpm: float | None
    beat_interval_seconds: float | None
    total_cut_count: int
    beat_aligned_cut_count: int
    phrase_aligned_cut_count: int
    alignment_score: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "usable": self.usable,
            "start_offset_seconds": self.start_offset_seconds,
            "tempo_bpm": self.tempo_bpm,
            "beat_interval_seconds": self.beat_interval_seconds,
            "total_cut_count": self.total_cut_count,
            "beat_aligned_cut_count": self.beat_aligned_cut_count,
            "phrase_aligned_cut_count": self.phrase_aligned_cut_count,
            "alignment_score": self.alignment_score,
            "reason": self.reason,
        }


def pcm_energy_envelope(
    samples: list[int] | tuple[int, ...],
    *,
    sample_rate: int,
    frames_per_second: int = 50,
) -> list[float]:
    if sample_rate <= 0 or frames_per_second <= 0 or not samples:
        return []
    window = max(1, round(sample_rate / frames_per_second))
    envelope: list[float] = []
    for start in range(0, len(samples), window):
        chunk = samples[start : start + window]
        if not chunk:
            continue
        mean_square = sum(float(sample) * float(sample) for sample in chunk) / len(chunk)
        envelope.append(math.log1p(math.sqrt(mean_square)))
    return envelope


def _onset_envelope(energy: list[float]) -> list[float]:
    if len(energy) < 3:
        return []
    baseline = energy[0]
    onsets: list[float] = [0.0]
    for value in energy[1:]:
        baseline = baseline * 0.82 + value * 0.18
        onsets.append(max(0.0, value - baseline))
    peak = max(onsets, default=0.0)
    if peak <= 0:
        return onsets
    return [value / peak for value in onsets]


def _normalized_autocorrelation(values: list[float], lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return 0.0
    left = values[lag:]
    right = values[:-lag]
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_power = sum(value * value for value in left)
    right_power = sum(value * value for value in right)
    denominator = math.sqrt(left_power * right_power)
    return numerator / denominator if denominator > 1e-12 else 0.0


def estimate_rhythm_from_envelope(
    energy_envelope: list[float],
    *,
    frames_per_second: int,
    duration_seconds: float,
    min_bpm: float = 60,
    max_bpm: float = 180,
    phrase_bars: int = 4,
) -> RhythmEstimate:
    if (
        frames_per_second <= 0
        or duration_seconds <= 0
        or len(energy_envelope) < frames_per_second * 4
    ):
        return RhythmEstimate(None, None, 0.0, (), (), 0.0)

    onsets = _onset_envelope(energy_envelope)
    minimum_lag = max(1, round(frames_per_second * 60 / max_bpm))
    maximum_lag = min(len(onsets) // 2, round(frames_per_second * 60 / min_bpm))
    if maximum_lag <= minimum_lag:
        return RhythmEstimate(None, None, 0.0, (), (), 0.0)

    scored: list[tuple[float, int, float]] = []
    for lag in range(minimum_lag, maximum_lag + 1):
        bpm = frames_per_second * 60 / lag
        correlation = _normalized_autocorrelation(onsets, lag)
        tempo_prior = max(0.88, 1.0 - abs(bpm - 120.0) / 900.0)
        scored.append((correlation * tempo_prior, lag, correlation))
    scored.sort(reverse=True)
    best_adjusted, best_lag, best_correlation = scored[0]
    background = median(item[0] for item in scored)
    confidence = min(
        1.0,
        max(0.0, best_correlation * 0.55 + max(0.0, best_adjusted - background) * 1.8),
    )
    if best_correlation < 0.08 or confidence < 0.08:
        return RhythmEstimate(None, None, 0.0, (), (), round(confidence, 3))

    offset_scores: list[tuple[float, int]] = []
    for offset in range(best_lag):
        positions = range(offset, len(onsets), best_lag)
        score = sum(onsets[position] for position in positions)
        offset_scores.append((score, offset))
    _, best_offset = max(offset_scores, default=(0.0, 0))

    beat_interval = best_lag / frames_per_second
    beat_offset = best_offset / frames_per_second
    tempo_bpm = 60 / beat_interval
    beat_times: list[float] = []
    cursor = beat_offset
    while cursor <= duration_seconds + 1e-6 and len(beat_times) < 4096:
        beat_times.append(round(cursor, 3))
        cursor += beat_interval

    phrase_beats = max(4, phrase_bars * 4)
    phrase_times = tuple(beat_times[::phrase_beats])
    return RhythmEstimate(
        tempo_bpm=round(tempo_bpm, 2),
        beat_interval_seconds=round(beat_interval, 6),
        beat_offset_seconds=round(beat_offset, 6),
        beat_times=tuple(beat_times),
        phrase_times=phrase_times,
        confidence=round(confidence, 3),
    )


def _grid_distance(value: float, *, origin: float, interval: float) -> float:
    phase = (value - origin) % interval
    return min(phase, interval - phase)


def plan_music_timing(
    graph: EditDecisionGraph,
    profile: MusicProfile,
    *,
    max_beat_distance_seconds: float = 0.14,
    minimum_confidence: float = 0.18,
    phrase_bars: int = 4,
) -> MusicTimingPlan:
    interval = profile.beat_interval_seconds
    if interval is None or interval <= 0 or profile.rhythm_confidence < minimum_confidence:
        return MusicTimingPlan(
            usable=False,
            start_offset_seconds=0.0,
            tempo_bpm=profile.tempo_bpm,
            beat_interval_seconds=interval,
            total_cut_count=max(0, len(graph.segments) - 1),
            beat_aligned_cut_count=0,
            phrase_aligned_cut_count=0,
            alignment_score=0.0,
            reason="Music rhythm confidence is too low for deterministic beat sync.",
        )

    cuts = [segment.output_start for segment in graph.segments[1:]]
    if not cuts:
        return MusicTimingPlan(
            usable=True,
            start_offset_seconds=round(profile.beat_offset_seconds, 3),
            tempo_bpm=profile.tempo_bpm,
            beat_interval_seconds=interval,
            total_cut_count=0,
            beat_aligned_cut_count=0,
            phrase_aligned_cut_count=0,
            alignment_score=1.0,
            reason="The music entrance begins on its detected beat grid.",
        )

    phrase_interval = interval * max(4, phrase_bars * 4)
    candidates = {0.0, profile.beat_offset_seconds}
    for cut in cuts:
        candidates.add((profile.beat_offset_seconds - cut) % interval)
        candidates.add((profile.beat_offset_seconds - cut) % phrase_interval)

    source_change_indexes = {
        index
        for index, segment in enumerate(graph.segments[1:])
        if segment.transition
        in {"source_change_cut", "match_cut", "continuity_cut", "soft_dissolve"}
    }
    phrase_tolerance = max(0.22, max_beat_distance_seconds * 1.8)
    best: tuple[float, float, int, int] | None = None
    for raw_offset in candidates:
        offset = raw_offset
        if profile.duration_seconds > 0:
            offset %= profile.duration_seconds
        beat_scores: list[float] = []
        phrase_scores: list[float] = []
        beat_aligned = 0
        phrase_aligned = 0
        for index, cut in enumerate(cuts):
            track_time = cut + offset
            beat_distance = _grid_distance(
                track_time,
                origin=profile.beat_offset_seconds,
                interval=interval,
            )
            beat_score = max(0.0, 1.0 - beat_distance / max_beat_distance_seconds)
            beat_scores.append(beat_score)
            if beat_distance <= max_beat_distance_seconds:
                beat_aligned += 1
            if index in source_change_indexes:
                phrase_distance = _grid_distance(
                    track_time,
                    origin=profile.beat_offset_seconds,
                    interval=phrase_interval,
                )
                phrase_score = max(0.0, 1.0 - phrase_distance / phrase_tolerance)
                phrase_scores.append(phrase_score)
                if phrase_distance <= phrase_tolerance:
                    phrase_aligned += 1

        beat_average = sum(beat_scores) / len(beat_scores)
        phrase_average = sum(phrase_scores) / len(phrase_scores) if phrase_scores else beat_average
        score = beat_average * 0.82 + phrase_average * 0.18
        candidate = (score, -offset, beat_aligned, phrase_aligned)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    score, negative_offset, beat_aligned, phrase_aligned = best
    offset = -negative_offset
    return MusicTimingPlan(
        usable=True,
        start_offset_seconds=round(offset, 3),
        tempo_bpm=profile.tempo_bpm,
        beat_interval_seconds=interval,
        total_cut_count=len(cuts),
        beat_aligned_cut_count=beat_aligned,
        phrase_aligned_cut_count=phrase_aligned,
        alignment_score=round(score, 3),
        reason=(
            f"Aligned {beat_aligned}/{len(cuts)} edit boundary(ies) to the detected beat grid "
            f"and {phrase_aligned} source-change boundary(ies) to musical phrases."
        ),
    )


def prepare_aligned_music(
    source_path: str | Path,
    output_path: str | Path,
    *,
    plan: MusicTimingPlan,
    duration_seconds: float,
    settings: Settings,
) -> None:
    if not plan.usable or duration_seconds <= 0:
        raise ValueError("A usable music timing plan and positive duration are required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        settings.ffmpeg_binary,
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(source_path),
        "-ss",
        f"{plan.start_offset_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "48000",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=settings.render_timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"Beat-aligned music preparation failed: {stderr[-2_000:]}") from exc
