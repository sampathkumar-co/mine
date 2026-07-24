from __future__ import annotations

import math
import subprocess
from collections.abc import Sequence
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
    asset_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "usable": self.usable,
            "asset_id": self.asset_id,
            "start_offset_seconds": self.start_offset_seconds,
            "tempo_bpm": self.tempo_bpm,
            "beat_interval_seconds": self.beat_interval_seconds,
            "total_cut_count": self.total_cut_count,
            "beat_aligned_cut_count": self.beat_aligned_cut_count,
            "phrase_aligned_cut_count": self.phrase_aligned_cut_count,
            "alignment_score": self.alignment_score,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> MusicTimingPlan | None:
        if not isinstance(payload, dict):
            return None
        try:
            tempo = payload.get("tempo_bpm")
            interval = payload.get("beat_interval_seconds")
            asset_id = payload.get("asset_id")
            return cls(
                usable=bool(payload.get("usable")),
                asset_id=str(asset_id) if asset_id else None,
                start_offset_seconds=max(0.0, float(payload.get("start_offset_seconds", 0))),
                tempo_bpm=float(tempo) if tempo is not None else None,
                beat_interval_seconds=float(interval) if interval is not None else None,
                total_cut_count=max(0, int(payload.get("total_cut_count", 0))),
                beat_aligned_cut_count=max(0, int(payload.get("beat_aligned_cut_count", 0))),
                phrase_aligned_cut_count=max(
                    0, int(payload.get("phrase_aligned_cut_count", 0))
                ),
                alignment_score=min(1.0, max(0.0, float(payload.get("alignment_score", 0)))),
                reason=str(payload.get("reason", ""))[:1_000],
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class MusicEnvelopeWindow:
    start_seconds: float
    end_seconds: float
    multiplier: float
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "multiplier": self.multiplier,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> MusicEnvelopeWindow | None:
        if not isinstance(payload, dict):
            return None
        try:
            start = max(0.0, float(payload.get("start_seconds", 0)))
            end = float(payload.get("end_seconds", 0))
            multiplier = min(1.5, max(0.1, float(payload.get("multiplier", 1))))
            if end <= start:
                return None
            return cls(start, end, multiplier, str(payload.get("reason", ""))[:500])
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class MusicSting:
    start_seconds: float
    duration_seconds: float
    gain_multiplier: float
    highpass_hz: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "gain_multiplier": self.gain_multiplier,
            "highpass_hz": self.highpass_hz,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> MusicSting | None:
        if not isinstance(payload, dict):
            return None
        try:
            start = max(0.0, float(payload.get("start_seconds", 0)))
            duration = min(1.0, max(0.08, float(payload.get("duration_seconds", 0))))
            gain = min(0.8, max(0.02, float(payload.get("gain_multiplier", 0.25))))
            highpass = min(5_000, max(300, int(payload.get("highpass_hz", 1_200))))
            return cls(start, duration, gain, highpass, str(payload.get("reason", ""))[:500])
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class SoundDesignPlan:
    usable: bool
    asset_id: str | None
    phrase_markers: tuple[float, ...]
    ducking_windows: tuple[MusicEnvelopeWindow, ...]
    lift_windows: tuple[MusicEnvelopeWindow, ...]
    stings: tuple[MusicSting, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "usable": self.usable,
            "asset_id": self.asset_id,
            "phrase_markers": list(self.phrase_markers),
            "ducking_windows": [window.as_dict() for window in self.ducking_windows],
            "lift_windows": [window.as_dict() for window in self.lift_windows],
            "stings": [sting.as_dict() for sting in self.stings],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> SoundDesignPlan | None:
        if not isinstance(payload, dict):
            return None
        asset_id = payload.get("asset_id")
        try:
            markers = tuple(
                sorted(
                    {
                        round(max(0.0, float(value)), 3)
                        for value in (payload.get("phrase_markers") or [])
                    }
                )
            )
        except (TypeError, ValueError):
            markers = ()
        ducking = tuple(
            window
            for item in (payload.get("ducking_windows") or [])
            if (window := MusicEnvelopeWindow.from_dict(item)) is not None
        )
        lifts = tuple(
            window
            for item in (payload.get("lift_windows") or [])
            if (window := MusicEnvelopeWindow.from_dict(item)) is not None
        )
        stings = tuple(
            sting
            for item in (payload.get("stings") or [])
            if (sting := MusicSting.from_dict(item)) is not None
        )
        return cls(
            usable=bool(payload.get("usable")),
            asset_id=str(asset_id) if asset_id else None,
            phrase_markers=markers[:256],
            ducking_windows=ducking[:64],
            lift_windows=lifts[:16],
            stings=stings[:8],
            reason=str(payload.get("reason", ""))[:1_000],
        )


def pcm_energy_envelope(
    samples: Sequence[int],
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
            asset_id=profile.asset_id,
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
            asset_id=profile.asset_id,
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
        asset_id=profile.asset_id,
    )


def _grid_times(
    *,
    duration_seconds: float,
    origin_seconds: float,
    interval_seconds: float,
    limit: int = 256,
) -> tuple[float, ...]:
    if duration_seconds <= 0 or interval_seconds <= 0:
        return ()
    cursor = origin_seconds % interval_seconds
    values: list[float] = []
    while cursor <= duration_seconds + 1e-6 and len(values) < limit:
        values.append(round(0.0 if cursor < 0.04 else cursor, 3))
        cursor += interval_seconds
    return tuple(values)


def _merge_windows(
    windows: list[MusicEnvelopeWindow],
    *,
    use_minimum_multiplier: bool,
) -> tuple[MusicEnvelopeWindow, ...]:
    if not windows:
        return ()
    ordered = sorted(windows, key=lambda item: (item.start_seconds, item.end_seconds))
    merged: list[MusicEnvelopeWindow] = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_seconds <= previous.end_seconds + 0.03:
            multiplier = (
                min(previous.multiplier, current.multiplier)
                if use_minimum_multiplier
                else max(previous.multiplier, current.multiplier)
            )
            merged[-1] = MusicEnvelopeWindow(
                start_seconds=previous.start_seconds,
                end_seconds=max(previous.end_seconds, current.end_seconds),
                multiplier=multiplier,
                reason=f"{previous.reason}; {current.reason}"[:500],
            )
        else:
            merged.append(current)
    return tuple(merged)


def plan_sound_design(
    graph: EditDecisionGraph,
    profile: MusicProfile,
    timing: MusicTimingPlan,
    *,
    phrase_bars: int = 4,
    max_lifts: int = 4,
    max_stings: int = 3,
) -> SoundDesignPlan:
    interval = timing.beat_interval_seconds
    duration = max(0.0, graph.selected_duration_seconds)
    if (
        not timing.usable
        or interval is None
        or interval <= 0
        or duration <= 0
        or timing.asset_id != profile.asset_id
    ):
        return SoundDesignPlan(
            usable=False,
            asset_id=profile.asset_id,
            phrase_markers=(),
            ducking_windows=(),
            lift_windows=(),
            stings=(),
            reason="Sound design stayed on the safe baseline because no trusted timing grid was available.",
        )

    beat_origin = (profile.beat_offset_seconds - timing.start_offset_seconds) % interval
    phrase_interval = interval * max(4, phrase_bars * 4)
    phrase_markers = _grid_times(
        duration_seconds=duration,
        origin_seconds=beat_origin,
        interval_seconds=phrase_interval,
    )

    ducking: list[MusicEnvelopeWindow] = []
    for segment in graph.segments:
        spoken = bool((segment.transcript_text or "").strip()) or segment.clip_role == "primary_speech"
        if not spoken:
            continue
        start = max(0.0, segment.output_start - 0.06)
        end = min(duration, segment.output_end + 0.18)
        previous_beat = start - ((start - beat_origin) % interval)
        if start - previous_beat <= 0.12:
            start = max(0.0, previous_beat)
        next_beat = end + ((beat_origin - end) % interval)
        if next_beat - end <= 0.28:
            end = min(duration, next_beat)
        if end - start < 0.08:
            continue
        multiplier = 0.72 if (segment.transcript_text or "").strip() else 0.82
        ducking.append(
            MusicEnvelopeWindow(
                start_seconds=round(start, 3),
                end_seconds=round(end, 3),
                multiplier=multiplier,
                reason="Phrase-aware speech protection.",
            )
        )
    ducking_windows = _merge_windows(ducking, use_minimum_multiplier=True)

    transitions = [
        (segment.output_start, segment.transition, segment.reason)
        for segment in graph.segments[1:]
        if segment.transition
        in {"source_change_cut", "match_cut", "continuity_cut", "soft_dissolve"}
    ]
    lifts: list[MusicEnvelopeWindow] = []
    intro_end = min(duration, max(0.5, interval * 1.5))
    if intro_end >= 0.3:
        lifts.append(
            MusicEnvelopeWindow(0.0, round(intro_end, 3), 1.06, "Opening section lift.")
        )

    tolerance = max(0.22, interval * 0.65)
    aligned_transitions: list[tuple[float, str, str]] = []
    for cut, transition, reason in transitions:
        nearest = min(phrase_markers, key=lambda marker: abs(marker - cut), default=None)
        if nearest is None or abs(nearest - cut) > tolerance:
            continue
        aligned_transitions.append((cut, transition, reason))
        lift_end = min(duration, cut + min(1.0, interval * 2.2))
        if lift_end - cut >= 0.25:
            lifts.append(
                MusicEnvelopeWindow(
                    round(max(0.0, cut - 0.04), 3),
                    round(lift_end, 3),
                    1.12,
                    "Musical phrase lift at a story-section change.",
                )
            )
        if len(lifts) >= max_lifts:
            break

    if duration >= 2.5 and len(lifts) < max_lifts:
        outro_start = max(0.0, duration - max(0.8, interval * 2.0))
        lifts.append(
            MusicEnvelopeWindow(
                round(outro_start, 3),
                round(duration, 3),
                1.08,
                "Closing section lift.",
            )
        )
    lift_windows = _merge_windows(lifts[:max_lifts], use_minimum_multiplier=False)

    stings: list[MusicSting] = []
    for cut, transition, reason in aligned_transitions:
        if transition == "soft_dissolve":
            continue
        sting_duration = min(0.46, max(0.22, interval * 0.75))
        stings.append(
            MusicSting(
                start_seconds=round(max(0.0, cut - 0.03), 3),
                duration_seconds=round(min(sting_duration, duration - cut + 0.03), 3),
                gain_multiplier=0.30,
                highpass_hz=1_200,
                reason=f"Restrained accent for {transition}: {reason}"[:500],
            )
        )
        if len(stings) >= max_stings:
            break

    return SoundDesignPlan(
        usable=True,
        asset_id=profile.asset_id,
        phrase_markers=phrase_markers,
        ducking_windows=ducking_windows,
        lift_windows=lift_windows,
        stings=tuple(stings),
        reason=(
            f"Planned {len(ducking_windows)} speech-protection window(s), "
            f"{len(lift_windows)} section lift(s), and {len(stings)} restrained sting(s)."
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
