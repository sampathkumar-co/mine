from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.rendering.ffmpeg import MediaProbe
from app.sensory.models import ReferenceStyleProfile, SceneRange
from app.sensory.scenes import detect_scenes


def build_reference_profile(
    *,
    asset_id: str | None,
    duration_seconds: float,
    scenes: list[SceneRange],
    brightness: float,
    saturation: float,
    motion_energy: float,
    sampled_frames: int,
) -> ReferenceStyleProfile:
    shot_count = max(1, len(scenes))
    average_shot = duration_seconds / shot_count if duration_seconds > 0 else 0
    cuts_per_minute = max(0.0, (shot_count - 1) * 60 / max(duration_seconds, 0.1))
    if average_shot and average_shot < 2.2:
        pace = "fast"
    elif average_shot > 5.5:
        pace = "slow"
    else:
        pace = "balanced"

    return ReferenceStyleProfile(
        source_asset_id=asset_id,
        duration_seconds=round(max(0.0, duration_seconds), 3),
        average_shot_seconds=round(max(0.0, average_shot), 3),
        cuts_per_minute=round(cuts_per_minute, 3),
        brightness=min(1.0, max(0.0, brightness)),
        saturation=min(1.0, max(0.0, saturation)),
        motion_energy=min(1.0, max(0.0, motion_energy)),
        pace=pace,
        sampled_frames=max(0, sampled_frames),
    )


def analyze_reference_style(
    path: str | Path,
    *,
    asset_id: str | None,
    media_probe: MediaProbe,
    settings: Settings,
) -> ReferenceStyleProfile:
    scenes = detect_scenes(
        path,
        duration_seconds=media_probe.duration_seconds,
        settings=settings,
    )
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened() or media_probe.duration_seconds <= 0:
        return build_reference_profile(
            asset_id=asset_id,
            duration_seconds=media_probe.duration_seconds,
            scenes=scenes,
            brightness=0.5,
            saturation=0.5,
            motion_energy=0.3,
            sampled_frames=0,
        )

    sample_count = settings.reference_frame_samples
    timestamps = np.linspace(0, max(0.0, media_probe.duration_seconds - 0.05), sample_count)
    brightness_values: list[float] = []
    saturation_values: list[float] = []
    motion_values: list[float] = []
    previous_gray: np.ndarray | None = None

    try:
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            resized = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            brightness_values.append(float(hsv[:, :, 2].mean() / 255))
            saturation_values.append(float(hsv[:, :, 1].mean() / 255))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None:
                difference = cv2.absdiff(gray, previous_gray)
                motion_values.append(float(difference.mean() / 255))
            previous_gray = gray
    finally:
        capture.release()

    sampled_frames = len(brightness_values)
    brightness = float(np.mean(brightness_values)) if brightness_values else 0.5
    saturation = float(np.mean(saturation_values)) if saturation_values else 0.5
    raw_motion = float(np.mean(motion_values)) if motion_values else 0.08
    motion_energy = min(1.0, raw_motion * 4.0)

    return build_reference_profile(
        asset_id=asset_id,
        duration_seconds=media_probe.duration_seconds,
        scenes=scenes,
        brightness=brightness,
        saturation=saturation,
        motion_energy=motion_energy,
        sampled_frames=sampled_frames,
    )
