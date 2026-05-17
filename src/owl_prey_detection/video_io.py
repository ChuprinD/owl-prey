from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameItem:
    frame_index: int
    timestamp_sec: float
    image_bgr: np.ndarray
    read_runtime_sec: float


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    duration_sec: float
    width: int
    height: int


def get_video_metadata(video_path: str | Path) -> VideoMetadata:
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if fps <= 0:
            fps = 25.0

        duration_sec = frame_count / fps if frame_count > 0 else 0.0

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
            width=width,
            height=height,
        )

    finally:
        cap.release()


def resolve_stride(
    video_fps: float,
    frame_stride: int = 1,
    sample_fps: Optional[float] = None,
) -> int:
    if sample_fps is None:
        return max(1, int(frame_stride))

    return max(1, int(round(video_fps / float(sample_fps))))


def iter_video_frames(
    video_path: str | Path,
    frame_stride: int = 1,
    sample_fps: Optional[float] = None,
) -> Iterator[FrameItem]:
    video_path = Path(video_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if fps <= 0:
        fps = 25.0

    stride = resolve_stride(
        video_fps=fps,
        frame_stride=frame_stride,
        sample_fps=sample_fps,
    )

    frame_index = 0

    try:
        while True:
            read_start = perf_counter()
            ok, frame = cap.read()
            read_runtime_sec = perf_counter() - read_start

            if not ok:
                break

            if frame_index % stride == 0:
                yield FrameItem(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    image_bgr=frame,
                    read_runtime_sec=read_runtime_sec,
                )

            frame_index += 1

    finally:
        cap.release()