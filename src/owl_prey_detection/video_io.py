from pathlib import Path
from time import perf_counter

import cv2


class FrameItem:
    def __init__(self, frame_index, timestamp_sec, image_bgr, read_runtime_sec):
        self.frame_index = frame_index
        self.timestamp_sec = timestamp_sec
        self.image_bgr = image_bgr
        self.read_runtime_sec = read_runtime_sec


class VideoMetadata:
    def __init__(self, fps, frame_count, duration_sec, width, height):
        self.fps = fps
        self.frame_count = frame_count
        self.duration_sec = duration_sec
        self.width = width
        self.height = height


def get_video_metadata(video_path):
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


def resolve_stride(video_fps, frame_stride=1, sample_fps=None):
    if sample_fps is None:
        return max(1, int(frame_stride))

    return max(1, int(round(video_fps / float(sample_fps))))


def iter_video_frames(video_path, frame_stride=1, sample_fps=None):
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


def count_interval_frames(intervals, fps, frame_count, stride):
    total = 0

    for start_sec, end_sec in intervals:
        start = int(start_sec * fps)
        end = min(int(end_sec * fps), frame_count - 1)
        first = start + (-start % stride)

        if first <= end:
            total += (end - first) // stride + 1

    return total


def iter_video_frames_in_intervals(
    video_path,
    intervals,
    frame_stride=1,
    sample_fps=None,
):
    if not intervals:
        return

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if fps <= 0:
        fps = 25.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = resolve_stride(fps, frame_stride, sample_fps)

    try:
        for start_sec, end_sec in intervals:
            frame_index = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            while frame_index <= end_frame and frame_index < frame_count:
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
