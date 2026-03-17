from __future__ import annotations

import cv2
import numpy as np


def sharpness_score(frame: np.ndarray) -> float:
    """Calculates a sharpness score based on the variance of the Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_motion_frames(
    video_path: str,
    skip_frames: int,
    motion_threshold: int,
    min_motion_pixels: int,
    min_sharpness: int,
    jpeg_quality: int = 92,
) -> list[tuple[bytes, int, float]]:
    """
    Extracts frames with significant motion and sufficient sharpness.
    Returns a sorted list of tuples: (jpeg_bytes, frame_no, sharpness_score).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    time_bins: dict[int, tuple[bytes, int, float]] = {}
    prev_gray = None
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        if frame_no % skip_frames != 0:
            continue

        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
        n_changed = np.sum(thresh > 0)

        if n_changed >= min_motion_pixels:
            sec = int(frame_no / fps)
            score = sharpness_score(frame)
            if score >= min_sharpness:
                # Keep the sharpest frame within this one-second bin
                if sec not in time_bins or score > time_bins[sec][2]:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    time_bins[sec] = (buf.tobytes(), frame_no, score)

        prev_gray = gray
    cap.release()

    return sorted(time_bins.values(), key=lambda x: x[1])
