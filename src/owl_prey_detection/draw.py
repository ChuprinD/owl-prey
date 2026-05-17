from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    annotated = image_bgr.copy()

    for detection in detections:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in detection["xyxy"]
        ]

        label = str(detection["label"])
        confidence = float(detection["confidence"])
        text = f"{label} {confidence:.2f}"

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated