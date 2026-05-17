from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class YoloDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.05,
        iou: float = 0.5,
        imgsz: int = 640,
        max_det: int = 100,
        device: str | int | None = None,
        augment: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self.augment = augment
        self.names = self.model.names

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(class_id, class_id))

        if isinstance(self.names, list) and 0 <= class_id < len(self.names):
            return str(self.names[class_id])

        return str(class_id)

    def predict_batch(
        self,
        images_bgr: list[np.ndarray],
        ) -> list[list[dict[str, Any]]]:
            results = self.model.predict(
                source=images_bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                max_det=self.max_det,
                device=self.device,
                augment=self.augment,
                verbose=False,
            )

            batch_detections: list[list[dict[str, Any]]] = []

            for result in results:
                detections: list[dict[str, Any]] = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0].item())
                        label = self._class_name(class_id)
                        confidence = float(box.conf[0].item())
                        xyxy = [float(value) for value in box.xyxy[0].tolist()]

                        detections.append(
                            {
                                "class_id": class_id,
                                "label": label,
                                "confidence": confidence,
                                "xyxy": xyxy,
                            }
                        )

                detections.sort(
                    key=lambda item: item["confidence"],
                    reverse=True,
                )

                batch_detections.append(detections)

            return batch_detections