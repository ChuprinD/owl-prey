from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL.ImagePalette import raw
import yaml


@dataclass
class PipelineConfig:
    model_path: Path
    output_dir: Path

    input_source: str = "local"
    video_path: Optional[Path] = None
    blob_name: Optional[str] = None
    download_dir: Path = Path("data/videos/downloaded")
    overwrite_download: bool = False

    frame_stride: int = 1
    sample_fps: Optional[float] = None

    class_groups: dict[str, list[str]] = field(default_factory=dict)
    required_groups: list[str] = field(default_factory=lambda: ["prey"])

    top_k: int = 5
    min_gap_seconds: float = 1.0

    inference_conf: float = 0.05
    inference_iou: float = 0.5
    inference_imgsz: int = 640
    inference_max_det: int = 100
    inference_device: str | int | None = None
    inference_augment: bool = False

    @staticmethod
    def from_yaml(path: str | Path) -> "PipelineConfig":
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        base_dir = path.parent

        inference_raw = raw.get("inference", {}) or {}
        selection_raw = raw.get("selection", {}) or {}
        input_raw = raw.get("input", {}) or {}

        model_path = Path(raw.get("model_path", "models/best.pt"))
        output_dir = Path(raw.get("output_dir", "runs/demo"))

        if not model_path.is_absolute():
            model_path = base_dir / model_path

        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir

        video_path = None
        video_path_raw = input_raw.get("video_path")

        if video_path_raw:
            video_path = Path(video_path_raw)

            if not video_path.is_absolute():
                video_path = base_dir / video_path

        download_dir = Path(input_raw.get("download_dir", "data/videos/downloaded"))

        if not download_dir.is_absolute():
            download_dir = base_dir / download_dir

        return PipelineConfig(
            model_path=model_path,
            output_dir=output_dir,
            input_source=str(input_raw.get("source", "local")),
            video_path=video_path,
            blob_name=input_raw.get("blob_name"),
            download_dir=download_dir,
            overwrite_download=bool(input_raw.get("overwrite_download", False)),
            frame_stride=int(raw.get("frame_stride", 1)),
            sample_fps=raw.get("sample_fps"),
            class_groups=dict(
                raw.get(
                    "class_groups",
                    {
                        "owl": ["owl"],
                        "prey": ["prey"],
                    },
                )
            ),
            required_groups=list(raw.get("required_groups", ["prey"])),
            top_k=int(raw.get("top_k", 5)),
            min_gap_seconds=float(raw.get("min_gap_seconds", 1.0)),
            inference_conf=float(inference_raw.get("conf", 0.05)),
            inference_iou=float(inference_raw.get("iou", 0.5)),
            inference_imgsz=int(inference_raw.get("imgsz", 640)),
            inference_max_det=int(inference_raw.get("max_det", 100)),
            inference_device=inference_raw.get("device"),
            inference_augment=bool(inference_raw.get("augment", False)),
        )

    def validate(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.input_source not in {"local", "azure_blob"}:
            raise ValueError("input.source must be either 'local' or 'azure_blob'.")

        if self.input_source == "local" and self.video_path is None:
            raise ValueError("input.video_path is required for local input.")

        if self.input_source == "azure_blob" and not self.blob_name:
            raise ValueError("input.blob_name is required for Azure Blob input.")

        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1.")

        if self.sample_fps is not None and self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive or null.")

        if self.top_k < 1:
            raise ValueError("top_k must be >= 1.")

        for group_name in self.required_groups:
            if group_name not in self.class_groups:
                raise ValueError(
                    f"required group '{group_name}' is missing in class_groups."
                )
        if not 0.0 <= self.inference_conf <= 1.0:
            raise ValueError("inference.conf must be in [0, 1].")

        if not 0.0 <= self.inference_iou <= 1.0:
            raise ValueError("inference.iou must be in [0, 1].")

        if self.inference_imgsz < 1:
            raise ValueError("inference.imgsz must be positive.")

        if self.inference_max_det < 1:
            raise ValueError("inference.max_det must be >= 1.")
