from pathlib import Path

import yaml


def _resolve_path(path, base_dir):
    resolved = Path(path)

    if resolved.is_absolute():
        return resolved

    return base_dir / resolved


class PipelineConfig:
    def __init__(
        self,
        model_path,
        output_dir,
        input_source="local",
        video_path=None,
        blob_name=None,
        download_dir=Path("data/videos/downloaded"),
        overwrite_download=False,
        frame_stride=1,
        sample_fps=None,
        class_groups=None,
        required_groups=None,
        top_k=5,
        min_gap_seconds=1.0,
        audio_enabled=False,
        audio_threshold_db=-20.0,
        audio_padding_seconds=2.0,
        audio_min_event_seconds=0.2,
        inference_conf=0.05,
        inference_iou=0.5,
        inference_imgsz=640,
        inference_max_det=100,
        inference_device=None,
        inference_augment=False,
        inference_batch_size=16,
        postprocess_min_conf_by_group=None,
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        self.input_source = input_source
        self.video_path = video_path
        self.blob_name = blob_name
        self.download_dir = download_dir
        self.overwrite_download = overwrite_download
        self.frame_stride = frame_stride
        self.sample_fps = sample_fps
        self.class_groups = {} if class_groups is None else class_groups
        self.required_groups = ["prey"] if required_groups is None else required_groups
        self.top_k = top_k
        self.min_gap_seconds = min_gap_seconds
        self.audio_enabled = audio_enabled
        self.audio_threshold_db = audio_threshold_db
        self.audio_padding_seconds = audio_padding_seconds
        self.audio_min_event_seconds = audio_min_event_seconds
        self.inference_conf = inference_conf
        self.inference_iou = inference_iou
        self.inference_imgsz = inference_imgsz
        self.inference_max_det = inference_max_det
        self.inference_device = inference_device
        self.inference_augment = inference_augment
        self.inference_batch_size = inference_batch_size
        self.postprocess_min_conf_by_group = (
            {}
            if postprocess_min_conf_by_group is None
            else postprocess_min_conf_by_group
        )

    @staticmethod
    def from_yaml(path):
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}

        base_dir = path.parent

        inference_raw = raw.get("inference", {}) or {}
        input_raw = raw.get("input", {}) or {}
        audio_raw = raw.get("audio", {}) or {}
        postprocess_raw = raw.get("postprocess", {}) or {}

        model_path = _resolve_path(
            raw.get("model_path", "models/best.pt"),
            base_dir,
        )
        output_dir = _resolve_path(raw.get("output_dir", "runs/demo"), base_dir)

        video_path = None
        video_path_raw = input_raw.get("video_path")

        if video_path_raw:
            video_path = _resolve_path(video_path_raw, base_dir)

        download_dir = _resolve_path(
            input_raw.get("download_dir", "data/videos/downloaded"),
            base_dir,
        )

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
            top_k=_parse_top_k(raw.get("top_k", 5)),
            min_gap_seconds=float(raw.get("min_gap_seconds", 1.0)),
            audio_enabled=bool(audio_raw.get("enabled", False)),
            audio_threshold_db=float(audio_raw.get("threshold_db", -20.0)),
            audio_padding_seconds=float(audio_raw.get("padding_seconds", 2.0)),
            audio_min_event_seconds=float(audio_raw.get("min_event_seconds", 0.2)),
            inference_conf=float(inference_raw.get("conf", 0.05)),
            inference_iou=float(inference_raw.get("iou", 0.5)),
            inference_imgsz=int(inference_raw.get("imgsz", 640)),
            inference_max_det=int(inference_raw.get("max_det", 100)),
            inference_device=inference_raw.get("device"),
            inference_augment=bool(inference_raw.get("augment", False)),
            inference_batch_size=int(inference_raw.get("batch_size", 16)),
            postprocess_min_conf_by_group={
                str(group_name): float(value)
                for group_name, value in (
                    postprocess_raw.get("min_conf_by_group") or {}
                ).items()
            },
        )

    def validate(self):
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

        if self.audio_padding_seconds < 0:
            raise ValueError("audio.padding_seconds must be >= 0.")

        if self.audio_min_event_seconds <= 0:
            raise ValueError("audio.min_event_seconds must be positive.")

        if isinstance(self.top_k, int):
            if self.top_k < 1:
                raise ValueError("top_k must be >= 1.")
        elif self.top_k != "all":
            raise ValueError("top_k must be a positive integer or 'all'.")

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

        if self.inference_batch_size < 1:
            raise ValueError("inference.batch_size must be >= 1.")

        self._validate_postprocess_min_conf()

    def _validate_postprocess_min_conf(self):
        for group_name, value in self.postprocess_min_conf_by_group.items():
            if group_name not in self.class_groups:
                raise ValueError(
                    "postprocess.min_conf_by_group contains unknown group "
                    f"'{group_name}'."
                )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"postprocess.min_conf_by_group.{group_name} must be in [0.0, 1.0]."
                )


def _parse_top_k(value):
    if isinstance(value, str):
        normalized = value.lower().strip()

        if normalized == "all":
            return "all"

        try:
            return int(normalized)
        except ValueError as exc:
            raise ValueError("top_k must be a positive integer or 'all'.") from exc

    return int(value)
