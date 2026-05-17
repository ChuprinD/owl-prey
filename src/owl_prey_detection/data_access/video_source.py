from __future__ import annotations

from pathlib import Path, PurePosixPath

from owl_prey_detection.config import PipelineConfig


def _blob_name_to_local_path(
    blob_name: str,
    download_dir: Path,
) -> Path:
    from owl_prey_detection.data_access.azure_blob import normalize_blob_name

    normalized_blob_name = normalize_blob_name(blob_name)
    blob_path = PurePosixPath(normalized_blob_name)

    if blob_path.is_absolute():
        raise ValueError(f"blob_name must be relative, got: {blob_name}")

    return download_dir.joinpath(*blob_path.parts)


def resolve_input_video(config: PipelineConfig) -> Path:
    if config.input_source == "local":
        if config.video_path is None:
            raise ValueError("input.video_path must be set when source is local.")

        if not config.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {config.video_path}")

        return config.video_path

    if config.input_source == "azure_blob":
        if not config.blob_name:
            raise ValueError("input.blob_name must be set when source is azure_blob.")

        from owl_prey_detection.data_access.azure_blob import AzureBlobClient

        destination_path = _blob_name_to_local_path(
            blob_name=config.blob_name,
            download_dir=config.download_dir,
        )

        client = AzureBlobClient()

        return client.download_blob(
            blob_name=config.blob_name,
            destination_path=destination_path,
            overwrite=config.overwrite_download,
        )

    raise ValueError(f"Unknown input source: {config.input_source}")