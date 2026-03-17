from __future__ import annotations

from pathlib import Path
from typing import Iterable

from azure.storage.blob import BlobProperties, BlobServiceClient

from owl_prey_detection.config.settings import (
    AZURE_CONNECTION_STRING,
    AZURE_CONTAINER_NAME,
)


class AzureBlobClient:
    def __init__(
        self,
        connection_string: str = AZURE_CONNECTION_STRING,
        container_name: str = AZURE_CONTAINER_NAME,
    ) -> None:
        self._service_client = BlobServiceClient.from_connection_string(connection_string)
        self._container_client = self._service_client.get_container_client(container_name)

    def list_blobs(self) -> Iterable[BlobProperties]:
        return self._container_client.list_blobs()

    def download_blob(self, blob_name: str, destination_path: Path) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        blob_client = self._container_client.get_blob_client(blob_name)

        with destination_path.open("wb") as file:
            stream = blob_client.download_blob()
            file.write(stream.readall())