import os
from pathlib import PurePosixPath
from urllib.parse import unquote, urlparse

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


def normalize_blob_name(blob_name):
    blob_name = blob_name.strip().replace("\\", "/")

    parsed = urlparse(blob_name)

    if parsed.scheme in {"http", "https"}:
        parts = [part for part in parsed.path.split("/") if part]

        if len(parts) < 2:
            raise ValueError(f"Could not extract blob name from URL: {blob_name}")

        blob_name = "/".join(parts[1:])

    blob_name = unquote(blob_name)
    blob_name = blob_name.lstrip("/")

    return blob_name


class AzureBlobClient:
    def __init__(
        self,
        connection_string=None,
        container_name=None,
    ):
        load_dotenv()

        if connection_string is None:
            connection_string = os.getenv("AZURE_CONNECTION_STRING")

        if container_name is None:
            container_name = os.getenv("AZURE_CONTAINER_NAME")

        if not connection_string:
            raise RuntimeError(
                "Environment variable AZURE_CONNECTION_STRING is not set."
            )

        if not container_name:
            raise RuntimeError(
                "Environment variable AZURE_CONTAINER_NAME is not set."
            )

        self._container_name = container_name

        self._service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        self._container_client = self._service_client.get_container_client(
            container_name
        )

    def list_blobs(
        self,
        name_starts_with=None,
    ):
        return self._container_client.list_blobs(
            name_starts_with=name_starts_with
        )

    def blob_exists(self, blob_name):
        blob_name = normalize_blob_name(blob_name)
        blob_client = self._container_client.get_blob_client(blob_name)

        return blob_client.exists()

    def find_blob_names_by_filename(
        self,
        filename,
        max_scan=100_000,
    ):
        target_filename = PurePosixPath(filename).name.lower()

        matches = []
        scanned = 0

        for blob in self.list_blobs():
            scanned += 1

            blob_filename = PurePosixPath(blob.name).name.lower()

            if blob_filename == target_filename:
                matches.append(blob.name)

            if scanned >= max_scan:
                break

        return matches

    def resolve_blob_name(self, blob_name):
        requested_blob_name = normalize_blob_name(blob_name)

        if self.blob_exists(requested_blob_name):
            return requested_blob_name

        filename = PurePosixPath(requested_blob_name).name
        matches = self.find_blob_names_by_filename(filename)

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            candidates = "\n".join(f"  - {name}" for name in matches[:20])

            raise FileNotFoundError(
                "Blob filename is ambiguous. Use full blob path in config.yaml.\n"
                f"Requested filename: {filename}\n"
                f"Container: {self._container_name}\n\n"
                f"Candidates:\n{candidates}"
            )

        raise FileNotFoundError(
            "Azure blob was not found.\n"
            f"Container: {self._container_name}\n"
            f"Requested blob_name: {requested_blob_name}\n\n"
            "Use the exact full blob path inside the container, or a unique filename."
        )

    def download_blob(
        self,
        blob_name,
        destination_path,
        overwrite=False,
    ):
        resolved_blob_name = self.resolve_blob_name(blob_name)

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists() and not overwrite:
            return destination_path

        blob_client = self._container_client.get_blob_client(resolved_blob_name)

        try:
            with destination_path.open("wb") as file:
                stream = blob_client.download_blob()
                stream.readinto(file)

        except ResourceNotFoundError as exc:
            raise FileNotFoundError(
                "Azure blob could not be downloaded.\n"
                f"Container: {self._container_name}\n"
                f"Resolved blob_name: {resolved_blob_name}"
            ) from exc

        return destination_path
