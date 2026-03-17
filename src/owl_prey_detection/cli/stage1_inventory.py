from __future__ import annotations

from pprint import pprint

from owl_prey_detection.data_access.azure_blob import AzureBlobClient
from owl_prey_detection.data_access.inventory import BlobInventoryBuilder
from owl_prey_detection.utils.paths import BLOB_INDEX_DIR, ensure_directories


def main() -> None:
    print("Stage 1: blob inventory started")

    ensure_directories()

    azure_client = AzureBlobClient()
    inventory_builder = BlobInventoryBuilder(azure_client=azure_client)

    print("Reading blob metadata from Azure...")
    df = inventory_builder.build_dataframe()

    all_blobs_csv_path = BLOB_INDEX_DIR / "blob_index.csv"
    video_blobs_csv_path = BLOB_INDEX_DIR / "video_index.csv"

    inventory_builder.save_to_csv(df, all_blobs_csv_path)
    inventory_builder.save_to_csv(df[df["is_video"] == True], video_blobs_csv_path)

    summary = inventory_builder.summarize(df)

    print("Inventory saved successfully")
    print(f"All blobs CSV: {all_blobs_csv_path}")
    print(f"Video blobs CSV: {video_blobs_csv_path}")
    pprint(summary)


if __name__ == "__main__":
    main()