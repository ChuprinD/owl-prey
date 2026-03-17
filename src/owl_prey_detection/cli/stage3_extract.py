from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from pathlib import Path

from azure.core.exceptions import ResourceExistsError

from owl_prey_detection.config.data_config import load_data_config
from owl_prey_detection.config.settings import AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME
from owl_prey_detection.data_access.azure_blob import AzureBlobClient
from owl_prey_detection.data_processing.extraction_worker import process_blob_mp
from owl_prey_detection.utils.paths import EXTRACTION_CHECKPOINT_DIR, EXTRACTION_OUTPUT_DIR, EXTRACTION_TEMP_DIR, ensure_directories


def load_checkpoint(checkpoint_file: Path) -> set[str]:
    """Loads a set of successfully processed blobs for resumption."""
    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            return set(json.load(f))
    return set()


def save_checkpoint(checkpoint_file: Path, done: set[str]) -> None:
    """Saves the set of successfully processed blobs."""
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(list(done), f)


def list_rawdata_blobs(client: AzureBlobClient, location_filter: str, month_filter: str | None = None) -> list[str]:
    """Retrieves MP4 blob names prefixed by the location, optionally filtered by month (e.g. '202503')."""
    all_blobs = client.list_blobs()
    selected_blobs = []
    
    for b in all_blobs:
        name = b.name
        if not name.startswith(location_filter + "/") or not name.lower().endswith(".mp4") or b.size <= 0:
            continue
            
        # Optional: check if the month string exists in the filename
        # File format example: Vosseveld_ch1_main_20250401013303_...mp4
        if month_filter and month_filter not in name:
            continue
            
        selected_blobs.append(name)
        
    return selected_blobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from rawdata videos")
    parser.add_argument("--location", type=str, required=True, help="E.g., Vosseveld, Haarle, Meddo")
    parser.add_argument("--month", type=str, default=None, help="Filter by YearMonth (e.g., '202503' for March)")
    parser.add_argument("--max-workers", type=int, default=None, help="Force CPU threads override")
    args = parser.parse_args()
    location_filter = args.location
    month_filter = args.month

    ensure_directories()
    
    # Reload paths properly checking defaults
    output_dir = EXTRACTION_OUTPUT_DIR
    temp_dir = EXTRACTION_TEMP_DIR
    
    # Differentiate checkpoint name if a month filter is applied
    ckpt_name = f"checkpoint_rawdata_{location_filter}_{month_filter}.json" if month_filter else f"checkpoint_rawdata_{location_filter}.json"
    checkpoint_file = EXTRACTION_CHECKPOINT_DIR / ckpt_name
    manifest_file = output_dir / "manifest_rawdata.csv"

    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extraction Output Directory: {output_dir}")
    print(f"Extraction Temp Directory: {temp_dir}")
    
    # Disk usage check
    _, _, free = shutil.disk_usage(str(output_dir.drive) + "\\")
    print(f"Free Disk Space: {free / 1024 ** 3:.0f} GB")

    data_config = load_data_config()
    ex_cfg = data_config.get("extraction", {})
    skip_frames = ex_cfg.get("skip_frames", 4)
    motion_threshold = ex_cfg.get("motion_threshold", 25)
    min_motion_pixels = ex_cfg.get("min_motion_pixels", 5000)
    min_sharpness = ex_cfg.get("min_sharpness", 40)
    jpeg_quality = ex_cfg.get("jpeg_quality", 92)
    max_workers_offset = ex_cfg.get("max_workers_offset", 2)

    cpu_count = multiprocessing.cpu_count()
    max_workers = args.max_workers or max(1, cpu_count - max_workers_offset)

    azure_client = AzureBlobClient()

    print(f"\nConnecting to container '{AZURE_CONTAINER_NAME}' to list '{location_filter}' videos...")
    
    done_blobs = load_checkpoint(checkpoint_file)
    all_blob_names = list_rawdata_blobs(azure_client, location_filter, month_filter)
    remaining_blob_names = [name for name in all_blob_names if name not in done_blobs]

    print(f"Total videos found: {len(all_blob_names):,}")
    print(f"Already processed:  {len(done_blobs):,}")
    print(f"Remaining:          {len(remaining_blob_names):,}\n")
    
    if not remaining_blob_names:
        print("Nothing left to process.")
        return

    _worker = partial(
        process_blob_mp,
        connection_string=AZURE_CONNECTION_STRING,
        container_name=AZURE_CONTAINER_NAME,
        output_dir=output_dir,
        temp_dir=temp_dir,
        skip_frames=skip_frames,
        motion_threshold=motion_threshold,
        min_motion_pixels=min_motion_pixels,
        min_sharpness=min_sharpness,
        jpeg_quality=jpeg_quality,
    )

    total_saved = 0
    total_errors = 0
    processed_count = 0
    start_time = datetime.now()

    print(f"Starting {max_workers} processes at {start_time.strftime('%H:%M:%S')}...\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(_worker, blob_name=blob_name): blob_name
            for blob_name in remaining_blob_names
        }

        for future in as_completed(future_to_name):
            blob_name = future_to_name[future]
            processed_count += 1
            short_name = Path(blob_name).name

            try:
                blob_name_ret, n, payload = future.result()
            except Exception as e:
                total_errors += 1
                print(f"[{processed_count:05d}/{len(remaining_blob_names)}] {short_name} -> CRASH: {e}")
                continue

            if n >= 0:
                done_blobs.add(blob_name_ret)
                total_saved += n
                if isinstance(payload, list) and payload:
                    write_header = not manifest_file.exists()
                    with open(manifest_file, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=payload[0].keys())
                        if write_header:
                            w.writeheader()
                        w.writerows(payload)
                print(f"[{processed_count:05d}/{len(remaining_blob_names)}] {short_name} -> {n} frames")
            else:
                total_errors += 1
                print(f"[{processed_count:05d}/{len(remaining_blob_names)}] {short_name} -> ERROR: {payload}")

            if processed_count % 10 == 0:
                save_checkpoint(checkpoint_file, done_blobs)
                elapsed_min = (datetime.now() - start_time).total_seconds() / 60
                rate = processed_count / elapsed_min if elapsed_min > 0 else 0
                eta_hours = (len(remaining_blob_names) - processed_count) / rate / 60 if rate > 0 else 0
                print(f"  --- checkpoint | {elapsed_min:.0f}min | ~{rate:.1f} vids/min | ETA {eta_hours:.1f}h ---")

    save_checkpoint(checkpoint_file, done_blobs)
    elapsed_min = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n{location_filter} completed in {elapsed_min:.0f}min")
    print(f"Frames saved: {total_saved:,}")
    print(f"Errors: {total_errors}")

if __name__ == "__main__":
    main()
