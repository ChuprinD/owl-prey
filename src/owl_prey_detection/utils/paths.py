from __future__ import annotations

from pathlib import Path

from owl_prey_detection.config.path_config import PROJECT_ROOT, load_paths_config


_CONFIG = load_paths_config()

DATA_RAW_DIR = PROJECT_ROOT / _CONFIG["data"]["raw_dir"]
PILOT_VIDEOS_DIR = PROJECT_ROOT / _CONFIG["data"]["pilot_videos_dir"]

DATA_INTERIM_DIR = PROJECT_ROOT / _CONFIG["data"]["interim_dir"]
BLOB_INDEX_DIR = PROJECT_ROOT / _CONFIG["data"]["blob_index_dir"]
METADATA_TABLES_DIR = DATA_INTERIM_DIR / "metadata_tables"

DATA_PROCESSED_DIR = PROJECT_ROOT / _CONFIG["data"]["processed_dir"]

REPORTS_DIR = PROJECT_ROOT / _CONFIG["reports"]["reports_dir"]
FIGURES_DIR = PROJECT_ROOT / _CONFIG["reports"]["figures_dir"]
TABLES_DIR = PROJECT_ROOT / _CONFIG["reports"]["tables_dir"]

LOGS_DIR = PROJECT_ROOT / _CONFIG["logs"]["logs_dir"]


def ensure_directories() -> None:
    directories: list[Path] = [
        DATA_RAW_DIR,
        PILOT_VIDEOS_DIR,
        DATA_INTERIM_DIR,
        BLOB_INDEX_DIR,
        METADATA_TABLES_DIR,
        DATA_PROCESSED_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)