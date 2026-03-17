from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATHS_CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"


def load_paths_config() -> dict:
    with PATHS_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)