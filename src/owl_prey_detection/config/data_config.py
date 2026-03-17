from __future__ import annotations

from pathlib import Path
import yaml


def load_data_config() -> dict:
    config_path = Path(__file__).resolve().parents[3] / "configs" / "data.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)