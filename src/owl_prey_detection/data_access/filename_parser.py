from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class ParsedVideoName:
    location: str
    camera: str
    stream: str
    start_time: datetime
    end_time: datetime


def parse_video_filename(filename: str) -> ParsedVideoName | None:
    name = Path(filename).stem
    parts = name.split("_")

    if len(parts) < 5:
        return None

    try:
        return ParsedVideoName(
            location=parts[0],
            camera=parts[1],
            stream=parts[2],
            start_time=datetime.strptime(parts[3], "%Y%m%d%H%M%S"),
            end_time=datetime.strptime(parts[4], "%Y%m%d%H%M%S"),
        )
    except Exception:
        return None