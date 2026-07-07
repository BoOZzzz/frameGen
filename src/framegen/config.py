from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(".framegen.json")


@dataclass
class AppConfig:
    python_exe: str | None = None
    ffmpeg_exe: str | None = None
    ffprobe_exe: str | None = None
    rife_python_exe: str | None = None
    rife_dir: str | None = None
    rife_model_dir: str | None = None
    video_codec: str | None = None
    video_crf: str | None = None
    video_preset: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return AppConfig(**payload)


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")


def merge_config(base: AppConfig, updates: dict[str, str | None]) -> AppConfig:
    merged = base.to_dict()
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    return AppConfig(**merged)
