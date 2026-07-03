from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    duration_seconds: float | None
    frame_count: int | None
    has_audio: bool


@dataclass(frozen=True)
class InterpolationPlan:
    input_path: Path
    output_path: Path
    source_fps: Fraction
    target_fps: Fraction
    backend: str
    interpolation_factor: float
    ai_pass_target_fps: Fraction
    preserve_audio: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class BackendCommand:
    argv: tuple[str, ...]
    description: str
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
