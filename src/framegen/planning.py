from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from .types import InterpolationPlan, VideoInfo


def build_plan(
    video: VideoInfo,
    output_path: Path,
    backend: str,
    target_fps: Fraction = Fraction(60, 1),
    preserve_audio: bool = True,
) -> InterpolationPlan:
    if video.fps <= 0:
        raise ValueError("Source FPS must be greater than zero.")
    if target_fps <= video.fps:
        raise ValueError("Target FPS must be higher than source FPS for interpolation.")

    interpolation_factor = float(target_fps / video.fps)
    notes: list[str] = []

    if backend in {"rife-cli", "film-cli"}:
        ai_pass_target_fps = target_fps
        notes.append("Backend supports direct target FPS generation.")
    elif backend == "rife-double":
        doubled_fps = video.fps
        passes = 0
        while doubled_fps < target_fps:
            doubled_fps *= 2
            passes += 1
        ai_pass_target_fps = doubled_fps
        notes.append(f"Using {passes} doubling pass(es), then decimating to target FPS if needed.")
    elif backend == "ffmpeg-minterpolate":
        ai_pass_target_fps = target_fps
        notes.append("Fallback is classical motion interpolation, not a neural model.")
    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

    if video.fps == Fraction(24, 1):
        notes.append("24fps film content usually benefits from scene-cut detection and conservative sharpening.")
    elif video.fps in {Fraction(24000, 1001), Fraction(30000, 1001)}:
        notes.append("NTSC-derived source detected; preserve exact fractional timing through the pipeline.")
    elif video.fps == Fraction(30, 1):
        notes.append("30fps video can usually map cleanly to 60fps with 1 synthesized frame between originals.")

    return InterpolationPlan(
        input_path=video.path,
        output_path=output_path,
        source_fps=video.fps,
        target_fps=target_fps,
        backend=backend,
        interpolation_factor=interpolation_factor,
        ai_pass_target_fps=ai_pass_target_fps,
        preserve_audio=preserve_audio and video.has_audio,
        notes=tuple(notes),
    )
