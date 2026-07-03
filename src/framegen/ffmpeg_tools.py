from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from .types import VideoInfo


class FFmpegError(RuntimeError):
    """Raised when ffmpeg tooling fails."""


def ensure_binary(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise FFmpegError(
            f"Required binary '{name}' was not found on PATH. Install FFmpeg and ensure "
            f"'{name}' is available."
        )
    return resolved


def resolve_binary(name: str, explicit_path: str | None = None) -> str:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.exists():
            raise FFmpegError(f"Configured binary for '{name}' does not exist: '{candidate}'.")
        return str(candidate)
    return ensure_binary(name)


def probe_video(path: Path, ffprobe_path: str | None = None) -> VideoInfo:
    ffprobe = resolve_binary("ffprobe", ffprobe_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffprobe failed.")

    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if not video_stream:
        raise FFmpegError(f"No video stream found in '{path}'.")

    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = Fraction(fps_raw)
    duration = video_stream.get("duration") or payload.get("format", {}).get("duration")
    nb_frames = video_stream.get("nb_frames")

    return VideoInfo(
        path=path,
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=fps,
        duration_seconds=float(duration) if duration is not None else None,
        frame_count=int(nb_frames) if nb_frames is not None and nb_frames.isdigit() else None,
        has_audio=audio_stream is not None,
    )


def remux_audio(
    source_video: Path,
    interpolated_video: Path,
    output_path: Path,
    ffmpeg_path: str | None = None,
) -> None:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_path)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(interpolated_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg audio remux failed.")
