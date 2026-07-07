from __future__ import annotations

import json
import math
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
    mux_audio_track(
        interpolated_video=interpolated_video,
        audio_source=source_video,
        output_path=output_path,
        ffmpeg_path=ffmpeg_path,
    )


def mux_audio_track(
    interpolated_video: Path,
    audio_source: Path,
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
        str(audio_source),
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


def encode_video(
    source_video: Path,
    output_path: Path,
    ffmpeg_path: str | None = None,
    video_codec: str = "h264_nvenc",
    video_crf: str = "18",
    video_preset: str = "p5",
) -> None:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_path)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-c:v",
        video_codec,
    ]
    if video_codec != "copy":
        if video_codec in {"h264_nvenc", "hevc_nvenc"}:
            command.extend(
                [
                    "-preset",
                    video_preset,
                    "-rc",
                    "vbr",
                    "-cq",
                    video_crf,
                    "-b:v",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ]
            )
        else:
            command.extend(
                [
                    "-preset",
                    video_preset,
                    "-crf",
                    video_crf,
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ]
            )
    command.extend(
        [
            "-an",
            str(output_path),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg encode failed.")


def extract_video_segment(
    source_video: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_path: str | None = None,
) -> None:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration_seconds:.6f}",
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg segment extraction failed.")


def extract_audio_segment(
    source_video: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    ffmpeg_path: str | None = None,
) -> None:
    ffmpeg = resolve_binary("ffmpeg", ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration_seconds:.6f}",
        "-map",
        "0:a:0?",
        "-vn",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg audio segment extraction failed.")


def split_video_into_segments(
    source_video: Path,
    output_dir: Path,
    segment_duration_seconds: float | None = None,
    segment_count: int | None = None,
    ffprobe_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> list[Path]:
    if segment_duration_seconds is not None and segment_duration_seconds <= 0:
        raise FFmpegError("Segment duration must be greater than zero.")
    if segment_count is not None and segment_count <= 0:
        raise FFmpegError("Segment count must be greater than zero.")

    video = probe_video(source_video, ffprobe_path=ffprobe_path)
    if video.duration_seconds is None or video.duration_seconds <= 0:
        raise FFmpegError("Could not determine source duration for resumable segmentation.")

    if segment_duration_seconds is None:
        resolved_segment_count = segment_count or 1
        segment_duration_seconds = video.duration_seconds / resolved_segment_count
    else:
        resolved_segment_count = max(1, math.ceil(video.duration_seconds / segment_duration_seconds))

    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    for index in range(resolved_segment_count):
        start_seconds = index * segment_duration_seconds
        remaining = max(video.duration_seconds - start_seconds, 0.0)
        chunk_duration = min(segment_duration_seconds, remaining) if index < resolved_segment_count - 1 else remaining
        if chunk_duration <= 0:
            continue
        output_path = output_dir / f"source_{index:04d}.mp4"
        if not _is_valid_segment(output_path, ffprobe_path):
            extract_video_segment(
                source_video,
                output_path,
                start_seconds=start_seconds,
                duration_seconds=chunk_duration,
                ffmpeg_path=ffmpeg_path,
            )
        segments.append(output_path)
    return segments


def split_audio_into_segments(
    source_video: Path,
    output_dir: Path,
    segment_duration_seconds: float | None = None,
    segment_count: int | None = None,
    ffprobe_path: str | None = None,
    ffmpeg_path: str | None = None,
) -> list[Path]:
    if segment_duration_seconds is not None and segment_duration_seconds <= 0:
        raise FFmpegError("Segment duration must be greater than zero.")
    if segment_count is not None and segment_count <= 0:
        raise FFmpegError("Segment count must be greater than zero.")

    video = probe_video(source_video, ffprobe_path=ffprobe_path)
    if not video.has_audio:
        return []
    if video.duration_seconds is None or video.duration_seconds <= 0:
        raise FFmpegError("Could not determine source duration for resumable segmentation.")

    if segment_duration_seconds is None:
        resolved_segment_count = segment_count or 1
        segment_duration_seconds = video.duration_seconds / resolved_segment_count
    else:
        resolved_segment_count = max(1, math.ceil(video.duration_seconds / segment_duration_seconds))

    output_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    for index in range(resolved_segment_count):
        start_seconds = index * segment_duration_seconds
        remaining = max(video.duration_seconds - start_seconds, 0.0)
        chunk_duration = min(segment_duration_seconds, remaining) if index < resolved_segment_count - 1 else remaining
        if chunk_duration <= 0:
            continue
        output_path = output_dir / f"audio_{index:04d}.mka"
        if not _is_valid_audio_segment(output_path, ffprobe_path):
            extract_audio_segment(
                source_video,
                output_path,
                start_seconds=start_seconds,
                duration_seconds=chunk_duration,
                ffmpeg_path=ffmpeg_path,
            )
        segments.append(output_path)
    return segments


def _is_valid_segment(path: Path, ffprobe_path: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        video = probe_video(path, ffprobe_path=ffprobe_path)
    except (FFmpegError, OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(video.duration_seconds is None or video.duration_seconds > 0 or video.frame_count)


def _is_valid_audio_segment(path: Path, ffprobe_path: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    ffprobe = resolve_binary("ffprobe", ffprobe_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return False
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    streams = payload.get("streams", [])
    return any(stream.get("codec_type") == "audio" for stream in streams)


def concat_videos(
    source_videos: list[Path],
    output_path: Path,
    ffmpeg_path: str | None = None,
) -> None:
    concat_media(source_videos, output_path, ffmpeg_path=ffmpeg_path)


def concat_media(
    source_paths: list[Path],
    output_path: Path,
    ffmpeg_path: str | None = None,
) -> None:
    if not source_paths:
        raise FFmpegError("At least one source file is required for concatenation.")

    ffmpeg = resolve_binary("ffmpeg", ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.parent / "concat.txt"
    manifest_lines = []
    for path in source_paths:
        escaped_path = str(path).replace("'", "''")
        manifest_lines.append(f"file '{escaped_path}'")
    manifest = "\n".join(manifest_lines) + "\n"
    list_path.write_text(manifest, encoding="utf-8")
    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FFmpegError(result.stderr.strip() or "ffmpeg concat failed.")
