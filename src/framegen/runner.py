from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

from .backends import build_backend_command, env_backend_template
from .ffmpeg_tools import (
    FFmpegError,
    concat_media,
    concat_videos,
    encode_video,
    mux_audio_track,
    probe_video,
    remux_audio,
    split_audio_into_segments,
    split_video_into_segments,
)
from .types import InterpolationPlan


class PipelineError(RuntimeError):
    """Raised when the interpolation pipeline fails."""


SOURCE_CHUNKS_DIRNAME = "source_chunks"
AUDIO_CHUNKS_DIRNAME = "audio_chunks"
SOURCE_CHUNKS_MANIFEST = "source_chunks_manifest.json"


def run_plan(plan: InterpolationPlan, work_dir: Path) -> Path:
    return run_plan_with_options(plan=plan, work_dir=work_dir, backend_options={})


def run_plan_with_options(
    plan: InterpolationPlan,
    work_dir: Path,
    backend_options: dict[str, str],
) -> Path:
    if backend_options.get("resume") == "true":
        return _run_plan_with_resume(plan=plan, work_dir=work_dir, backend_options=backend_options)
    return _run_single_plan(plan=plan, work_dir=work_dir, backend_options=backend_options, cleanup_work_dir=True)


def _run_single_plan(
    plan: InterpolationPlan,
    work_dir: Path,
    backend_options: dict[str, str],
    cleanup_work_dir: bool,
) -> Path:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    backend_cmd = build_backend_command(
        plan=plan,
        work_dir=work_dir,
        backend_command_template=env_backend_template(plan.backend),
        backend_options=backend_options,
    )

    print(f"[framegen] Starting backend: {backend_cmd.description}")
    print(f"[framegen] Command: {' '.join(backend_cmd.argv)}")
    return_code = _run_streaming_command(
        backend_cmd.argv,
        cwd=str(backend_cmd.cwd) if backend_cmd.cwd else None,
        env=dict(backend_cmd.env) if backend_cmd.env else None,
    )
    if return_code != 0:
        raise PipelineError(
            f"{backend_cmd.description} failed.\nCommand: {' '.join(backend_cmd.argv)}"
        )

    temp_output = work_dir / "interpolated_noaudio.mp4"
    if not temp_output.exists():
        raise PipelineError(
            f"Backend completed without creating expected file '{temp_output}'. "
            "Update the command template to match your backend."
        )

    encoded_output = temp_output
    if backend_options.get("video_codec", "libx264") != "copy":
        encoded_output = work_dir / "interpolated_encoded.mp4"
        encode_video(
            temp_output,
            encoded_output,
            ffmpeg_path=backend_options.get("ffmpeg_exe"),
            video_codec=backend_options.get("video_codec", "libx264"),
            video_crf=backend_options.get("video_crf", "18"),
            video_preset=backend_options.get("video_preset", "slow"),
        )

    if plan.preserve_audio:
        remux_audio(
            plan.input_path,
            encoded_output,
            plan.output_path,
            ffmpeg_path=backend_options.get("ffmpeg_exe"),
        )
    else:
        encoded_output.replace(plan.output_path)

    if cleanup_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    return plan.output_path


def _run_plan_with_resume(
    plan: InterpolationPlan,
    work_dir: Path,
    backend_options: dict[str, str],
) -> Path:
    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    temp_output = work_dir / "interpolated_noaudio.mp4"
    encoded_output = work_dir / "interpolated_encoded.mp4"
    ffmpeg_exe = backend_options.get("ffmpeg_exe")
    ffprobe_exe = backend_options.get("ffprobe_exe")

    if _is_valid_video(plan.output_path, ffprobe_exe):
        return plan.output_path

    state_path = work_dir / "resume_state.json"
    chunk_duration_raw = backend_options.get("chunk_duration_seconds", "")
    chunk_duration_seconds = float(chunk_duration_raw) if chunk_duration_raw else None
    chunk_count = int(backend_options.get("chunk_count", "10"))
    state_payload = {
        "input_path": str(plan.input_path.resolve()),
        "output_path": str(plan.output_path.resolve()),
        "backend": plan.backend,
        "target_fps": str(plan.target_fps),
        "ai_pass_target_fps": str(plan.ai_pass_target_fps),
        "chunk_count": chunk_count,
        "chunk_duration_seconds": chunk_duration_seconds,
        "video_codec": backend_options.get("video_codec", "libx264"),
        "video_crf": backend_options.get("video_crf", "18"),
        "video_preset": backend_options.get("video_preset", "slow"),
    }
    _validate_or_write_resume_state(state_path, state_payload)
    source_chunks: list[Path] = []
    audio_chunks: list[Path] = []
    curated_chunks = False

    if not _is_valid_video(temp_output, ffprobe_exe):
        source_chunks, audio_chunks, curated_chunks = get_or_create_source_chunks(
            source_video=plan.input_path,
            work_dir=work_dir,
            ffprobe_exe=ffprobe_exe,
            ffmpeg_exe=ffmpeg_exe,
            chunk_duration_seconds=chunk_duration_seconds,
            chunk_count=chunk_count,
        )
        output_chunks_dir = work_dir / "output_chunks"
        output_chunks_dir.mkdir(parents=True, exist_ok=True)
        single_run_options = {
            key: value
            for key, value in backend_options.items()
            if key not in {"resume", "chunk_count", "chunk_duration_seconds"}
        }
        chunk_outputs: list[Path] = []
        for index, source_chunk in enumerate(source_chunks):
            chunk_output = output_chunks_dir / f"interpolated_{index:04d}.mp4"
            chunk_outputs.append(chunk_output)
            if _is_valid_video(chunk_output, ffprobe_exe):
                continue
            chunk_plan = replace(
                plan,
                input_path=source_chunk,
                output_path=chunk_output,
                preserve_audio=False,
            )
            _run_single_plan(
                plan=chunk_plan,
                work_dir=work_dir / f"chunk_work_{index:04d}",
                backend_options=single_run_options,
                cleanup_work_dir=True,
            )

        concat_videos(chunk_outputs, temp_output, ffmpeg_path=ffmpeg_exe)

    encoded_path = temp_output
    if backend_options.get("video_codec", "libx264") != "copy":
        if not _is_valid_video(encoded_output, ffprobe_exe):
            encode_video(
                temp_output,
                encoded_output,
                ffmpeg_path=ffmpeg_exe,
                video_codec=backend_options.get("video_codec", "libx264"),
                video_crf=backend_options.get("video_crf", "18"),
                video_preset=backend_options.get("video_preset", "slow"),
            )
        encoded_path = encoded_output

    if plan.preserve_audio:
        if curated_chunks or audio_chunks:
            selected_audio_chunks = _select_audio_chunks_for_source_chunks(source_chunks, audio_chunks)
            if not selected_audio_chunks:
                raise PipelineError(
                    "No matching audio chunks were found for the selected source chunks. "
                    "Run split again or rerun with --no-audio."
                )
            concatenated_audio = work_dir / "selected_audio.mka"
            concat_media(selected_audio_chunks, concatenated_audio, ffmpeg_path=ffmpeg_exe)
            mux_audio_track(
                interpolated_video=encoded_path,
                audio_source=concatenated_audio,
                output_path=plan.output_path,
                ffmpeg_path=ffmpeg_exe,
            )
        else:
            remux_audio(
                plan.input_path,
                encoded_path,
                plan.output_path,
                ffmpeg_path=ffmpeg_exe,
            )
    else:
        encoded_path.replace(plan.output_path)
    return plan.output_path


def _validate_or_write_resume_state(path: Path, payload: dict[str, object]) -> None:
    if not path.exists():
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != payload:
        raise PipelineError(
            f"Resume state mismatch in '{path}'. Reuse the same input/output/settings or choose a new work dir."
        )


def _is_valid_video(path: Path, ffprobe_path: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        video = probe_video(path, ffprobe_path=ffprobe_path)
    except (FFmpegError, OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(video.duration_seconds is None or video.duration_seconds > 0 or video.frame_count)


def _run_streaming_command(
    argv: tuple[str, ...],
    cwd: str | None,
    env: dict[str, str] | None,
) -> int:
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=False,
    )
    return result.returncode


def get_or_create_source_chunks(
    source_video: Path,
    work_dir: Path,
    ffprobe_exe: str | None,
    ffmpeg_exe: str | None,
    chunk_duration_seconds: float | None,
    chunk_count: int,
) -> tuple[list[Path], list[Path], bool]:
    source_chunks_dir = work_dir / SOURCE_CHUNKS_DIRNAME
    existing_chunks = _list_valid_source_chunks(source_chunks_dir, ffprobe_exe)
    audio_chunks_dir = work_dir / AUDIO_CHUNKS_DIRNAME
    existing_audio_chunks = _list_valid_audio_chunks(audio_chunks_dir, ffprobe_exe)
    manifest = _load_source_chunks_manifest(work_dir)
    if existing_chunks:
        curated = _is_curated_chunk_set(existing_chunks, manifest, source_video)
        print(f"[framegen] Using existing source chunks: {len(existing_chunks)} chunk(s)")
        return existing_chunks, existing_audio_chunks, curated

    source_chunks = split_video_into_segments(
        source_video,
        source_chunks_dir,
        segment_duration_seconds=chunk_duration_seconds,
        segment_count=chunk_count if chunk_duration_seconds is None else None,
        ffprobe_path=ffprobe_exe,
        ffmpeg_path=ffmpeg_exe,
    )
    audio_chunks = split_audio_into_segments(
        source_video,
        audio_chunks_dir,
        segment_duration_seconds=chunk_duration_seconds,
        segment_count=chunk_count if chunk_duration_seconds is None else None,
        ffprobe_path=ffprobe_exe,
        ffmpeg_path=ffmpeg_exe,
    )
    _write_source_chunks_manifest(work_dir, source_video, source_chunks, audio_chunks)
    print(f"[framegen] Prepared source chunks: {len(source_chunks)} chunk(s)")
    if audio_chunks:
        print(f"[framegen] Prepared audio chunks: {len(audio_chunks)} chunk(s)")
    return source_chunks, audio_chunks, False


def prepare_source_chunks(
    source_video: Path,
    work_dir: Path,
    ffprobe_exe: str | None,
    ffmpeg_exe: str | None,
    chunk_duration_seconds: float | None,
    chunk_count: int,
) -> list[Path]:
    source_chunks, _, _ = get_or_create_source_chunks(
        source_video=source_video,
        work_dir=work_dir,
        ffprobe_exe=ffprobe_exe,
        ffmpeg_exe=ffmpeg_exe,
        chunk_duration_seconds=chunk_duration_seconds,
        chunk_count=chunk_count,
    )
    return source_chunks


def _list_valid_source_chunks(source_chunks_dir: Path, ffprobe_exe: str | None) -> list[Path]:
    if not source_chunks_dir.exists():
        return []
    chunks = sorted(source_chunks_dir.glob("source_*.mp4"))
    return [chunk for chunk in chunks if _is_valid_video(chunk, ffprobe_exe)]


def _list_valid_audio_chunks(audio_chunks_dir: Path, ffprobe_exe: str | None) -> list[Path]:
    if not audio_chunks_dir.exists():
        return []
    chunks = sorted(audio_chunks_dir.glob("audio_*.mka"))
    return [chunk for chunk in chunks if chunk.exists() and chunk.stat().st_size > 0]


def _load_source_chunks_manifest(work_dir: Path) -> dict[str, object] | None:
    manifest_path = work_dir / SOURCE_CHUNKS_MANIFEST
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_source_chunks_manifest(
    work_dir: Path,
    source_video: Path,
    source_chunks: list[Path],
    audio_chunks: list[Path],
) -> None:
    manifest_path = work_dir / SOURCE_CHUNKS_MANIFEST
    payload = {
        "input_path": str(source_video.resolve()),
        "chunk_names": [chunk.name for chunk in source_chunks],
        "audio_chunk_names": [chunk.name for chunk in audio_chunks],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _is_curated_chunk_set(chunks: list[Path], manifest: dict[str, object] | None, source_video: Path) -> bool:
    if manifest is None:
        return False
    if manifest.get("input_path") != str(source_video.resolve()):
        return True
    manifest_names = manifest.get("chunk_names")
    if not isinstance(manifest_names, list):
        return False
    current_names = [chunk.name for chunk in chunks]
    return current_names != manifest_names


def _select_audio_chunks_for_source_chunks(source_chunks: list[Path], audio_chunks: list[Path]) -> list[Path]:
    audio_by_index = {_chunk_index(path.name): path for path in audio_chunks}
    selected: list[Path] = []
    for source_chunk in source_chunks:
        chunk_index = _chunk_index(source_chunk.name)
        if chunk_index not in audio_by_index:
            raise PipelineError(
                f"Missing audio chunk for source chunk '{source_chunk.name}'. "
                "Restore the matching audio chunk or rerun with --no-audio."
            )
        selected.append(audio_by_index[chunk_index])
    return selected


def _chunk_index(filename: str) -> str:
    stem = Path(filename).stem
    return stem.split("_")[-1]
