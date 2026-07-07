from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, AppConfig, load_config, merge_config, save_config
from .ffmpeg_tools import FFmpegError, probe_video
from .planning import build_plan
from .runner import PipelineError, prepare_source_chunks, run_plan_with_options


DEFAULT_VENDOR_RIFE_DIR = Path("vendor") / "ECCV2022-RIFE"
DEFAULT_VENDOR_RIFE_MODEL_DIR = DEFAULT_VENDOR_RIFE_DIR / "train_log"
DEFAULT_LOCAL_FFMPEG_DIR = Path("vendor") / "ffmpeg"
DEFAULT_INPUT_DIR = Path("input")
DEFAULT_OUTPUT_DIR = Path("output")
RIFE_HD_MODEL_URL = "https://drive.google.com/uc?export=download&id=1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_"
DEFAULT_INTERPOLATION_BACKEND = "rife-cli"
DEFAULT_VIDEO_CODEC = "h264_nvenc"
DEFAULT_VIDEO_CRF = "18"
DEFAULT_VIDEO_PRESET = "p5"
DEFAULT_RESUME_CHUNK_COUNT = 10


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="framegen",
        description="Interpolate movie/video footage from 24/30fps sources to 60fps with pluggable backends.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Inspect a source video and print an interpolation plan.")
    _add_common_args(plan_parser)
    _add_config_arg(plan_parser)

    run_parser = subparsers.add_parser("run", help="Execute an interpolation job.")
    _add_common_args(run_parser)
    _add_config_arg(run_parser)
    run_parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("tmp") / "framegen",
        help="Directory for intermediate files.",
    )
    run_parser.add_argument(
        "--rife-dir",
        type=Path,
        default=DEFAULT_VENDOR_RIFE_DIR,
        help="Path to the vendored or external RIFE checkout.",
    )
    run_parser.add_argument(
        "--rife-model-dir",
        type=Path,
        default=DEFAULT_VENDOR_RIFE_MODEL_DIR,
        help="Directory containing RIFE model weights, typically train_log.",
    )
    run_parser.add_argument(
        "--scale",
        default="1.0",
        choices=["0.25", "0.5", "1.0", "2.0", "4.0"],
        help="RIFE processing scale. Lower values reduce VRAM use for high-resolution sources.",
    )
    run_parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable RIFE fp16 mode when supported by the GPU.",
    )
    run_parser.add_argument(
        "--uhd",
        action="store_true",
        help="Enable RIFE UHD mode, which defaults to a more conservative scale path for 4K.",
    )
    run_parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Use chunked processing and keep intermediate state so rerunning the same command can resume interrupted work. Enabled by default.",
    )
    run_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Disable resumable chunked processing and run the job as one uninterrupted backend pass.",
    )
    run_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk length in seconds when --resume is enabled. Overrides the default split-into-10-chunks behavior.",
    )
    run_parser.add_argument(
        "--chunk-count",
        type=int,
        default=DEFAULT_RESUME_CHUNK_COUNT,
        help="How many chunks resumable mode should split the source into. Defaults to 10.",
    )
    _add_output_encoding_args(run_parser)

    split_parser = subparsers.add_parser("split", help="Prepare source chunks under the work directory without running interpolation.")
    split_parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the source video. Bare filenames default to the input folder.")
    _add_config_arg(split_parser)
    split_parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("tmp") / "framegen",
        help="Directory where source_chunks will be created.",
    )
    split_parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk length in seconds. Overrides the default split-into-10-chunks behavior.",
    )
    split_parser.add_argument(
        "--chunk-count",
        type=int,
        default=DEFAULT_RESUME_CHUNK_COUNT,
        help="How many chunks to split the source into. Defaults to 10.",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Check local backend prerequisites.")
    _add_config_arg(doctor_parser)
    doctor_parser.add_argument(
        "--rife-dir",
        type=Path,
        default=None,
        help="Path to the vendored or external RIFE checkout.",
    )
    doctor_parser.add_argument(
        "--rife-model-dir",
        type=Path,
        default=None,
        help="Directory containing RIFE model weights, typically train_log.",
    )
    doctor_parser.add_argument(
        "--ffmpeg-exe",
        type=Path,
        default=None,
        help="Explicit path to ffmpeg.exe if it is not on PATH.",
    )
    doctor_parser.add_argument(
        "--ffprobe-exe",
        type=Path,
        default=None,
        help="Explicit path to ffprobe.exe if it is not on PATH.",
    )
    doctor_parser.add_argument(
        "--rife-python-exe",
        type=Path,
        default=None,
        help="Explicit path to the Python executable used for the RIFE backend.",
    )

    configure_parser = subparsers.add_parser("configure", help="Save local executable and backend paths.")
    _add_config_arg(configure_parser)
    configure_parser.add_argument("--python-exe", type=Path, default=None, help="Path to the main Python executable.")
    configure_parser.add_argument("--ffmpeg-exe", type=Path, default=None, help="Path to ffmpeg.exe.")
    configure_parser.add_argument("--ffprobe-exe", type=Path, default=None, help="Path to ffprobe.exe.")
    configure_parser.add_argument(
        "--rife-python-exe",
        type=Path,
        default=None,
        help="Path to the Python executable that should run RIFE.",
    )
    configure_parser.add_argument("--rife-dir", type=Path, default=None, help="Path to the RIFE checkout.")
    configure_parser.add_argument(
        "--rife-model-dir",
        type=Path,
        default=None,
        help="Path to the RIFE model directory containing .pkl files.",
    )
    _add_output_encoding_args(configure_parser)

    setup_parser = subparsers.add_parser("setup-rife", help="Install RIFE Python requirements into a chosen Python.")
    _add_config_arg(setup_parser)
    setup_parser.add_argument(
        "--rife-python-exe",
        type=Path,
        default=None,
        help="Path to the Python executable that should host the RIFE dependencies.",
    )
    setup_parser.add_argument(
        "--rife-dir",
        type=Path,
        default=None,
        help="Path to the RIFE checkout.",
    )
    setup_parser.add_argument(
        "--modern-python",
        action="store_true",
        help="Install a Python 3.12-friendly dependency set instead of the repo's pinned requirements.",
    )

    ffmpeg_parser = subparsers.add_parser("setup-ffmpeg", help="Download a local FFmpeg bundle into the workspace.")
    _add_config_arg(ffmpeg_parser)
    ffmpeg_parser.add_argument(
        "--ffmpeg-zip",
        type=Path,
        default=Path("tmp") / "ffmpeg-release-essentials.zip",
        help="Where to download the FFmpeg archive.",
    )
    ffmpeg_parser.add_argument(
        "--install-dir",
        type=Path,
        default=DEFAULT_LOCAL_FFMPEG_DIR,
        help="Where to extract the FFmpeg bundle.",
    )

    model_parser = subparsers.add_parser("download-rife-model", help="Download the default RIFE HD model weights.")
    _add_config_arg(model_parser)
    model_parser.add_argument(
        "--rife-python-exe",
        type=Path,
        default=None,
        help="Python executable used to install and run gdown for model download.",
    )
    model_parser.add_argument(
        "--rife-model-dir",
        type=Path,
        default=None,
        help="Target directory for downloaded RIFE model weights.",
    )
    model_parser.add_argument(
        "--archive-path",
        type=Path,
        default=Path("tmp") / "rife-hd-model.zip",
        help="Where to save the downloaded model archive before extraction.",
    )

    args = parser.parse_args()
    try:
        config = load_config(args.config)

        if args.command == "configure":
            updated = merge_config(
                config,
                {
                    "python_exe": _path_str(args.python_exe),
                    "ffmpeg_exe": _path_str(args.ffmpeg_exe),
                    "ffprobe_exe": _path_str(args.ffprobe_exe),
                    "rife_python_exe": _path_str(args.rife_python_exe),
                    "rife_dir": _path_str(args.rife_dir),
                    "rife_model_dir": _path_str(args.rife_model_dir),
                    "video_codec": getattr(args, "video_codec", None),
                    "video_crf": str(args.video_crf) if getattr(args, "video_crf", None) is not None else None,
                    "video_preset": getattr(args, "video_preset", None),
                },
            )
            save_config(updated, args.config)
            print(json.dumps(updated.to_dict(), indent=2))
            return

        if args.command == "doctor":
            print(json.dumps(_doctor_to_dict(args, config), indent=2))
            return

        if args.command == "setup-rife":
            _setup_rife(args, config)
            return

        if args.command == "setup-ffmpeg":
            _setup_ffmpeg(args, config)
            return

        if args.command == "download-rife-model":
            _download_rife_model(args, config)
            return

        args.input = _resolve_media_path(args.input, DEFAULT_INPUT_DIR, create_parent=False)
        if hasattr(args, "work_dir"):
            args.work_dir = args.work_dir.resolve()

        if args.command == "split":
            _split_source(args, config)
            return

        args.output = _resolve_output_path(args.input, args.output)
        ffprobe_exe = _resolve_configured_value(None, config.ffprobe_exe)
        video = probe_video(args.input, ffprobe_path=ffprobe_exe)
        plan = build_plan(
            video=video,
            output_path=args.output,
            backend=DEFAULT_INTERPOLATION_BACKEND,
            target_fps=Fraction(args.target_fps),
            preserve_audio=not args.no_audio,
        )

        if args.command == "plan":
            print(json.dumps(_plan_to_dict(video, plan), indent=2))
            return

        backend_options = _backend_options_from_args(args, config, plan.backend)
        result = run_plan_with_options(plan=plan, work_dir=args.work_dir, backend_options=backend_options)
        print(f"Created: {result}")
    except (FFmpegError, PipelineError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to the source video. Bare filenames default to the input folder.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path for the interpolated output video. Bare filenames default to the output folder. If omitted, FrameGen writes to output/<input>_60fps.<ext>.",
    )
    parser.add_argument(
        "--target-fps",
        default="60",
        help="Target output FPS as an integer or fraction, for example '60' or '60000/1001'.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio remuxing even when the source contains audio.",
    )


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the FrameGen config file.",
    )


def _add_output_encoding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video-codec",
        choices=["h264_nvenc", "hevc_nvenc", "libx264", "libx265", "copy"],
        default=None,
        help="Final output video codec. Defaults to h264_nvenc for NVIDIA GPU encoding.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=None,
        help="Quality target for final encoding. Lower is larger/higher quality. Defaults to 18.",
    )
    parser.add_argument(
        "--video-preset",
        choices=[
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "p7",
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        default=None,
        help="Encoder preset for final encoding. Defaults to p5 for NVIDIA GPU encoding.",
    )


def _plan_to_dict(video, plan) -> dict[str, object]:
    return {
        "input": str(video.path),
        "resolution": f"{video.width}x{video.height}",
        "source_fps": str(video.fps),
        "target_fps": str(plan.target_fps),
        "backend": plan.backend,
        "interpolation_factor": round(plan.interpolation_factor, 4),
        "ai_pass_target_fps": str(plan.ai_pass_target_fps),
        "preserve_audio": plan.preserve_audio,
        "notes": list(plan.notes),
    }


def _doctor_to_dict(args: argparse.Namespace, config: AppConfig) -> dict[str, object]:
    rife_dir = Path(_resolve_configured_value(_path_str(args.rife_dir), config.rife_dir) or DEFAULT_VENDOR_RIFE_DIR)
    rife_model_dir = Path(
        _resolve_configured_value(_path_str(args.rife_model_dir), config.rife_model_dir) or DEFAULT_VENDOR_RIFE_MODEL_DIR
    )
    inference_script = rife_dir / "inference_video.py"
    requirements_file = rife_dir / "requirements.txt"
    model_dir_exists = rife_model_dir.exists()
    ffmpeg_path = _find_executable(_resolve_configured_value(_path_str(args.ffmpeg_exe), config.ffmpeg_exe), "ffmpeg")
    ffprobe_path = _find_executable(_resolve_configured_value(_path_str(args.ffprobe_exe), config.ffprobe_exe), "ffprobe")
    rife_python = _find_executable(
        _resolve_configured_value(_path_str(args.rife_python_exe), config.rife_python_exe),
        "python",
    )
    current_python_version = {
        "major": sys.version_info.major,
        "minor": sys.version_info.minor,
        "micro": sys.version_info.micro,
    }
    rife_python_version = _python_version(rife_python) if rife_python else None
    rife_python_runtime_ready = _python_modules_available(
        rife_python,
        ["torch", "cv2", "skvideo", "moviepy", "numpy"],
    ) if rife_python else False
    torch_runtime = _torch_runtime(rife_python) if rife_python else None
    python_compatible_with_official_rife_requirements = bool(
        rife_python_version and rife_python_version["major"] == 3 and rife_python_version["minor"] <= 11
    )
    model_files = []
    warnings = []
    if model_dir_exists:
        model_files = sorted(path.name for path in rife_model_dir.glob("*.pkl"))
    if rife_python is None:
        warnings.append("No RIFE Python executable was configured or found on PATH.")
    elif not python_compatible_with_official_rife_requirements:
        warnings.append(
            "Official RIFE requirements are not expected to install cleanly on Python 3.12+. "
            "This is acceptable if the configured RIFE Python already has the required runtime packages installed."
        )
    if rife_python and not rife_python_runtime_ready:
        warnings.append("Configured RIFE Python is missing one or more runtime packages: torch, cv2, skvideo, moviepy, numpy.")
    if torch_runtime and not torch_runtime.get("cuda_available", False):
        warnings.append("Configured Torch runtime is CPU-only or cannot see a CUDA device.")
    if ffmpeg_path is None or ffprobe_path is None:
        warnings.append("ffmpeg and ffprobe are required but were not found on PATH.")
    if not model_files:
        warnings.append("No RIFE .pkl weight files were found in the configured model directory.")

    return {
        "rife_dir": str(rife_dir.resolve()),
        "rife_exists": rife_dir.exists(),
        "inference_script": str(inference_script.resolve()),
        "inference_script_exists": inference_script.exists(),
        "requirements_file_exists": requirements_file.exists(),
        "current_python_version": current_python_version,
        "rife_python_exe": rife_python,
        "rife_python_version": rife_python_version,
        "rife_python_runtime_ready": rife_python_runtime_ready,
        "torch_runtime": torch_runtime,
        "python_compatible_with_official_rife_requirements": python_compatible_with_official_rife_requirements,
        "ffmpeg_exe": ffmpeg_path,
        "ffprobe_exe": ffprobe_path,
        "model_dir": str(rife_model_dir.resolve()),
        "model_dir_exists": model_dir_exists,
        "model_files": model_files,
        "warnings": warnings,
        "ready_for_rife": (
            inference_script.exists()
            and model_dir_exists
            and len(model_files) > 0
            and ffmpeg_path is not None
            and ffprobe_path is not None
            and rife_python_runtime_ready
            and bool(torch_runtime and torch_runtime.get("cuda_available", False))
        ),
    }


def _backend_options_from_args(args: argparse.Namespace, config: AppConfig, backend: str) -> dict[str, str]:
    ffmpeg_exe = _resolve_required_executable(config.ffmpeg_exe, "ffmpeg", "--ffmpeg-exe")
    ffprobe_exe = _resolve_required_executable(config.ffprobe_exe, "ffprobe", "--ffprobe-exe")
    output_encoding = _resolve_output_encoding(args, config)

    if backend not in {"rife-cli", "rife-double"}:
        return {
            "ffmpeg_exe": ffmpeg_exe,
            "ffprobe_exe": ffprobe_exe,
            **output_encoding,
        }

    rife_dir = Path(_resolve_configured_value(_path_str(args.rife_dir), config.rife_dir) or DEFAULT_VENDOR_RIFE_DIR)
    rife_model_dir = Path(
        _resolve_configured_value(_path_str(args.rife_model_dir), config.rife_model_dir) or DEFAULT_VENDOR_RIFE_MODEL_DIR
    )
    rife_python_exe = _resolve_required_executable(
        config.rife_python_exe or config.python_exe,
        "python",
        "--rife-python-exe",
    )

    if not rife_dir.exists():
        raise ValueError(f"RIFE checkout not found at '{rife_dir}'.")
    if not (rife_dir / "inference_video.py").exists():
        raise ValueError(f"RIFE inference script not found at '{rife_dir / 'inference_video.py'}'.")
    if not rife_model_dir.exists():
        raise ValueError(
            f"RIFE model directory not found at '{rife_model_dir}'. "
            "Download the pretrained weights and place them there."
        )
    if not list(rife_model_dir.glob("*.pkl")):
        raise ValueError(
            f"No RIFE model weights were found in '{rife_model_dir}'. "
            "Expected one or more .pkl files."
        )

    return {
        "python_exe": rife_python_exe,
        "ffmpeg_exe": ffmpeg_exe,
        "ffprobe_exe": ffprobe_exe,
        "vendor_rife_dir": str(rife_dir.resolve()),
        "model_dir": str(rife_model_dir.resolve()),
        "scale": args.scale,
        "fp16_flag": "--fp16" if args.fp16 else "",
        "uhd_flag": "--UHD" if args.uhd else "",
        **output_encoding,
    }


def _resolve_output_encoding(args: argparse.Namespace, config: AppConfig) -> dict[str, str]:
    video_codec = _resolve_configured_value(getattr(args, "video_codec", None), config.video_codec) or DEFAULT_VIDEO_CODEC
    video_crf = _resolve_configured_value(
        str(getattr(args, "video_crf", None)) if getattr(args, "video_crf", None) is not None else None,
        config.video_crf,
    ) or DEFAULT_VIDEO_CRF
    video_preset = _resolve_configured_value(getattr(args, "video_preset", None), config.video_preset)
    video_preset = _normalize_video_preset(video_codec, video_preset)

    if video_codec != "copy":
        crf_value = int(video_crf)
        if not 0 <= crf_value <= 51:
            raise ValueError("CRF must be between 0 and 51.")

    return {
        "video_codec": video_codec,
        "video_crf": video_crf,
        "video_preset": video_preset,
        "resume": "true" if getattr(args, "resume", False) else "false",
        "chunk_count": str(getattr(args, "chunk_count", DEFAULT_RESUME_CHUNK_COUNT)),
        "chunk_duration_seconds": (
            str(getattr(args, "chunk_duration"))
            if getattr(args, "chunk_duration", None) is not None
            else ""
        ),
    }


def _normalize_video_preset(video_codec: str, preset: str | None) -> str:
    if video_codec in {"h264_nvenc", "hevc_nvenc"}:
        return preset if preset in {"p1", "p2", "p3", "p4", "p5", "p6", "p7"} else DEFAULT_VIDEO_PRESET
    return preset if preset is not None else "slow"


def _split_source(args: argparse.Namespace, config: AppConfig) -> None:
    ffmpeg_exe = _resolve_required_executable(config.ffmpeg_exe, "ffmpeg", "--ffmpeg-exe")
    ffprobe_exe = _resolve_required_executable(config.ffprobe_exe, "ffprobe", "--ffprobe-exe")
    source_chunks = prepare_source_chunks(
        source_video=args.input,
        work_dir=args.work_dir,
        ffprobe_exe=ffprobe_exe,
        ffmpeg_exe=ffmpeg_exe,
        chunk_duration_seconds=args.chunk_duration,
        chunk_count=args.chunk_count,
    )
    print(
        json.dumps(
            {
                "work_dir": str(args.work_dir),
                "source_chunks_dir": str((args.work_dir / "source_chunks").resolve()),
                "chunk_count": len(source_chunks),
                "chunks": [str(path.resolve()) for path in source_chunks],
            },
            indent=2,
        )
    )


def _setup_rife(args: argparse.Namespace, config: AppConfig) -> None:
    rife_python = _resolve_required_executable(
        _resolve_configured_value(_path_str(args.rife_python_exe), config.rife_python_exe or config.python_exe),
        "python",
        "--rife-python-exe",
    )
    rife_dir = Path(_resolve_configured_value(_path_str(args.rife_dir), config.rife_dir) or DEFAULT_VENDOR_RIFE_DIR)
    requirements_path = rife_dir / "requirements.txt"
    if not requirements_path.exists():
        raise ValueError(f"RIFE requirements file not found at '{requirements_path}'.")
    rife_python_version = _python_version(rife_python)
    use_modern_python = bool(
        args.modern_python
        or (
            rife_python_version
            and rife_python_version["major"] == 3
            and rife_python_version["minor"] >= 12
        )
    )

    if use_modern_python:
        modern_requirements = [
            "numpy>=1.26,<2",
            "tqdm>=4.66",
            "sk-video>=1.1.10",
            "scipy>=1.11,<1.14",
            "opencv-python==4.10.0.84",
            "moviepy>=1.0.3",
            "torch>=2.4",
            "torchvision>=0.19",
            "imageio-ffmpeg>=0.5",
            "gdown>=5.2",
        ]
        subprocess.run([rife_python, "-m", "pip", "install", *modern_requirements], check=True)
        payload = {
            "rife_python_exe": rife_python,
            "python_version": rife_python_version,
            "requirements_mode": "modern-python",
            "installed": modern_requirements,
        }
    else:
        subprocess.run([rife_python, "-m", "pip", "install", "-r", str(requirements_path), "gdown>=5.2"], check=True)
        payload = {
            "rife_python_exe": rife_python,
            "python_version": rife_python_version,
            "requirements_mode": "official",
            "requirements_installed": str(requirements_path),
        }

    print(json.dumps(payload, indent=2))


def _setup_ffmpeg(args: argparse.Namespace, config: AppConfig) -> None:
    archive_path = args.ffmpeg_zip
    install_dir = args.install_dir
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Invoke-WebRequest -Uri '{url}' -OutFile '{archive_path}'",
        ],
        check=True,
    )

    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        zip_handle.extractall(install_dir)

    ffmpeg_exe, ffprobe_exe = _find_ffmpeg_binaries(install_dir)
    updated = merge_config(
        config,
        {
            "ffmpeg_exe": ffmpeg_exe,
            "ffprobe_exe": ffprobe_exe,
        },
    )
    save_config(updated, args.config)
    print(json.dumps({"ffmpeg_exe": ffmpeg_exe, "ffprobe_exe": ffprobe_exe, "config": str(args.config)}, indent=2))


def _download_rife_model(args: argparse.Namespace, config: AppConfig) -> None:
    rife_python = _resolve_required_executable(
        _resolve_configured_value(_path_str(args.rife_python_exe), config.rife_python_exe or config.python_exe),
        "python",
        "--rife-python-exe",
    )
    model_dir = Path(
        _resolve_configured_value(_path_str(args.rife_model_dir), config.rife_model_dir) or DEFAULT_VENDOR_RIFE_MODEL_DIR
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive_path
    extract_dir = archive_path.parent / "rife-hd-model"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([rife_python, "-m", "pip", "install", "gdown>=5.2"], check=True)
    subprocess.run(
        [
            rife_python,
            "-m",
            "gdown",
            RIFE_HD_MODEL_URL,
            "-O",
            str(archive_path),
        ],
        check=True,
    )

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), str(extract_dir))
    copied_files = []
    for source_path in extract_dir.rglob("*.pkl"):
        destination = model_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_files.append(destination.name)
    for source_path in extract_dir.rglob("*.py"):
        destination = model_dir / source_path.name
        shutil.copy2(source_path, destination)
        copied_files.append(destination.name)
    init_file = model_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")
        copied_files.append(init_file.name)

    updated = merge_config(config, {"rife_model_dir": str(model_dir.resolve())})
    save_config(updated, args.config)
    print(
        json.dumps(
            {
                "rife_model_dir": str(model_dir.resolve()),
                "model_files": sorted(path.name for path in model_dir.glob("*.pkl")),
                "copied_files": sorted(copied_files),
            },
            indent=2,
        )
    )


def _resolve_required_executable(
    configured_path: str | None,
    name: str,
    configure_flag: str | None = None,
) -> str:
    resolved = _find_executable(configured_path, name)
    if not resolved:
        flag_hint = configure_flag or f"--{name}-exe"
        raise ValueError(
            f"Could not resolve '{name}'. Configure it with `framegen configure {flag_hint} ...` "
            f"or make it available on PATH."
        )
    return resolved


def _find_executable(explicit_path: str | None, fallback_name: str) -> str | None:
    if explicit_path:
        candidate = Path(explicit_path)
        return str(candidate.resolve()) if candidate.exists() else None
    return shutil.which(fallback_name)


def _path_str(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def _resolve_configured_value(cli_value: str | None, config_value: str | None) -> str | None:
    return cli_value if cli_value is not None else config_value


def _resolve_media_path(path: Path, default_dir: Path, create_parent: bool) -> Path:
    candidate = path
    if not path.is_absolute() and path.parent == Path("."):
        candidate = default_dir / path
    resolved = candidate.resolve()
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        default_name = f"{input_path.stem}_60fps{input_path.suffix or '.mp4'}"
        return _resolve_media_path(Path(default_name), DEFAULT_OUTPUT_DIR, create_parent=True)
    return _resolve_media_path(output_path, DEFAULT_OUTPUT_DIR, create_parent=True)


def _python_version(executable: str) -> dict[str, int] | None:
    try:
        result = subprocess.run(
            [executable, "-c", "import sys, json; print(json.dumps({'major': sys.version_info[0], 'minor': sys.version_info[1], 'micro': sys.version_info[2]}))"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None


def _python_modules_available(executable: str, module_names: list[str]) -> bool:
    module_list = ",".join(repr(name) for name in module_names)
    code = (
        "import importlib.util, json; "
        f"mods=[{module_list}]; "
        "missing=[name for name in mods if importlib.util.find_spec(name) is None]; "
        "print(json.dumps({'missing': missing}))"
    )
    try:
        result = subprocess.run(
            [executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout.strip())
        return len(payload.get("missing", [])) == 0
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def _torch_runtime(executable: str) -> dict[str, object] | None:
    code = (
        "import json, torch; "
        "payload={"
        "'torch_version': torch.__version__, "
        "'torch_cuda_version': torch.version.cuda, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count(), "
        "'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}; "
        "print(json.dumps(payload))"
    )
    try:
        result = subprocess.run(
            [executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def _find_ffmpeg_binaries(root: Path) -> tuple[str, str]:
    ffmpeg_match = next(root.rglob("ffmpeg.exe"), None)
    ffprobe_match = next(root.rglob("ffprobe.exe"), None)
    if ffmpeg_match is None or ffprobe_match is None:
        raise ValueError(f"Could not find ffmpeg.exe and ffprobe.exe under '{root}'.")
    return str(ffmpeg_match.resolve()), str(ffprobe_match.resolve())


if __name__ == "__main__":
    main()
