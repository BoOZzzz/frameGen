from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from .types import BackendCommand, InterpolationPlan


class BackendError(RuntimeError):
    """Raised when backend configuration is invalid."""


def _resolve_template(template: str, variables: dict[str, str]) -> tuple[str, ...]:
    try:
        rendered = template.format(**variables)
    except KeyError as exc:
        raise BackendError(f"Missing template variable: {exc.args[0]}") from exc
    return tuple(token[1:-1] if len(token) >= 2 and token[0] == token[-1] == '"' else token for token in shlex.split(rendered, posix=False))


def build_backend_command(
    plan: InterpolationPlan,
    work_dir: Path,
    backend_command_template: str | None = None,
    backend_options: dict[str, str] | None = None,
) -> BackendCommand:
    temp_output = work_dir / "interpolated_noaudio.mp4"
    options = backend_options or {}
    ffmpeg_dir = ""
    if options.get("ffmpeg_exe"):
        ffmpeg_dir = str(Path(options["ffmpeg_exe"]).resolve().parent)
    variables = {
        "input": str(plan.input_path),
        "output": str(temp_output),
        "target_fps": _format_fps_argument(plan.ai_pass_target_fps),
        "exp": _estimate_exp(plan.source_fps, plan.ai_pass_target_fps),
        "work_dir": str(work_dir),
        "python": options.get("python_exe", sys.executable),
        "ffmpeg": options.get("ffmpeg_exe", "ffmpeg"),
        "vendor_rife_dir": options.get("vendor_rife_dir", ""),
        "model_dir": options.get("model_dir", ""),
        "scale": options.get("scale", "1.0"),
        "fp16_flag": options.get("fp16_flag", ""),
        "uhd_flag": options.get("uhd_flag", ""),
    }

    if plan.backend == "rife-cli":
        template = backend_command_template or (
            "\"{python}\" \"{vendor_rife_dir}\\inference_video.py\" --video \"{input}\" --output \"{output}\" "
            "--fps {target_fps} --model \"{model_dir}\" --scale {scale} {fp16_flag} {uhd_flag}"
        )
        description = "Run an existing RIFE Python checkout to generate interpolated frames."
        cwd = Path(options["vendor_rife_dir"]) if options.get("vendor_rife_dir") else None
    elif plan.backend == "rife-double":
        template = backend_command_template or (
            "\"{python}\" \"{vendor_rife_dir}\\inference_video.py\" --video \"{input}\" --output \"{output}\" "
            "--exp {exp} --model \"{model_dir}\" --scale {scale} {fp16_flag} {uhd_flag}"
        )
        description = "Run RIFE in exponential doubling mode, then retime if needed."
        cwd = Path(options["vendor_rife_dir"]) if options.get("vendor_rife_dir") else None
    elif plan.backend == "film-cli":
        template = backend_command_template or (
            "\"{python}\" -m eval.interpolator_cli --input_video \"{input}\" --output_video \"{output}\" --target_fps {target_fps}"
        )
        description = "Run a FILM-style CLI wrapper."
        cwd = None
    elif plan.backend == "ffmpeg-minterpolate":
        template = backend_command_template or (
            "\"{ffmpeg}\" -y -i \"{input}\" -vf minterpolate=fps={target_fps}:mi_mode=mci -an \"{output}\""
        )
        description = "Run FFmpeg's classical minterpolate fallback."
        cwd = None
    else:
        raise BackendError(f"Unsupported backend '{plan.backend}'.")

    env = None
    if plan.backend in {"rife-cli", "rife-double"} and ffmpeg_dir:
        path_value = os.environ.get("PATH", "")
        env = dict(os.environ)
        env["PATH"] = ffmpeg_dir + os.pathsep + path_value if path_value else ffmpeg_dir
        env["IMAGEIO_FFMPEG_EXE"] = options["ffmpeg_exe"]

    return BackendCommand(
        argv=_resolve_template(template, variables),
        description=description,
        cwd=cwd,
        env=env,
    )


def _estimate_exp(source_fps, target_fps) -> str:
    exp = 0
    current = source_fps
    while current < target_fps:
        current *= 2
        exp += 1
    return str(exp)


def _format_fps_argument(target_fps) -> str:
    if getattr(target_fps, "denominator", None) == 1:
        return str(target_fps.numerator)
    return str(round(float(target_fps)))


def env_backend_template(backend: str) -> str | None:
    mapping = {
        "rife-cli": os.getenv("FRAMEGEN_RIFE_CMD"),
        "rife-double": os.getenv("FRAMEGEN_RIFE_DOUBLE_CMD"),
        "film-cli": os.getenv("FRAMEGEN_FILM_CMD"),
        "ffmpeg-minterpolate": os.getenv("FRAMEGEN_FFMPEG_CMD"),
    }
    return mapping.get(backend)
