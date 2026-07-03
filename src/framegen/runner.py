from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .backends import build_backend_command, env_backend_template
from .ffmpeg_tools import remux_audio
from .types import InterpolationPlan


class PipelineError(RuntimeError):
    """Raised when the interpolation pipeline fails."""


def run_plan(plan: InterpolationPlan, work_dir: Path) -> Path:
    return run_plan_with_options(plan=plan, work_dir=work_dir, backend_options={})


def run_plan_with_options(
    plan: InterpolationPlan,
    work_dir: Path,
    backend_options: dict[str, str],
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

    if plan.preserve_audio:
        remux_audio(
            plan.input_path,
            temp_output,
            plan.output_path,
            ffmpeg_path=backend_options.get("ffmpeg_exe"),
        )
    else:
        temp_output.replace(plan.output_path)

    shutil.rmtree(work_dir, ignore_errors=True)
    return plan.output_path


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
