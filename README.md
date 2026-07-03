# FrameGen

FrameGen is a movie-focused frame interpolation orchestrator. The goal is to turn film and video sources such as 24fps, 23.976fps, 30fps, or 29.97fps into smoother 60fps outputs while keeping the project flexible enough to swap interpolation models over time.

This repository is intentionally starting as an orchestration layer instead of a from-scratch neural model. That gives us a usable path sooner:

- Probe source video with `ffprobe`
- Build a job plan for 24/30fps to 60fps conversion
- Call a pluggable interpolation backend such as RIFE or FILM
- Remux original audio back onto the interpolated output

## Why this architecture

For a practical MVP, wrapping an existing interpolation model is the right tradeoff.

- RIFE explicitly supports arbitrary-timestep interpolation and its repository documents video inference with a target FPS path. Source: [RIFE paper](https://arxiv.org/abs/2011.06294), [RIFE repo](https://github.com/hzwer/ECCV2022-RIFE)
- FILM is another strong option for large motion scenes and publishes an official TensorFlow implementation. Source: [FILM paper](https://arxiv.org/abs/2202.04901), [FILM repo](https://github.com/google-research/frame-interpolation)
- FFmpeg's `minterpolate` is useful as a fallback and for debugging pipeline behavior even though it is not an AI model.

## Current status

This repo contains the project skeleton and a working CLI orchestration layer. It now includes the official RIFE repository under [vendor/ECCV2022-RIFE](C:/Users/wzeng/Documents/Projects/FrameGen/vendor/ECCV2022-RIFE), but it does **not** bundle large model checkpoints yet.

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

You will also need:

- `ffmpeg`
- `ffprobe`
- One interpolation backend

## Commands

Save local executable paths once:

```powershell
framegen configure `
  --ffmpeg-exe "C:\ffmpeg\bin\ffmpeg.exe" `
  --ffprobe-exe "C:\ffmpeg\bin\ffprobe.exe" `
  --rife-python-exe "C:\Python311\python.exe" `
  --rife-dir "C:\Users\wzeng\Documents\Projects\FrameGen\vendor\ECCV2022-RIFE" `
  --rife-model-dir "C:\Users\wzeng\Documents\Projects\FrameGen\vendor\ECCV2022-RIFE\train_log"
```

Inspect a source file and print the plan:

```powershell
framegen plan --input movie.mp4 --output movie_60fps.mp4 --backend rife-cli
```

Run an interpolation job:

```powershell
framegen run --input movie.mp4 --output movie_60fps.mp4 --backend rife-cli
```

Check whether the local RIFE checkout and model weights are ready:

```powershell
framegen doctor
```

Install RIFE dependencies into the configured backend Python:

```powershell
framegen setup-rife
```

This shell still does not inherit your normal PATH, so explicit paths in `framegen configure` are the most dependable way to make the tool runnable here.

Fallback without a neural model:

```powershell
framegen run --input movie.mp4 --output movie_60fps.mp4 --backend ffmpeg-minterpolate
```

## RIFE setup

FrameGen now defaults `rife-cli` and `rife-double` to the vendored RIFE checkout. To make them actually runnable, you still need:

- A Python 3.10 or 3.11 environment for the RIFE backend
- Python dependencies required by RIFE
- Pretrained RIFE `.pkl` weights inside `vendor\ECCV2022-RIFE\train_log\`
- `ffmpeg` and `ffprobe` on PATH

You can also point to another checkout or model directory:

```powershell
framegen run `
  --input movie.mp4 `
  --output movie_60fps.mp4 `
  --backend rife-cli `
  --config .framegen.json `
  --rife-dir C:\models\ECCV2022-RIFE `
  --rife-model-dir C:\models\ECCV2022-RIFE\train_log `
  --scale 0.5 `
  --uhd
```

## Backend templates

FrameGen assumes the backend writes its silent video result to a temporary file chosen by the orchestrator. You can override the command template with environment variables.

### `rife-cli`

Default template used by FrameGen:

```text
"{python}" "{vendor_rife_dir}\inference_video.py" --video "{input}" --output "{output}" --fps {target_fps} --model "{model_dir}" --scale {scale} {fp16_flag} {uhd_flag}
```

Override:

```powershell
$env:FRAMEGEN_RIFE_CMD = '"{python}" "C:\models\ECCV2022-RIFE\inference_video.py" --video "{input}" --output "{output}" --fps {target_fps} --model "{model_dir}" --scale {scale}'
```

### `rife-double`

Default template:

```text
"{python}" "{vendor_rife_dir}\inference_video.py" --video "{input}" --output "{output}" --exp {exp} --model "{model_dir}" --scale {scale} {fp16_flag} {uhd_flag}
```

This mode doubles repeatedly until it reaches or exceeds the target FPS.

### `film-cli`

This is a placeholder contract for a FILM video wrapper:

```text
python -m eval.interpolator_cli --input_video {input} --output_video {output} --target_fps {target_fps}
```

The official FILM repository focuses on frame/image interpolation, so a video-oriented wrapper is a reasonable next layer for us to build.

## Recommended roadmap

1. Add a one-command bootstrap for RIFE dependencies and checkpoint download.
2. Add scene-cut detection so we do not interpolate through hard cuts.
3. Add tile-based processing for 4K sources and limited VRAM GPUs.
4. Add quality presets for `film`, `animation`, and `sports`.
5. Add a small desktop or web UI once the CLI behavior is stable.

## Notes on 24fps to 60fps

24 to 60 is a 2.5x conversion, so not every output frame maps cleanly to a single midpoint insertion. That is why arbitrary-timestep interpolation support matters. A backend limited to repeated doubling can still work, but it will usually generate a higher intermediate rate first and then retime back to 60fps.

## Licensing and model distribution

Keep model weights and large checkpoints outside the repo unless we make an explicit packaging decision later.
