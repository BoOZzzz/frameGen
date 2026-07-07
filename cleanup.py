from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _remove_path(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    else:
        path.unlink()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove temporary FrameGen artifacts produced by resumable or interrupted runs."
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("tmp") / "framegen",
        help="FrameGen work directory to clean. Defaults to tmp/framegen.",
    )
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    removed_any = False

    if work_dir.exists():
        print(f"[cleanup] Removing work directory: {work_dir}")
        _remove_path(work_dir)
        removed_any = True

    root_tmp = (Path.cwd() / "tmp").resolve()
    if work_dir.parent == root_tmp:
        for pattern in ("chunk_work_*",):
            for path in root_tmp.glob(pattern):
                print(f"[cleanup] Removing leftover chunk work dir: {path}")
                _remove_path(path)
                removed_any = True

    if not removed_any:
        print(f"[cleanup] Nothing to remove under {work_dir}")
    else:
        print("[cleanup] Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
