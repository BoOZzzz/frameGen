from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    child_code = (
        "import runpy, sys; "
        f"project_root = r'{project_root}'; "
        f"src_dir = r'{src_dir}'; "
        "sys.path = [src_dir] + [p for p in sys.path if p not in ('', project_root)]; "
        "sys.argv = ['framegen', *sys.argv[1:]]; "
        "runpy.run_module('framegen.cli', run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", child_code, *sys.argv[1:]],
        check=False,
        cwd=project_root,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
