from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parent
    launcher = project_root / "framegen.py"
    result = subprocess.run(
        [sys.executable, str(launcher), "split", *sys.argv[1:]],
        check=False,
        cwd=project_root,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
