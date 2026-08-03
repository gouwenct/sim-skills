from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


SKILL_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = Path(__file__).with_name("fwhm_cli.py")
TOOLS_ROOT = SKILL_ROOT.parent.parent / "tools"
sys.path.insert(0, str(TOOLS_ROOT))
from python_env import resolve_python


def main() -> int:
    arguments = sys.argv[1:]
    python = resolve_python("fwhm")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    return subprocess.call(
        [str(python), "-B", str(CLI_PATH), *arguments],
        env=environment,
    )


if __name__ == "__main__":
    raise SystemExit(main())
