"""Dispatch the decorrelation skill's measurement, batch, and render commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


COMMANDS = {
    "measure": "decorr_pipeline.py",
    "batch": "run_formal_decorr_batch.py",
    "render": "render_decorr_results.py",
    "peak-qc": "analyze_axial_peak_qc.py",
}
SKILL_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = SKILL_ROOT.parent.parent / "tools"
sys.path.insert(0, str(TOOLS_ROOT))
from python_env import resolve_python


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        choices = ", ".join(COMMANDS)
        print(f"Usage: run_decorr.py <{choices}> [arguments]", file=sys.stderr)
        return 2
    script = Path(__file__).resolve().parent / COMMANDS[sys.argv[1]]
    python = resolve_python("decorr")
    command = [str(python), "-B", str(script), *sys.argv[2:]]
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
