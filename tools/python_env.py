"""Resolve a compatible Python without modifying any environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROFILES = {
    "core": ("numpy", "tifffile", "matplotlib", "PIL"),
    "fwhm": ("numpy", "scipy", "tifffile", "matplotlib", "skimage", "PIL"),
    "decorr": ("numpy", "scipy", "pandas", "tifffile", "matplotlib", "openpyxl"),
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_in(prefix: Path) -> Path:
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def _is_3dmpsim(prefix: Path) -> bool:
    return prefix.name.lower().startswith("3dmpsim")


def _candidate_prefixes() -> list[Path]:
    prefixes: list[Path] = []
    active = os.environ.get("CONDA_PREFIX")
    if active and _is_3dmpsim(Path(active)):
        prefixes.append(Path(active))

    conda = shutil.which("conda")
    if conda:
        result = subprocess.run(
            [conda, "env", "list", "--json"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            prefixes.extend(Path(item) for item in json.loads(result.stdout).get("envs", []))

    roots = [Path.home()]
    if os.name == "nt":
        roots.extend(Path(f"{letter}:\\") for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ")
    for root in roots:
        for distribution in ("anaconda3", "miniconda3", "miniforge3"):
            envs = root / distribution / "envs"
            if envs.is_dir():
                prefixes.extend(envs.glob("3DMPSIM*"))

    unique: dict[str, Path] = {}
    for prefix in prefixes:
        if _is_3dmpsim(prefix):
            unique[str(prefix.resolve()).lower()] = prefix.resolve()
    return list(unique.values())


def _missing(python: Path, modules: tuple[str, ...]) -> list[str]:
    code = (
        "import importlib.util,json;"
        f"mods={list(modules)!r};"
        "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
    )
    result = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return list(modules)
    return json.loads(result.stdout)


def resolve_python(profile: str) -> Path:
    modules = PROFILES[profile]
    explicit = os.environ.get("SIM_SKILLS_PYTHON")
    if explicit:
        python = Path(explicit).expanduser().resolve()
        missing = _missing(python, modules) if python.is_file() else list(modules)
        if missing:
            raise RuntimeError(f"SIM_SKILLS_PYTHON is missing: {', '.join(missing)}")
        return python

    valid: list[Path] = []
    current_prefix = Path(sys.prefix)
    prefixes = ([current_prefix] if _is_3dmpsim(current_prefix) else []) + _candidate_prefixes()
    for prefix in prefixes:
        python = _python_in(prefix)
        if python.is_file() and not _missing(python, modules) and python not in valid:
            valid.append(python)

    if len(valid) == 1:
        return valid[0]
    if len(valid) > 1:
        choices = "\n".join(f"- {path}" for path in valid)
        raise RuntimeError(
            "Multiple compatible 3DMPSIM environments were found. "
            f"Ask the user to choose one and set SIM_SKILLS_PYTHON:\n{choices}"
        )

    shared = _python_in(REPO_ROOT / ".venv")
    if shared.is_file() and not _missing(shared, modules):
        return shared
    raise RuntimeError(
        "No compatible 3DMPSIM* or shared UV environment was found. "
        "Ask the user whether to create the CPU-only UV environment; do not create it automatically."
    )


def _self_check() -> None:
    assert _is_3dmpsim(Path("3DMPSIM310"))
    assert not _is_3dmpsim(Path("base"))
    assert "fwhm" in PROFILES and "decorr" in PROFILES


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        _self_check()
        print("python_env self-check passed")
    else:
        profile = sys.argv[1] if len(sys.argv) > 1 else "core"
        try:
            print(resolve_python(profile))
        except RuntimeError as error:
            print(error, file=sys.stderr)
            raise SystemExit(2) from None
