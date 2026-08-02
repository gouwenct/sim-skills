#!/usr/bin/env python
"""Thin model-aware wrapper around the legacy SIMcheck_V2 core."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import tifffile

import simcheck_core as _SC

PAGE_AXES = "CTZAP"


@dataclass(frozen=True)
class AcquisitionSpec:
    n_channels: int
    n_timepoints: int
    n_z: int
    n_directions: int
    n_phases: int
    page_order: str

    def sizes(self) -> dict[str, int]:
        return {
            "C": self.n_channels,
            "T": self.n_timepoints,
            "Z": self.n_z,
            "A": self.n_directions,
            "P": self.n_phases,
        }

    def expected_frames(self) -> int:
        return int(np.prod(list(self.sizes().values())))


@dataclass(frozen=True)
class SimModeProfile:
    name: str
    n_directions: int
    n_phases: int
    fpj_peak_orders: int
    mcnr_orders: tuple[int, ...]
    mcnr_noise_method: str
    mcnr_calibrated: bool


SIM_MODE_PROFILES = {
    "2d-sim": SimModeProfile(
        "2d-sim", 3, 3, 1, (1,), "background_modulation_std", False
    ),
    "3d-sim": SimModeProfile(
        "3d-sim", 3, 5, 2, (1, 2), "highest_frequency_spatial_std", True
    ),
    "3d-mp-sim": SimModeProfile(
        "3d-mp-sim", 3, 7, 0, (1, 2, 3), "background_modulation_std", False
    ),
    "4b-sim": SimModeProfile("4b-sim", 4, 13, 0, (), "", False),
}
GENERIC_SIM_MODE = SimModeProfile(
    "generic", 0, 0, 2, (1, 2), "highest_frequency_spatial_std", False
)


def resolve_sim_mode(
    name: str,
    n_directions: int,
    n_phases: int,
) -> SimModeProfile:
    normalized = name.strip().lower()
    if normalized == "generic":
        orders, noise_method, calibrated = _SC.mcnr_policy_for_phases(n_phases)
        return SimModeProfile(
            "generic",
            0,
            0,
            2,
            orders,
            noise_method,
            calibrated and n_phases == 5,
        )
    if normalized == "auto":
        return next(
            (
                profile
                for profile in SIM_MODE_PROFILES.values()
                if (profile.n_directions, profile.n_phases)
                == (n_directions, n_phases)
            ),
            resolve_sim_mode("generic", n_directions, n_phases),
        )
    profile = SIM_MODE_PROFILES[normalized]
    if (n_directions, n_phases) != (
        profile.n_directions,
        profile.n_phases,
    ):
        raise ValueError(
            f"{normalized} requires {profile.n_directions} directions and "
            f"{profile.n_phases} phases"
        )
    return profile


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _normalize_axis_order(spec: AcquisitionSpec) -> str:
    order = "".join(
        axis for axis in spec.page_order.upper() if not axis.isspace()
    )
    unknown = sorted(set(order) - set(PAGE_AXES))
    if unknown or len(set(order)) != len(order):
        raise ValueError(f"invalid page order: {spec.page_order}")
    omitted = [
        axis
        for axis, size in spec.sizes().items()
        if size != 1 and axis not in order
    ]
    if omitted:
        raise ValueError(f"page order omits non-singleton axes: {omitted}")
    return order


def _read_tiff_stack(path: Path) -> np.ndarray:
    stack = np.asarray(tifffile.imread(path), dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"{path}: expected page stack (N,Y,X), got {stack.shape}")
    return stack


def reshape_pages_to_ctzapyx(
    stack: np.ndarray,
    spec: AcquisitionSpec,
) -> np.ndarray:
    order = _normalize_axis_order(spec)
    if stack.shape[0] != spec.expected_frames():
        raise ValueError(
            f"frames={stack.shape[0]}, expected={spec.expected_frames()} "
            f"for {asdict(spec)}"
        )
    sizes = spec.sizes()
    present_shape = [sizes[axis] for axis in order]
    arranged = stack.reshape(*present_shape, *stack.shape[-2:])
    present = list(order)
    for axis in PAGE_AXES:
        if axis not in present:
            arranged = np.expand_dims(arranged, axis=len(present))
            present.append(axis)
    permutation = [present.index(axis) for axis in PAGE_AXES] + [
        len(present),
        len(present) + 1,
    ]
    return np.ascontiguousarray(np.transpose(arranged, permutation))


def discover_inputs(
    input_path: Path,
    *,
    name_glob: str,
    recursive: bool,
    excluded_root: Path | None = None,
) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"input is not TIFF: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    iterator: Iterable[Path] = (
        input_path.rglob(name_glob)
        if recursive
        else input_path.glob(name_glob)
    )
    excluded = excluded_root.resolve() if excluded_root else None
    paths = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        resolved = path.resolve()
        if excluded and (resolved == excluded or excluded in resolved.parents):
            continue
        paths.append(resolved)
    paths.sort(key=lambda path: str(path).lower())
    if not paths:
        raise FileNotFoundError(f"no TIFF matched {name_glob!r} under {input_path}")
    return paths


def _item_output_dir(
    source: Path,
    input_path: Path,
    output_root: Path,
) -> Path:
    if input_path.is_file():
        return output_root
    relative = source.relative_to(input_path)
    return output_root / relative.parent / source.stem


def _preflight_output_dirs(
    sources: Sequence[Path],
    input_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    item_dirs = [
        _item_output_dir(source, input_path, output_root) for source in sources
    ]
    duplicates = {
        path for path, count in Counter(item_dirs).items() if count > 1
    }
    if duplicates:
        raise ValueError(f"multiple inputs map to output directories: {duplicates}")
    if overwrite:
        return
    conflicts = [
        path for path in item_dirs if path.exists() and any(path.iterdir())
    ]
    if conflicts:
        raise FileExistsError(
            "non-empty SIMcheck_V2 output exists; use --overwrite: "
            + ", ".join(str(path) for path in conflicts)
        )


def _timepoint_label(path: Path) -> str:
    match = re.search(r"_(\d{3,5})$", path.stem)
    return f"t{int(match.group(1)):04d}" if match else "t0001"


def _parse_checks(value: str) -> set[str]:
    checks = {item.strip().lower() for item in value.split(",") if item.strip()}
    unsupported = checks - {"mcnr", "fpj"}
    if unsupported:
        raise ValueError(f"unsupported checks: {sorted(unsupported)}")
    return checks


def _run_one(
    source: Path,
    item_dir: Path,
    spec: AcquisitionSpec,
    profile: SimModeProfile,
    checks: set[str],
    mcnr_zw: int,
    preview: bool,
) -> tuple[_SC.SampleMcnrSummary, list[_SC.CheckReport]]:
    started = time.perf_counter()
    canonical = reshape_pages_to_ctzapyx(_read_tiff_stack(source), spec)
    raw = np.ascontiguousarray(np.transpose(canonical, (0, 4, 2, 3, 1, 5, 6)))
    raw_dir = item_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports: list[_SC.CheckReport] = []
    if "mcnr" in checks and profile.mcnr_orders:
        reports.append(
            _SC.check_mcnr(
                raw,
                raw_dir,
                harmonic_orders=profile.mcnr_orders,
                noise_method=profile.mcnr_noise_method,
                calibrated=profile.mcnr_calibrated,
                zw=mcnr_zw,
            )
        )
    if "fpj" in checks:
        reports.append(
            _SC.check_fpj(
                raw,
                raw_dir,
                peak_orders=profile.fpj_peak_orders,
            )
        )
    meta = {
        "sim_mode": profile.name,
        "order": spec.page_order,
        "n_phases": spec.n_phases,
        "n_angles": spec.n_directions,
        "n_z": spec.n_z,
        "n_c": spec.n_channels,
        "n_t": spec.n_timepoints,
        "device": _SC.get_device_name(),
        "checks": sorted(checks),
        "mcnr_orders": list(profile.mcnr_orders),
        "mcnr_noise_method": profile.mcnr_noise_method,
        "mcnr_calibrated": profile.mcnr_calibrated,
    }
    _SC.write_markdown(
        item_dir / "SIMcheck_V2_report.md",
        source,
        "raw",
        reports,
        meta,
    )
    _SC.write_csv(item_dir / "SIMcheck_V2_stats.csv", reports)
    if preview and "fpj" in checks:
        from fpj_preview import preview_fpj

        for channel in range(1, spec.n_channels + 1):
            fpj = raw_dir / f"FPJ_C{channel}.tif"
            preview_fpj(
                fpj,
                source.stem,
                raw_dir / f"FPJ_C{channel}-preview.png",
            )
    metrics = _SC.extract_mcnr_from_reports(reports)
    summary = _SC.sample_summary_from_mcnr_dict(
        source.stem,
        metrics,
        track=profile.name,
        dr_sub=source.parent.name,
        source=str(source),
        out_dir=str(item_dir),
        elapsed_s=time.perf_counter() - started,
    )
    return summary, reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-aware wrapper preserving the SIMcheck_V2 output contract"
    )
    parser.add_argument("input", type=Path, help="phase-resolved TIFF or folder")
    parser.add_argument("--sim-mode", choices=["auto", "generic", *SIM_MODE_PROFILES], default="auto")
    parser.add_argument("--n-z", type=_positive_int, required=True)
    parser.add_argument("--n-directions", type=_positive_int, required=True)
    parser.add_argument("--n-phases", type=_positive_int, required=True)
    parser.add_argument("--n-channels", type=_positive_int, default=1)
    parser.add_argument("--n-timepoints", type=_positive_int, default=1)
    parser.add_argument("--page-order", required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--name-glob", default="*.tif*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--checks", default="mcnr,fpj")
    parser.add_argument("--mcnr-z-half-window", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> list[Path]:
    if args.n_timepoints != 1:
        raise ValueError(
            "SIMcheck_V2 compatibility layout requires one timepoint per TIFF"
        )
    if args.mcnr_z_half_window < 0:
        raise ValueError("--mcnr-z-half-window must be >= 0")
    input_path = args.input.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else (
            input_path.parent / "SIMcheck_V2"
            if input_path.is_file()
            else input_path / "SIMcheck_V2"
        )
    )
    sources = discover_inputs(
        input_path,
        name_glob=args.name_glob,
        recursive=args.recursive,
        excluded_root=output_root,
    )
    profile = resolve_sim_mode(
        args.sim_mode,
        args.n_directions,
        args.n_phases,
    )
    checks = _parse_checks(args.checks)
    if "mcnr" in checks and not profile.mcnr_orders:
        checks.remove("mcnr")
    spec = AcquisitionSpec(
        args.n_channels,
        args.n_timepoints,
        args.n_z,
        args.n_directions,
        args.n_phases,
        args.page_order.upper(),
    )
    _preflight_output_dirs(
        sources,
        input_path,
        output_root,
        overwrite=args.overwrite,
    )
    plan = {
        "core": str(Path(_SC.__file__).resolve()),
        "output_contract": "SIMcheck_V2",
        "input": str(input_path),
        "output_root": str(output_root),
        "sources": [str(source) for source in sources],
        "acquisition": asdict(spec),
        "sim_mode": profile.name,
        "checks": sorted(checks),
        "fpj_peak_orders": profile.fpj_peak_orders,
        "mcnr_orders": list(profile.mcnr_orders),
        "mcnr_calibrated": profile.mcnr_calibrated,
        "device": args.device,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return []

    device = _SC._init_device(prefer_gpu=args.device != "cpu")
    if args.device == "cuda" and not device.startswith("cuda"):
        raise RuntimeError(f"CUDA requested but unavailable; backend={device}")
    rows = []
    item_dirs = []
    for source in sources:
        item_dir = _item_output_dir(source, input_path, output_root)
        item_dir.mkdir(parents=True, exist_ok=True)
        summary, _ = _run_one(
            source,
            item_dir,
            spec,
            profile,
            checks,
            args.mcnr_z_half_window,
            not args.no_preview,
        )
        rows.append(summary)
        item_dirs.append(item_dir)
        print(f"[done] {source.name} -> {item_dir}")

    targets = [output_root, *item_dirs]
    _SC.write_batch_mcnr_documents(
        rows,
        targets,
        {
            "batch": input_path.name,
            "dr_sub": input_path.name if input_path.is_dir() else input_path.parent.name,
            "timepoint": ",".join(_timepoint_label(source) for source in sources),
            "device": device,
            "checks": ",".join(sorted(checks)),
            "order": spec.page_order,
            "n_angles": spec.n_directions,
            "n_phases": spec.n_phases,
            "n_z": spec.n_z,
            "mcnr_calibrated": profile.mcnr_calibrated,
            "note": (
                "Extended MCNR is uncalibrated; five-phase thresholds are not applied."
                if not profile.mcnr_calibrated and "mcnr" in checks
                else ""
            ),
        },
    )
    return item_dirs


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
