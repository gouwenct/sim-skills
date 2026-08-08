#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute preliminary FPJ directly from cam1/cam2 camera raw data.

The raw camera width is split into four 512-pixel regions by the bundled
MP preprocessing workflow. Region 4 is cam1's first horizontal segment;
region 5 is cam2's first segment after the existing horizontal flip.

For a 3x5 acquisition, pages are interpreted as ``T-Z-A-phase`` with phase
fastest.  The number of timepoints is inferred from the page count.

Only up to 10 timepoints are processed. By default the first leading
timepoints are used (a contiguous page range keeps IO fast for very large
acquisitions); pass ``--sample-evenly`` to sample timepoints evenly across the
full range instead.  The TIFF header is read first to learn the page count, and
only the processed timepoints' page ranges are loaded, so very large
acquisitions do not need to be read entirely into memory.

If the page count is not a multiple of ``n_z * n_directions * n_phases`` (for
example after a >4 GB file was split), trailing incomplete frames are dropped
with a warning.  When a data set was split across multiple camera files,
``--input-dir`` auto-discovery picks the first file by name.

Output layout (four sibling folders; no reports or CSV are written):

    raw_fpj/
      region4_tif/FPJ_region4_T042.tif
      region5_tif/FPJ_region5_T042.tif
      region4_png/FPJ_region4_T042.png
      region5_png/FPJ_region5_T042.png

The ``T042`` suffix is the original (1-based) timepoint index in the file.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


DEFAULT_SIMCHECK_DIR = Path(
    os.environ.get("SIMCHECK_SKILL_DIR")
    or Path(__file__).resolve().parents[2] / "simcheck"
)


def _load_simcheck(simcheck_dir: Path):
    scripts_dir = simcheck_dir
    if not (scripts_dir / "simcheck_core.py").is_file():
        scripts_dir = simcheck_dir / "scripts"
    if not (scripts_dir / "simcheck_core.py").is_file():
        raise FileNotFoundError(
            f"simcheck_core.py not found under {simcheck_dir}; "
            "use --simcheck-dir or SIMCHECK_SKILL_DIR"
        )
    sys.path.insert(0, str(scripts_dir))
    import simcheck_core  # type: ignore

    try:
        from fpj_preview import preview_fpj  # type: ignore
    except Exception:
        preview_fpj = None
    return simcheck_core, preview_fpj


def _positive_int(value: str) -> int:
    try:
        ivalue = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {ivalue}")
    return ivalue


def _resolve_cam2(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"cam2 path does not exist: {path}")
    candidates = sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if len(candidates) != 1:
        names = ", ".join(p.name for p in candidates) or "none"
        raise ValueError(f"cam2 directory must contain exactly one TIFF; found: {names}")
    return candidates[0]


def _tiff_files(directory: Path) -> list[Path]:
    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )


def _discover_first(directory: Path, prefix: str, what: str) -> Path:
    candidates = sorted(
        p for p in _tiff_files(directory) if p.name.lower().startswith(prefix.lower())
    )
    if not candidates:
        raise ValueError(f"{what}: no {prefix}* TIFF found under {directory}")
    if len(candidates) > 1:
        print(
            f"{what}: {len(candidates)} TIFFs found; using first by name: "
            f"{candidates[0].name}"
        )
    return candidates[0]


def _discover_cam1(input_dir: Path) -> Path:
    return _discover_first(input_dir, "cam1", "cam1")


def _discover_cam2(input_dir: Path) -> Path:
    candidates = sorted(
        p for p in _tiff_files(input_dir) if p.name.lower().startswith("cam2")
    )
    if not candidates:
        cam2_dir = input_dir / "cam2"
        if cam2_dir.is_dir():
            candidates = sorted(
                p
                for p in _tiff_files(cam2_dir)
                if p.name.lower().startswith("cam2")
            )
    if not candidates:
        raise ValueError(
            f"cam2: no cam2* TIFF found under {input_dir} or its cam2/ subdirectory"
        )
    if len(candidates) > 1:
        print(
            f"cam2: {len(candidates)} TIFFs found; using first by name: "
            f"{candidates[0].name}"
        )
    return candidates[0]


def _tiff_shape(path: Path) -> tuple[int, int, int]:
    """Read only the TIFF header/IFD chain; returns (pages, Y, X)."""
    with tifffile.TiffFile(str(path)) as tf:
        shape = tuple(tf.series[0].shape)
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) != 3:
        raise ValueError(f"expected pages,Y,X TIFF for {path}, got {shape}")
    return shape


def _read_pages(path: Path, start: int, stop: int) -> np.ndarray:
    """Read pages [start, stop) into float32 without loading the whole file."""
    data = np.asarray(
        tifffile.imread(str(path), key=slice(start, stop)), dtype=np.float32
    )
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3:
        raise ValueError(f"expected pages,Y,X TIFF for {path}, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"input contains non-finite pixels: {path}")
    return data


def _sample_timepoints(n_timepoints: int, max_samples: int = 10) -> list[int]:
    """Pick the first timepoints (0-based) without loading the whole file.

    Returns ``list(range(min(n_timepoints, max_samples)))``; when there are
    fewer timepoints than the cap, all are kept. The contiguous leading page
    range keeps IO fast for very large acquisitions.
    """
    return list(range(min(n_timepoints, max_samples)))


def _sample_evenly(n_timepoints: int, max_samples: int = 10) -> list[int]:
    """Evenly sample up to max_samples timepoints (0-based), endpoints included.

    Only used when ``--sample-evenly`` is requested. N <= max_samples keeps all
    timepoints; otherwise np.linspace covers the full range.
    """
    if n_timepoints <= max_samples:
        return list(range(n_timepoints))
    picks = np.unique(
        np.round(np.linspace(0, n_timepoints - 1, max_samples)).astype(int)
    )
    return [int(i) for i in picks]


def _extract_region(stack: np.ndarray, *, flip_x: bool) -> np.ndarray:
    pages, height, width = stack.shape
    if width % 4:
        raise ValueError(
            f"raw width must be divisible by 4, got shape {stack.shape} (width={width})"
        )
    region = stack[:, :, : width // 4]
    if flip_x:
        region = region[:, :, ::-1]
    return np.ascontiguousarray(region)


def _to_raw_tensor(
    region4: np.ndarray,
    region5: np.ndarray,
    *,
    n_z: int,
    n_directions: int,
    n_phases: int,
) -> np.ndarray:
    """Return the SIMcheck core layout ``(C,phase,Z,A,T,Y,X)``.

    ``region4``/``region5`` must already contain exactly one timepoint's worth
    of pages (``n_z * n_directions * n_phases``) in ``T-Z-A-phase`` order.
    """
    frames_per_timepoint = n_z * n_directions * n_phases
    blocks = []
    for region in (region4, region5):
        block = region
        if block.shape[0] != frames_per_timepoint:
            raise ValueError(
                f"timepoint block has {block.shape[0]} pages; "
                f"expected {frames_per_timepoint}"
            )
        height, width = block.shape[1:]
        # Stored order is T-Z-A-phase, with phase fastest.
        block = block.reshape(n_z, n_directions, n_phases, height, width)
        block = np.transpose(block, (2, 0, 1, 3, 4))
        blocks.append(block)
    raw = np.stack(blocks, axis=0)[..., None, :, :]
    expected = (2, n_phases, n_z, n_directions, 1, region4.shape[1], region4.shape[2])
    assert raw.shape == expected, (raw.shape, expected)
    return raw


def _move_output(source: Path, target: Path, *, overwrite: bool) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace: {target}")
        target.unlink()
    source.replace(target)


def _precheck_targets(output_root: Path, timepoint: int, *, preview_fpj: Any) -> None:
    """Fail before any write if any destination already exists (unless --overwrite)."""
    targets = [
        output_root / "region4_tif" / f"FPJ_region4_T{timepoint:03d}.tif",
        output_root / "region5_tif" / f"FPJ_region5_T{timepoint:03d}.tif",
    ]
    if preview_fpj is not None:
        targets += [
            output_root / "region4_png" / f"FPJ_region4_T{timepoint:03d}.png",
            output_root / "region5_png" / f"FPJ_region5_T{timepoint:03d}.png",
        ]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise FileExistsError(
            "output exists; pass --overwrite to replace: " + ", ".join(existing)
        )


def _write_timepoint(
    *,
    simcheck: Any,
    raw: np.ndarray,
    output_root: Path,
    timepoint: int,
    peak_orders: int,
    overwrite: bool,
    preview_fpj: Any,
) -> list[Path]:
    if not overwrite:
        _precheck_targets(output_root, timepoint, preview_fpj=preview_fpj)
    written: list[Path] = []
    with tempfile.TemporaryDirectory(prefix=f".T{timepoint:03d}_", dir=str(output_root)) as temp:
        temp_dir = Path(temp)
        simcheck.check_fpj(raw, temp_dir, peak_orders=peak_orders)
        for channel_index, region_name in enumerate(("region4", "region5"), start=1):
            source_fpj = temp_dir / f"FPJ_C{channel_index}.tif"
            tif_dir = output_root / f"{region_name}_tif"
            png_dir = output_root / f"{region_name}_png"
            tif_dir.mkdir(parents=True, exist_ok=True)
            png_dir.mkdir(parents=True, exist_ok=True)
            target_fpj = tif_dir / f"FPJ_{region_name}_T{timepoint:03d}.tif"
            _move_output(source_fpj, target_fpj, overwrite=overwrite)
            written.append(target_fpj)
            if preview_fpj is not None:
                preview_path = png_dir / f"FPJ_{region_name}_T{timepoint:03d}.png"
                preview_fpj(target_fpj, f"T{timepoint:03d} {region_name}", preview_path)
                written.append(preview_path)
    return written


def run(args: argparse.Namespace) -> int:
    simcheck, preview_fpj = _load_simcheck(args.simcheck_dir)
    if args.input_dir is not None:
        if args.cam1 is not None or args.cam2 is not None:
            raise ValueError("--input-dir cannot be combined with --cam1/--cam2")
        input_dir = args.input_dir.resolve()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {input_dir}")
        cam1 = _discover_cam1(input_dir).resolve()
        cam2 = _discover_cam2(input_dir).resolve()
    else:
        if args.cam1 is None or args.cam2 is None:
            raise ValueError("provide --input-dir or both --cam1 and --cam2")
        cam1 = args.cam1.resolve()
        cam2 = _resolve_cam2(args.cam2).resolve()
    if not cam1.is_file():
        raise FileNotFoundError(f"cam1 file does not exist: {cam1}")

    frames_per_timepoint = args.n_z * args.n_directions * args.n_phases

    def _truncate(shape: tuple[int, int, int]) -> tuple[int, int, int]:
        pages = shape[0]
        complete = (pages // frames_per_timepoint) * frames_per_timepoint
        if complete != pages:
            print(
                f"warning: {pages} pages is not a multiple of {frames_per_timepoint} "
                f"pages per timepoint; dropping trailing {pages - complete} frames"
            )
            pages = complete
        if pages == 0:
            raise ValueError(
                f"no complete timepoint in {shape[0]} pages "
                f"({frames_per_timepoint} per timepoint)"
            )
        return (pages, shape[1], shape[2])

    cam1_shape = _truncate(_tiff_shape(cam1))
    cam2_shape = _truncate(_tiff_shape(cam2))
    if cam1_shape != cam2_shape:
        raise ValueError(f"cam1/cam2 shapes differ: {cam1_shape} vs {cam2_shape}")
    pages = cam1_shape[0]
    timepoints = pages // frames_per_timepoint

    if args.sample_evenly:
        picks = _sample_evenly(timepoints)
    else:
        picks = _sample_timepoints(timepoints)
    output_root = args.output_root.resolve() if args.output_root else cam1.parent / "raw_fpj"
    output_root.mkdir(parents=True, exist_ok=True)

    peak_orders = args.peak_orders
    if peak_orders is None:
        peak_orders = {3: 1, 5: 2}.get(args.n_phases, 0)
    region_shape = (cam1_shape[1], cam1_shape[2] // 4)
    print(f"cam1: {cam1} shape={cam1_shape}")
    print(f"cam2: {cam2} shape={cam2_shape}")
    print(
        f"timepoints: {timepoints} total, processing {len(picks)} "
        f"({frames_per_timepoint} pages each): {picks} "
        f"[{'evenly sampled' if args.sample_evenly else 'first leading'}]"
    )
    print(f"regions: {region_shape}; output: {output_root}")
    for timepoint_index in picks:
        start = timepoint_index * frames_per_timepoint
        stop = start + frames_per_timepoint
        cam1_chunk = _read_pages(cam1, start, stop)
        cam2_chunk = _read_pages(cam2, start, stop)
        region4 = _extract_region(cam1_chunk, flip_x=False)
        region5 = _extract_region(cam2_chunk, flip_x=True)
        raw = _to_raw_tensor(
            region4,
            region5,
            n_z=args.n_z,
            n_directions=args.n_directions,
            n_phases=args.n_phases,
        )
        timepoint = timepoint_index + 1
        written = _write_timepoint(
            simcheck=simcheck,
            raw=raw,
            output_root=output_root,
            timepoint=timepoint,
            peak_orders=peak_orders,
            overwrite=args.overwrite,
            preview_fpj=None if args.no_preview else preview_fpj,
        )
        print(f"T{timepoint:03d}: wrote {', '.join(p.name for p in written)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute raw-data FPJ for regions 4 and 5 without registration."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="directory holding cam1*/cam2* TIFFs to auto-discover (mutually "
        "exclusive with --cam1/--cam2)",
    )
    parser.add_argument("--cam1", type=Path, default=None, help="cam1 raw TIFF")
    parser.add_argument(
        "--cam2", type=Path, default=None, help="cam2 raw TIFF or its directory"
    )
    parser.add_argument("--output-root", type=Path, help="default: cam1 parent/raw_fpj")
    parser.add_argument("--n-z", type=_positive_int, default=1)
    parser.add_argument("--n-directions", type=_positive_int, default=3)
    parser.add_argument("--n-phases", type=_positive_int, default=5)
    parser.add_argument("--peak-orders", type=int, default=None)
    parser.add_argument(
        "--sample-evenly",
        action="store_true",
        help="sample up to 10 timepoints evenly across the full range instead "
        "of using only the first leading timepoints (default)",
    )
    parser.add_argument("--simcheck-dir", type=Path, default=DEFAULT_SIMCHECK_DIR)
    parser.add_argument("--no-preview", action="store_true", help="skip FPJ PNG previews")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
