#!/usr/bin/env python
"""Check paired MP-SIM raw TIFFs for dropped or repeated frames."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}
CAM1_TIME_RE = re.compile(
    r"Time_From_Start\s*=\s*(\d+):(\d+):(\d+(?:\.\d+)?)"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be positive, got {parsed}")
    return parsed


def _gap_factor(value: str) -> float:
    parsed = float(value)
    if parsed <= 1:
        raise argparse.ArgumentTypeError(f"must be greater than 1, got {parsed}")
    return parsed


def _natural_key(path: Path) -> list[tuple[int, Any]]:
    parts = re.split(r"(\d+)", str(path).lower())
    return [(1, int(part)) if part.isdigit() else (0, part) for part in parts]


def _discover_camera_files(root: Path, camera: str) -> list[Path]:
    files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in TIFF_SUFFIXES
        and path.name.lower().startswith(camera)
    ]
    camera_dir = root / camera
    if camera_dir.is_dir():
        files.extend(
            path
            for path in camera_dir.iterdir()
            if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
        )
    unique = {path.resolve(): path.resolve() for path in files}
    result = sorted(unique.values(), key=_natural_key)
    if not result:
        raise ValueError(
            f"no {camera} TIFF found directly under {root} or {camera_dir}"
        )
    return result


def _page_time_ms(page: tifffile.TiffPage) -> float | None:
    tag = page.tags.get("MicroManagerMetadata")
    if tag is not None and isinstance(tag.value, dict):
        value = tag.value.get("ElapsedTime-ms")
        if value is not None:
            return float(value)
    match = CAM1_TIME_RE.search(page.description or "")
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return (int(hours) * 3600 + int(minutes) * 60 + float(seconds)) * 1000


def _page_sequence_index(page: tifffile.TiffPage) -> int | None:
    tag = page.tags.get("MicroManagerMetadata")
    if tag is None or not isinstance(tag.value, dict):
        return None
    value = tag.value.get("FrameIndex", tag.value.get("Frame"))
    return None if value is None else int(value)


def _timepoint(frame_number: int, frames_per_timepoint: int) -> int:
    return (frame_number - 1) // frames_per_timepoint + 1


def _position(
    frame_number: int, *, n_z: int, n_directions: int, n_phases: int
) -> tuple[int, int, int]:
    within = (frame_number - 1) % (n_z * n_directions * n_phases)
    z_index, within_z = divmod(within, n_directions * n_phases)
    direction_index, phase_index = divmod(within_z, n_phases)
    return z_index + 1, direction_index + 1, phase_index + 1


def _metadata_gaps(
    *,
    camera: str,
    timestamps: list[float | None],
    sequence_indices: list[int | None],
    frame_offset: int,
    frames_per_timepoint: int,
    gap_factor: float,
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    indexed_after_frames: set[int] = set()
    for index in range(1, len(sequence_indices)):
        previous = sequence_indices[index - 1]
        current = sequence_indices[index]
        if previous is None or current is None or current <= previous + 1:
            continue
        after_frame = frame_offset + index
        indexed_after_frames.add(after_frame)
        gaps.append(
            {
                "type": "frame_index_gap",
                "camera": camera,
                "after_frame": after_frame,
                "before_frame": after_frame + 1,
                "missing": current - previous - 1,
                "timepoint": after_frame // frames_per_timepoint + 1,
            }
        )

    intervals = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
        if timestamps[index] is not None
        and timestamps[index - 1] is not None
        and timestamps[index] > timestamps[index - 1]
    ]
    if len(intervals) < 3:
        return gaps
    median_interval = float(np.median(intervals))
    if median_interval <= 0:
        return gaps
    for index in range(1, len(timestamps)):
        previous = timestamps[index - 1]
        current = timestamps[index]
        if previous is None or current is None:
            continue
        interval = current - previous
        after_frame = frame_offset + index
        estimated_missing = max(0, int(round(interval / median_interval)) - 1)
        if (
            after_frame not in indexed_after_frames
            and interval > median_interval * gap_factor
            and estimated_missing > 0
        ):
            gaps.append(
                {
                    "type": "timing_gap",
                    "camera": camera,
                    "after_frame": after_frame,
                    "before_frame": after_frame + 1,
                    "missing": estimated_missing,
                    "timepoint": after_frame // frames_per_timepoint + 1,
                    "interval_ms": interval,
                    "median_interval_ms": median_interval,
                }
            )
    return gaps


def _scan_camera(
    camera: str,
    files: list[Path],
    *,
    frames_per_timepoint: int,
    gap_factor: float,
) -> dict[str, Any]:
    frame_count = 0
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    previous_stats: tuple[float, float, float, float] | None = None
    previous_location: tuple[Path, int] | None = None
    duplicates: list[dict[str, Any]] = []
    metadata_gaps: list[dict[str, Any]] = []

    for path in files:
        file_offset = frame_count
        timestamps: list[float | None] = []
        sequence_indices: list[int | None] = []
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                raise ValueError(f"empty TIFF: {path}")
            for page_index, page in enumerate(tif.pages):
                array = np.asarray(page.asarray())
                if array.ndim != 2:
                    raise ValueError(
                        f"expected one Y,X image per page in {path}, got {array.shape}"
                    )
                current_shape = tuple(int(value) for value in array.shape)
                current_dtype = str(array.dtype)
                if shape is None:
                    shape, dtype = current_shape, current_dtype
                elif current_shape != shape or current_dtype != dtype:
                    raise ValueError(
                        f"inconsistent page geometry in {camera}: "
                        f"expected {shape} {dtype}, got {current_shape} {current_dtype} "
                        f"at {path} page {page_index + 1}"
                    )
                if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                    raise ValueError(f"non-finite pixels in {path} page {page_index + 1}")
                stats = (
                    float(array.mean(dtype=np.float64)),
                    float(array.max()),
                    float(array.min()),
                    float(array.std(dtype=np.float64)),
                )
                frame_count += 1
                if previous_stats == stats and previous_location is not None:
                    duplicates.append(
                        {
                            "camera": camera,
                            "previous_frame": frame_count - 1,
                            "current_frame": frame_count,
                            "previous_file": previous_location[0],
                            "previous_page": previous_location[1],
                            "current_file": path,
                            "current_page": page_index + 1,
                        }
                    )
                previous_stats = stats
                previous_location = (path, page_index + 1)
                timestamps.append(_page_time_ms(page))
                sequence_indices.append(_page_sequence_index(page))
        metadata_gaps.extend(
            _metadata_gaps(
                camera=camera,
                timestamps=timestamps,
                sequence_indices=sequence_indices,
                frame_offset=file_offset,
                frames_per_timepoint=frames_per_timepoint,
                gap_factor=gap_factor,
            )
        )

    return {
        "camera": camera,
        "files": files,
        "frames": frame_count,
        "shape": shape,
        "dtype": dtype,
        "bytes": sum(path.stat().st_size for path in files),
        "duplicates": duplicates,
        "metadata_gaps": metadata_gaps,
    }


def run(args: argparse.Namespace) -> int:
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    frames_per_timepoint = args.n_z * args.n_directions * args.n_phases
    scans = {
        camera: _scan_camera(
            camera,
            _discover_camera_files(root, camera),
            frames_per_timepoint=frames_per_timepoint,
            gap_factor=args.gap_factor,
        )
        for camera in ("cam1", "cam2")
    }
    cam1, cam2 = scans["cam1"], scans["cam2"]
    if (cam1["shape"], cam1["dtype"]) != (cam2["shape"], cam2["dtype"]):
        raise ValueError(
            "cam1/cam2 page geometry differs: "
            f"{cam1['shape']} {cam1['dtype']} vs {cam2['shape']} {cam2['dtype']}"
        )

    print(f"input: {root}")
    print(
        f"acquisition: n_z={args.n_z} directions={args.n_directions} "
        f"phases={args.n_phases} frames_per_timepoint={frames_per_timepoint}"
    )
    for camera in ("cam1", "cam2"):
        scan = scans[camera]
        print(
            f"{camera}: files={len(scan['files'])} frames={scan['frames']} "
            f"shape={scan['shape']} dtype={scan['dtype']} bytes={scan['bytes']}"
        )
        for path in scan["files"]:
            print(f"  {path}")
    print(
        "FILE_SIZE_DIFF_KB: "
        f"{abs(cam1['bytes'] - cam2['bytes']) / 1024.0:.3f} "
        "(informational; TIFF metadata overhead differs by camera)"
    )

    anomalies: list[dict[str, Any]] = []
    if cam1["frames"] != cam2["frames"]:
        print(
            "CAMERA_FRAME_COUNT_MISMATCH: "
            f"cam1={cam1['frames']} cam2={cam2['frames']}"
        )
        anomalies.append({"type": "camera_frame_count_mismatch"})
    for camera in ("cam1", "cam2"):
        scan = scans[camera]
        remainder = scan["frames"] % frames_per_timepoint
        if remainder:
            timepoint = scan["frames"] // frames_per_timepoint + 1
            print(
                f"INCOMPLETE_TIMEPOINT: camera={camera} timepoint={timepoint} "
                f"frames={remainder}/{frames_per_timepoint}"
            )
            anomalies.append({"type": "incomplete_timepoint"})
        for duplicate in scan["duplicates"]:
            frame = duplicate["current_frame"]
            z_index, direction, phase = _position(
                frame,
                n_z=args.n_z,
                n_directions=args.n_directions,
                n_phases=args.n_phases,
            )
            print(
                f"DUPLICATE_STATS: camera={camera} "
                f"frames={duplicate['previous_frame']},{frame} "
                f"timepoint={_timepoint(frame, frames_per_timepoint)} "
                f"z={z_index} direction={direction} phase={phase} "
                f"locations={duplicate['previous_file']}#{duplicate['previous_page']},"
                f"{duplicate['current_file']}#{duplicate['current_page']}"
            )
            anomalies.append({"type": "duplicate_stats"})
        for gap in scan["metadata_gaps"]:
            label = "FRAME_INDEX_GAP" if gap["type"] == "frame_index_gap" else "TIMING_GAP"
            print(
                f"{label}: camera={camera} after_frame={gap['after_frame']} "
                f"before_frame={gap['before_frame']} "
                f"estimated_missing={gap['missing']} timepoint={gap['timepoint']}"
            )
            anomalies.append(gap)

    print(f"RESULT: {'ANOMALIES' if anomalies else 'CLEAN'}")
    return 1 if anomalies else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check all paired cam1/cam2 raw TIFF pages for dropped or repeated frames."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="sample directory containing cam1*/cam2* TIFFs or cam1/cam2 subdirectories",
    )
    parser.add_argument("--n-z", type=_positive_int, default=1)
    parser.add_argument("--n-directions", type=_positive_int, default=3)
    parser.add_argument("--n-phases", type=_positive_int, default=5)
    parser.add_argument(
        "--gap-factor",
        type=_gap_factor,
        default=1.75,
        help="flag timestamp intervals larger than this multiple of the median (default: 1.75)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
