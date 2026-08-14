#!/usr/bin/env python
"""Stack same-type 3D TIFFs in a folder into a sibling ImageJ TZYX hyperstack."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

TIFF_SUFFIXES = {".tif", ".tiff"}
HYPERSTACK_NAME = re.compile(r"hyperstack", re.IGNORECASE)
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
DEFAULT_MAX_DEPTH = 2
AMBIGUOUS_EXIT = 2


def natural_key(name: str) -> list[Any]:
    parts = NUMBER_RE.split(name)
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        if NUMBER_RE.fullmatch(part):
            key.append(float(part))
        else:
            key.append(part.lower())
    return key


def _is_tiff(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in TIFF_SUFFIXES


def inspect_tiff(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0]
        array = series.asarray()
        axes = str(series.axes or "")
        imagej = handle.imagej_metadata or {}
        imagej_axes = str(imagej.get("axes") or "")
        frames = int(imagej.get("frames") or 0)
        hyper = bool(imagej.get("hyperstack"))
    return {
        "path": path,
        "array": np.asarray(array),
        "ndim": int(array.ndim),
        "shape": tuple(int(v) for v in array.shape),
        "axes": axes.upper(),
        "imagej_axes": imagej_axes.upper(),
        "frames": frames,
        "imagej_hyperstack": hyper,
    }


def is_skipped_hyperstack(info: dict[str, Any]) -> bool:
    name = info["path"].name
    if HYPERSTACK_NAME.search(name):
        return True
    if info["ndim"] >= 4:
        return True
    axes = info["imagej_axes"] or info["axes"]
    if info["imagej_hyperstack"] or info["frames"] > 1:
        return True
    if "T" in axes and "Z" in axes:
        return True
    if info["ndim"] == 3 and "T" in axes and "Z" not in axes:
        return True
    return False


def qualified_3d_tiffs(folder: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in folder.iterdir():
        if not _is_tiff(path):
            continue
        info = inspect_tiff(path)
        if is_skipped_hyperstack(info):
            continue
        if info["ndim"] != 3:
            continue
        axes = info["imagej_axes"] or info["axes"]
        if axes and axes not in {"ZYX", "YXZ", "ZXY"} and "Z" in axes and axes != "ZYX":
            # unlabeled or ZYX-compatible 3D only; unknown labeled 3D kept as ZYX per lock
            pass
        candidates.append(info)
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: natural_key(item["path"].name))
    majority_shape = Counter(item["shape"] for item in ordered).most_common(1)[0][0]
    return [item for item in ordered if item["shape"] == majority_shape]


def child_dirs(folder: Path) -> list[Path]:
    return sorted((p for p in folder.iterdir() if p.is_dir()), key=lambda p: natural_key(p.name))


def contains_tiff_folder(folder: Path) -> bool:
    if len(qualified_3d_tiffs(folder)) >= 2:
        return True
    return any(contains_tiff_folder(child) for child in child_dirs(folder))


def classify(folder: Path) -> str:
    here = qualified_3d_tiffs(folder)
    stackable_children = [child for child in child_dirs(folder) if len(qualified_3d_tiffs(child)) >= 2]
    has_here = len(here) >= 2
    has_below = len(stackable_children) >= 1
    if has_here and has_below:
        return "ambiguous"
    if has_here:
        return "self"
    if has_below:
        return "children"
    return "empty"


def discover_tiff_folders(root: Path, max_depth: int) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (stackable folders, ambiguous folders, deeper-than-max folders)."""
    stackable: list[Path] = []
    ambiguous: list[Path] = []
    too_deep: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        action = classify(folder)
        if action == "ambiguous":
            if depth <= max_depth:
                ambiguous.append(folder)
            else:
                too_deep.append(folder)
            return
        if action == "self":
            if depth <= max_depth:
                stackable.append(folder)
            else:
                too_deep.append(folder)
            return
        if action == "children":
            for child in child_dirs(folder):
                if len(qualified_3d_tiffs(child)) >= 2:
                    if depth + 1 <= max_depth:
                        stackable.append(child)
                    else:
                        too_deep.append(child)
                elif classify(child) in {"children", "ambiguous", "empty"}:
                    if depth + 1 <= max_depth:
                        walk(child, depth + 1)
                    elif contains_tiff_folder(child):
                        too_deep.append(child)
            return
        for child in child_dirs(folder):
            if depth + 1 <= max_depth:
                walk(child, depth + 1)
            elif contains_tiff_folder(child):
                too_deep.append(child)

    walk(root, 0)
    unique_stackable = []
    seen: set[Path] = set()
    for folder in stackable:
        key = folder.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_stackable.append(folder)
    return unique_stackable, ambiguous, too_deep


def output_path(folder: Path, output_dir: Path | None) -> Path:
    dest_dir = output_dir if output_dir is not None else folder.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / f"{folder.name}_hyperstack.tif"
    if not candidate.exists():
        return candidate
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stamped = dest_dir / f"{folder.name}_hyperstack-{stamp}.tif"
    index = 1
    while stamped.exists():
        stamped = dest_dir / f"{folder.name}_hyperstack-{stamp}-{index}.tif"
        index += 1
    return stamped


def write_hyperstack(folder: Path, output_dir: Path | None = None) -> Path:
    items = sorted(qualified_3d_tiffs(folder), key=lambda item: natural_key(item["path"].name))
    if len(items) < 2:
        raise ValueError(f"need at least 2 same-shape 3D TIFFs in {folder}")
    volumes = [np.asarray(item["array"], dtype=np.float32) for item in items]
    stack = np.stack(volumes, axis=0)
    labels = [item["path"].stem for item in items]
    dest = output_path(folder, output_dir)
    tifffile.imwrite(
        dest,
        stack,
        imagej=True,
        photometric="minisblack",
        metadata={"axes": "TZYX", "mode": "grayscale", "Labels": labels},
    )
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a folder of same-type 3D TIFFs into a sibling ImageJ TZYX hyperstack."
    )
    parser.add_argument("input_folder", type=Path, help="Folder to inspect")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for all hyperstacks. Default: beside each stacked folder.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="0=specified folder only, 1=+children, 2=+grandchildren (default 2).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.input_folder.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    if args.max_depth < 0:
        print("ERROR: --max-depth must be >= 0", file=sys.stderr)
        return 1

    stackable, ambiguous, too_deep = discover_tiff_folders(root, args.max_depth)
    if ambiguous:
        print("NEED_CONFIRM: both this layer and a child folder have qualified 3D TIFFs:")
        for folder in ambiguous:
            print(f"  {folder}")
        if not stackable:
            return AMBIGUOUS_EXIT

    written: list[Path] = []
    for folder in stackable:
        dest = output_path(folder, args.output_dir)
        print(f"{'DRY' if args.dry_run else 'WRITE'} {folder} -> {dest}")
        if not args.dry_run:
            dest = write_hyperstack(folder, args.output_dir)
            print(f"  saved {dest}")
        written.append(dest)

    if not stackable and not ambiguous and not too_deep:
        print(f"ERROR: no qualified 3D TIFF folder under {root} within depth {args.max_depth}")
        return 1

    if too_deep:
        print("NEED_CONFIRM: folders deeper than --max-depth were not processed:")
        for folder in too_deep:
            print(f"  {folder}")
        return AMBIGUOUS_EXIT

    if ambiguous:
        return AMBIGUOUS_EXIT
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
