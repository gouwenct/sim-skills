from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


TIFF_SUFFIXES = {".tif", ".tiff"}
NUMBERED_DIR_PATTERN = re.compile(r"^(?P<number>\d{2})、")


@dataclass
class MoveSummary:
    planned_renames: int = 0
    renamed: int = 0
    planned: int = 0
    moved: int = 0
    skipped: int = 0
    conflicts: int = 0
    errors: int = 0


def _natural_name_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name.casefold())
    )


def _number_data_directories(
    data_root: Path,
    execute: bool,
    summary: MoveSummary,
) -> list[tuple[Path, Path]]:
    data_dirs = [path for path in data_root.iterdir() if path.is_dir()]
    used_numbers = {
        int(match.group("number"))
        for path in data_dirs
        if (match := NUMBERED_DIR_PATTERN.match(path.name))
        and match.group("number") != "99"
    }
    next_number = 1
    directory_locations: list[tuple[Path, Path]] = []

    for data_dir in sorted(data_dirs, key=_natural_name_key):
        if NUMBERED_DIR_PATTERN.match(data_dir.name):
            directory_locations.append((data_dir, data_dir))
            continue

        if data_dir.name.casefold().startswith("peizhun"):
            number = 99
        else:
            while next_number in used_numbers or next_number == 99:
                next_number += 1
            if next_number > 98:
                raise ValueError("No free directory numbers remain in the 01-98 range")
            number = next_number
            used_numbers.add(number)
            next_number += 1

        numbered_dir = data_dir.with_name(f"{number:02d}、{data_dir.name}")
        if numbered_dir.exists():
            summary.conflicts += 1
            print(f"[CONFLICT] Numbered directory already exists: {numbered_dir}")
            directory_locations.append((data_dir, data_dir))
            continue

        summary.planned_renames += 1
        if not execute:
            print(f"[RENAME PLAN] {data_dir} -> {numbered_dir}")
            directory_locations.append((data_dir, numbered_dir))
            continue

        try:
            data_dir.rename(numbered_dir)
            summary.renamed += 1
            print(f"[RENAMED] {data_dir} -> {numbered_dir}")
            directory_locations.append((numbered_dir, numbered_dir))
        except OSError as exc:
            summary.errors += 1
            print(f"[ERROR] {data_dir}: {exc}")
            directory_locations.append((data_dir, data_dir))

    return directory_locations


def move_cam2_tiffs(data_root: Path, execute: bool = False) -> MoveSummary:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Data root does not exist: {data_root}")

    summary = MoveSummary()
    move_plan: list[tuple[Path, Path]] = []

    directory_locations = _number_data_directories(data_root, execute, summary)
    for source_dir, destination_dir in directory_locations:
        cam2_dir = source_dir / "cam2"
        if not cam2_dir.is_dir():
            summary.skipped += 1
            print(f"[SKIP] No cam2 directory: {source_dir}")
            continue

        cam2_tiffs = sorted(
            path
            for path in cam2_dir.iterdir()
            if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
        )
        if not cam2_tiffs:
            summary.skipped += 1
            print(f"[SKIP] No direct TIFF files: {cam2_dir}")
            continue

        for source in cam2_tiffs:
            destination = destination_dir / source.name
            if destination.exists():
                summary.conflicts += 1
                print(f"[CONFLICT] Target already exists: {destination}")
                continue
            move_plan.append((source, destination))

    summary.planned = len(move_plan)

    for source, destination in move_plan:
        if not execute:
            print(f"[PLAN] {source} -> {destination}")
            continue

        try:
            if destination.exists():
                summary.conflicts += 1
                print(f"[CONFLICT] Target appeared before move: {destination}")
                continue
            source.rename(destination)
            summary.moved += 1
            print(f"[MOVED] {source} -> {destination}")
        except OSError as exc:
            summary.errors += 1
            print(f"[ERROR] {source}: {exc}")

    mode = "EXECUTE" if execute else "DRY-RUN"
    print(
        f"[{mode} SUMMARY] planned_renames={summary.planned_renames} "
        f"renamed={summary.renamed} planned={summary.planned} moved={summary.moved} "
        f"skipped={summary.skipped} conflicts={summary.conflicts} "
        f"errors={summary.errors}"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Number first-level data directories, then move direct TIFF "
        "files from each cam2 folder into its parent data directory."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Data root to process",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply planned renames and moves; the default is dry-run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = move_cam2_tiffs(args.data_root, execute=args.execute)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 2
    return 1 if summary.conflicts or summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
