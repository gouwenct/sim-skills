from __future__ import annotations

import argparse
from pathlib import Path

import tifffile

import Utils as _U

TIFF_SUFFIXES = {".tif", ".tiff"}


def expand_batch(
    root_dir: Path,
    *,
    target_x: int = 462,
    target_y: int = 462,
    target_z: int = 120,
    save_suffix: str = "_padding",
) -> list[Path]:
    written: list[Path] = []
    for tif_path in sorted(root_dir.glob("*/Dr/*/*")):
        if not tif_path.is_file() or tif_path.suffix.lower() not in TIFF_SUFFIXES:
            continue
        if tif_path.parent.name.endswith(save_suffix):
            continue

        save_dir = tif_path.parent.with_name(f"{tif_path.parent.name}{save_suffix}")
        save_path = save_dir / f"{tif_path.stem}.tif"
        if save_path.exists():
            print(f"skip existing: {save_path}")
            continue

        image = tifffile.imread(tif_path)
        image = _U.pad_or_cut(image, x=target_x, y=target_y, z=target_z)
        image = _U.image_reshape(image, dim=3)
        save_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            save_path,
            image.astype("float32"),
            imagej=True,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        print(f"finish: {tif_path}")
        written.append(save_path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pad MP-SIM Dr TIFF stacks")
    parser.add_argument("root_dir", type=Path, help="dataOut batch root")
    parser.add_argument("--x", type=int, default=462)
    parser.add_argument("--y", type=int, default=462)
    parser.add_argument("--z", type=int, default=120)
    parser.add_argument("--suffix", default="_padding")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(root_dir)
    expand_batch(
        root_dir,
        target_x=args.x,
        target_y=args.y,
        target_z=args.z,
        save_suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
