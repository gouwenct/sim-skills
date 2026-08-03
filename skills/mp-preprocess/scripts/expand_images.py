from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

TIFF_SUFFIXES = {".tif", ".tiff"}


def _as_3d(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    while image.ndim < 3:
        image = np.expand_dims(image, axis=0)
    while image.ndim > 3:
        image = image.reshape((image.shape[0] * image.shape[1], *image.shape[2:]))
    return image


def _center_pad_or_crop(image: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    image = _as_3d(image)
    slices = []
    padding = []
    for current, target in zip(image.shape, shape):
        if target <= 0:
            raise ValueError("target dimensions must be positive")
        start = max(0, (current - target + 1) // 2)
        slices.append(slice(start, start + min(current, target)))
        difference = max(0, target - current)
        padding.append(((difference + 1) // 2, difference // 2))
    return np.pad(image[tuple(slices)], padding, mode="constant")


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

        image = _center_pad_or_crop(
            tifffile.imread(tif_path), (target_z, target_x, target_y)
        )
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
