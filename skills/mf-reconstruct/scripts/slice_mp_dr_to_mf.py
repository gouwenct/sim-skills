"""
将 3D-MP Dr（单时间点，Z×(3方向×5相位)=8×15 帧）切成 8 层 MF-SIM 2D slice。

约定（当前 MP8 / sequence55）：
  输入 shape: (120, H, W) = (8 * 15, H, W)
  reshape → (8, 15, H, W)
  每层输出: (15, H, W)

示例：
  python slice_mp_dr_to_mf.py ^
    --src ".../14、mp_51/Dr/01-488_padding/Dr_stack-..._0001.tif" ^
    --out-dir ".../14、mp_51/Dr/01-488_padding_slice"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tifffile

N_Z = 8
N_FRAMES_PER_Z = 15  # 3 angles × 5 phases
EXPECTED_Z = N_Z * N_FRAMES_PER_Z


def slice_one_stack(
    src: Path,
    out_dir: Path,
    *,
    n_z: int = N_Z,
    n_frames: int = N_FRAMES_PER_Z,
    name_prefix: str = "MF-slice",
    overwrite: bool = False,
) -> list[Path]:
    data = tifffile.imread(src)
    if data.ndim != 3:
        raise ValueError(f"expect 3D stack (Z*frames, H, W), got {data.shape} from {src}")

    total, h, w = data.shape
    expected = n_z * n_frames
    if total != expected:
        raise ValueError(
            f"{src.name}: first dim={total}, expect {expected} (={n_z}*{n_frames})"
        )

    volume = data.reshape(n_z, n_frames, h, w)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for zi in range(n_z):
        out_path = out_dir / f"{name_prefix}_z{zi:02d}.tif"
        if out_path.exists() and not overwrite:
            print(f"skip existing: {out_path}")
            written.append(out_path)
            continue
        tifffile.imwrite(
            out_path,
            volume[zi].astype("float32"),
            imagej=True,
            photometric="minisblack",
            metadata={"axes": "ZYX"},
        )
        print(f"write: {out_path} shape={volume[zi].shape}")
        written.append(out_path)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slice one MP Dr timepoint into MF Z-slices")
    p.add_argument("--src", type=Path, required=True, help="source Dr stack tif (120,H,W)")
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="output Dr subfolder, e.g. .../Dr/01-488_padding_slice",
    )
    p.add_argument("--n-z", type=int, default=N_Z)
    p.add_argument("--n-frames", type=int, default=N_FRAMES_PER_Z)
    p.add_argument("--name-prefix", type=str, default="MF-slice")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    out_dir = args.out_dir.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    slice_one_stack(
        src,
        out_dir,
        n_z=args.n_z,
        n_frames=args.n_frames,
        name_prefix=args.name_prefix,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
