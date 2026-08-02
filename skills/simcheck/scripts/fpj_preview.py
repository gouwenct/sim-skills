#!/usr/bin/env python3
"""Make chat-friendly PNG previews for SIMcheck FPJ and recon lihe TIFFs.

Usage:
  # FPJ merged-page preview
  python -B preview_for_chat.py fpj path/to/FPJ_C1.tif --label "13、35"

  # lihe: center-half XY + min-max contrast; max-proj; D peaks + optional MCNR
  python -B preview_for_chat.py lihe path/to/lihe_....tif --label "13、35"

Outputs PNG next to source (or --out), with data name and direction tags.
lihe preview crops central half of XY (H/2×W/2) and uses min–max stretch.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import tifffile

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Pillow required: {exc}") from exc


def _to_uint8(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.8,
    mode: str = "percentile",
) -> np.ndarray:
    """Normalize 2D image to uint8.

    mode:
      - percentile: robust display (default for FPJ)
      - minmax: full min–max stretch (default for lihe)
    """
    x = np.asarray(img, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if mode == "minmax":
        lo, hi = float(x.min()), float(x.max())
    else:
        lo, hi = np.percentile(x, [p_low, p_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    return (y * 255.0).astype(np.uint8)


def _center_half_xy(img: np.ndarray) -> np.ndarray:
    """Keep only the central half in XY (e.g. 1024×1024 → center 512×512).

    Works on (..., H, W). Odd sizes floor to H//2, W//2.
    """
    a = np.asarray(img)
    if a.ndim < 2:
        raise ValueError(f"need at least 2D for center crop, got {a.shape}")
    h, w = int(a.shape[-2]), int(a.shape[-1])
    ch, cw = max(1, h // 2), max(1, w // 2)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return a[..., y0 : y0 + ch, x0 : x0 + cw]


def _load_font(size: int = 18):
    for name in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ):
        p = Path(name)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font) -> None:
    x, y = xy
    # dark outline for readability on bright peaks
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 80), font=font)


def preview_fpj(tif_path: Path, label: str, out: Optional[Path] = None) -> Path:
    arr = tifffile.imread(str(tif_path))
    if arr.ndim == 3:
        plane = arr[0]
    elif arr.ndim == 2:
        plane = arr
    else:
        plane = np.max(arr, axis=tuple(range(arr.ndim - 2)))

    u8 = _to_uint8(plane)
    rgb = np.stack([u8, u8, u8], axis=-1)
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(20)
    title = f"{label} | FPJ_C1"
    _draw_label(draw, (8, 8), title, font)

    out = out or tif_path.with_name(tif_path.stem + f"__{ _safe(label) }__preview.png")
    im.save(out)
    print(out)
    return out


def preview_fpj_pages(
    tif_path: Path,
    label: str,
    out_dir: Path,
) -> dict[str, Path]:
    arr = tifffile.imread(str(tif_path))
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"FPJ expected merged+direction pages, got {arr.shape}")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = ["merged"] + [f"D{i}" for i in range(1, arr.shape[0])]
    written = {}
    for index, name in enumerate(names):
        path = out_dir / f"{tif_path.stem}_{name}.png"
        u8 = _to_uint8(arr[index])
        rgb = np.stack([u8, u8, u8], axis=-1)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)
        _draw_label(draw, (8, 8), f"{label} | {name}", _load_font(20))
        image.save(path)
        written[name] = path
    return written


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)[:80]


def _peak_xy(img2d: np.ndarray) -> Tuple[int, int, float]:
    x = np.asarray(img2d, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)
    # suppress exact center artifact sometimes injected as 0.1
    cy, cx = x.shape[0] // 2, x.shape[1] // 2
    x = x.copy()
    x[cy, cx] = 0
    idx = np.argmax(x)
    yy, xx = np.unravel_index(idx, x.shape)
    return int(xx), int(yy), float(x[yy, xx])


def _draw_arrow_to_peak(
    draw: ImageDraw.ImageDraw,
    xx: int,
    yy: int,
    im_w: int,
    im_h: int,
    color: Tuple[int, int, int],
    tag: str,
    font,
    *,
    gap: int = 14,
    shaft: int = 28,
    head: int = 8,
    angle_deg: float = -35.0,
) -> None:
    """Draw an arrow that points at the peak but does not cover it.

    Arrow tip stops ``gap`` pixels short of (xx, yy). Label sits near the tail.
    """
    # Prefer outward radial direction from image center so arrows leave the ring
    cx, cy = im_w / 2.0, im_h / 2.0
    vx, vy = float(xx) - cx, float(yy) - cy
    nrm = (vx * vx + vy * vy) ** 0.5
    if nrm < 1e-6:
        # peak near center: default to upper-right approach
        ux, uy = 0.7071, -0.7071
    else:
        # unit vector from center → peak; approach from outside (same direction)
        ux, uy = vx / nrm, vy / nrm

    # Optional small angular offset so multi-dir arrows do not stack
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    ux2 = ux * cos_a - uy * sin_a
    uy2 = ux * sin_a + uy * cos_a
    n2 = (ux2 * ux2 + uy2 * uy2) ** 0.5
    if n2 > 1e-6:
        ux, uy = ux2 / n2, uy2 / n2

    # Tip just outside the bright core
    tip_x = xx - ux * gap
    tip_y = yy - uy * gap
    # Tail further outward
    tail_x = tip_x - ux * shaft
    tail_y = tip_y - uy * shaft

    # Clamp to image with margin
    def _clamp(x: float, y: float) -> Tuple[float, float]:
        m = 4.0
        return (
            min(max(x, m), im_w - 1 - m),
            min(max(y, m), im_h - 1 - m),
        )

    tip_x, tip_y = _clamp(tip_x, tip_y)
    tail_x, tail_y = _clamp(tail_x, tail_y)

    # Shaft
    draw.line((tail_x, tail_y, tip_x, tip_y), fill=color, width=2)

    # Arrow head (two short strokes)
    # perpendicular
    px, py = -uy, ux
    h1x = tip_x - ux * head + px * (head * 0.55)
    h1y = tip_y - uy * head + py * (head * 0.55)
    h2x = tip_x - ux * head - px * (head * 0.55)
    h2y = tip_y - uy * head - py * (head * 0.55)
    draw.line((tip_x, tip_y, h1x, h1y), fill=color, width=2)
    draw.line((tip_x, tip_y, h2x, h2y), fill=color, width=2)

    # Label near tail, offset further outward so it is not on the peak
    lab_x = int(tail_x - ux * 10 - 6)
    lab_y = int(tail_y - uy * 10 - 8)
    lab_x = max(2, min(lab_x, im_w - 36))
    lab_y = max(2, min(lab_y, im_h - 20))
    _draw_label(draw, (lab_x, lab_y), tag, font)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return int(box[2] - box[0]), int(box[3] - box[1])


def _draw_label_top_right(
    draw: ImageDraw.ImageDraw,
    im_w: int,
    lines: Sequence[str],
    font,
    margin: int = 10,
    line_gap: int = 4,
) -> None:
    """Draw multi-line text block anchored to top-right."""
    sizes = [_text_size(draw, ln, font) for ln in lines]
    block_w = max((w for w, _ in sizes), default=0)
    y = margin
    for (tw, th), ln in zip(sizes, lines):
        x = max(margin, im_w - margin - tw)
        _draw_label(draw, (x, y), ln, font)
        y += th + line_gap


def preview_lihe(
    tif_path: Path,
    label: str,
    out: Optional[Path] = None,
    mcnr: Optional[dict] = None,
) -> Path:
    """Preview max-projection layer of lihe stack and tag direction peaks.

    essential stack convention (Para_Estimate_GPU / docs):
      layer0: max of directions
      layer1..3: direction 1..3
    debug/full sometimes inserts min then max then directions -> (5,H,W):
      layer0 min, layer1 max, layer2..4 dirs

    mcnr: optional dict with D1/D2/D3/mean and optional risks; drawn at top-right.
    """
    arr = tifffile.imread(str(tif_path))
    if arr.ndim != 3:
        raise ValueError(f"lihe expected (N,H,W), got {arr.shape}")

    n = arr.shape[0]
    if n == 4:
        max_proj = arr[0]
        dirs = [arr[1], arr[2], arr[3]]
        mode = "essential4"
    elif n >= 5:
        # min, max, d1, d2, d3
        max_proj = arr[1]
        dirs = [arr[2], arr[3], arr[4]]
        mode = "debug5"
    elif n == 3:
        max_proj = np.max(arr, axis=0)
        dirs = [arr[0], arr[1], arr[2]]
        mode = "dirs3"
    else:
        max_proj = np.max(arr, axis=0)
        dirs = [arr[i] for i in range(n)]
        mode = f"fallback{n}"

    # center-half XY crop (1024→512); peak coords are in cropped frame
    h0, w0 = int(max_proj.shape[-2]), int(max_proj.shape[-1])
    max_proj = _center_half_xy(max_proj)
    dirs = [_center_half_xy(d) for d in dirs]
    h1, w1 = int(max_proj.shape[-2]), int(max_proj.shape[-1])

    # min–max contrast for normal display
    u8 = _to_uint8(max_proj, mode="minmax")
    rgb = np.stack([u8, u8, u8], axis=-1)
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(20)
    font_s = _load_font(16)
    font_mcnr = _load_font(18)
    _draw_label(
        draw,
        (8, 8),
        f"{label} | mid {w1}x{h1} minmax",
        font,
    )

    # mark each direction peak with side arrows (do not cover the bright core)
    colors = [(255, 80, 80), (80, 220, 120), (80, 160, 255), (255, 160, 40)]
    # slight angle offsets so D1/D2/D3 arrows do not stack on top of each other
    angle_offsets = (-40.0, 0.0, 40.0, -20.0)
    for i, d in enumerate(dirs):
        xx, yy, val = _peak_xy(d)
        color = colors[i % len(colors)]
        tag = f"D{i + 1}"
        _draw_arrow_to_peak(
            draw,
            xx,
            yy,
            im.width,
            im.height,
            color,
            tag,
            font_s,
            gap=16,
            shaft=32,
            head=9,
            angle_deg=angle_offsets[i % len(angle_offsets)],
        )
        # bottom-left: peak intensity only (no XY coordinates)
        _draw_label(draw, (8, 36 + i * 22), f"D{i + 1} peak={val:.4g}", font_s)

    # top-right MCNR numbers
    if mcnr:
        def _fmt(key: str, risk_key: str | None = None) -> str:
            try:
                v = float(mcnr.get(key))
                s = f"{v:.3f}"
            except Exception:
                s = str(mcnr.get(key, ""))
            if risk_key and mcnr.get(risk_key):
                s = f"{s} {mcnr.get(risk_key)}"
            return s

        lines = [
            "MCNR",
            f"D1 {_fmt('D1', 'D1_risk')}",
            f"D2 {_fmt('D2', 'D2_risk')}",
            f"D3 {_fmt('D3', 'D3_risk')}",
        ]
        if mcnr.get("mean") is not None:
            overall = mcnr.get("overall") or mcnr.get("overall_risk") or ""
            try:
                mean_s = f"{float(mcnr['mean']):.3f}"
            except Exception:
                mean_s = str(mcnr.get("mean", ""))
            lines.append(f"mean {mean_s} {overall}".strip())
        _draw_label_top_right(draw, im.width, lines, font_mcnr)

    out = out or tif_path.with_name(tif_path.stem + f"__{ _safe(label) }__maxproj.png")
    im.save(out)
    print(out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chat PNG previews for FPJ / lihe")
    ap.add_argument("mode", choices=["fpj", "lihe"])
    ap.add_argument("tif", type=Path)
    ap.add_argument("--label", default="", help="data name, e.g. 13、35")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mcnr-d1", type=float, default=None)
    ap.add_argument("--mcnr-d2", type=float, default=None)
    ap.add_argument("--mcnr-d3", type=float, default=None)
    ap.add_argument("--mcnr-mean", type=float, default=None)
    ap.add_argument("--mcnr-overall", default="")
    args = ap.parse_args(list(argv) if argv is not None else None)

    tif: Path = args.tif
    if not tif.exists():
        raise SystemExit(f"missing: {tif}")
    label = args.label or tif.parent.parent.name if tif.parent else tif.stem

    if args.mode == "fpj":
        preview_fpj(tif, label=label, out=args.out)
    else:
        mcnr = None
        if any(v is not None for v in (args.mcnr_d1, args.mcnr_d2, args.mcnr_d3, args.mcnr_mean)):
            mcnr = {
                "D1": args.mcnr_d1,
                "D2": args.mcnr_d2,
                "D3": args.mcnr_d3,
                "mean": args.mcnr_mean,
                "overall": args.mcnr_overall,
            }
        preview_lihe(tif, label=label, out=args.out, mcnr=mcnr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
