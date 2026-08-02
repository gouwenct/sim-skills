#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SIMcheck_V2 — Python 实现的 3D-SIM 数据质控（对齐 Ball et al. 2015 / SIMcheck 1.3 核心检查）

相对 `SIMcheck.py`（仅轴序导出）：
  本脚本真正计算 CIP / TIV / MIV / MCNR / FPJ，以及重建侧 RIH-MMR / SAM-ZMV / FTR，
  并写出 Markdown 报告、CSV 指标与图像。

ImageJ 官方插件（可对照 / 启动）：
  D:\\Program Files (x86)\\fiji-win64\\FIJI-Plugins\\SIMcheck_-1.3.jar
  源码参考：https://github.com/MicronOxford/SIMcheck （GPL-3.0）

维序约定（与 SIMcheck 一致，默认 OMX raw）：
  CPZAT = Channel, Phase, Z, Angle, Time
  单通道时堆栈切片顺序为：for T for A for Z for P : frame

用法示例：
  $python -B SIMcheck_V2.py raw.tif --mode raw --n-angles 3 --n-phases 5 --n-z 8
  # 默认只跑 MCNR+FPJ；优先 CUDA（torch），无 GPU 再回退 CPU

Python：D:\\anaconda3\\envs\\3DMPSIM310\\python.exe
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

# ---------------------------------------------------------------------------
# Device: prefer GPU (torch CUDA), else CPU torch, else NumPy
# ---------------------------------------------------------------------------
_TORCH = None
_DEVICE = None
_DEVICE_NAME = "numpy-cpu"


def _init_device(prefer_gpu: bool = True) -> str:
    """Initialize global compute backend. Returns device name string."""
    global _TORCH, _DEVICE, _DEVICE_NAME
    try:
        import torch

        _TORCH = torch
        if prefer_gpu and torch.cuda.is_available():
            _DEVICE = torch.device("cuda")
            _DEVICE_NAME = f"cuda:{torch.cuda.get_device_name(0)}"
            # avoid first-call overhead later
            torch.zeros(1, device=_DEVICE)
        else:
            _DEVICE = torch.device("cpu")
            _DEVICE_NAME = "torch-cpu"
    except Exception:
        _TORCH = None
        _DEVICE = None
        _DEVICE_NAME = "numpy-cpu"
    return _DEVICE_NAME


def get_device_name() -> str:
    if _DEVICE_NAME == "numpy-cpu" and _TORCH is None:
        _init_device(True)
    return _DEVICE_NAME

# ---------------------------------------------------------------------------
# Defaults: local Fiji + SIMcheck jar
# ---------------------------------------------------------------------------
DEFAULT_FIJI = Path(r"D:\Program Files (x86)\fiji-win64\Fiji.app\ImageJ-win64.exe")
DEFAULT_SIMCHECK_JAR = Path(
    r"D:\Program Files (x86)\fiji-win64\FIJI-Plugins\SIMcheck_-1.3.jar"
)
DEFAULT_FIJI_APP = Path(r"D:\Program Files (x86)\fiji-win64\Fiji.app")

# Stat thresholds from SIMcheck 1.3 ResultSet / paper methods
THRESH = {
    "tiv_yes": 40.0,
    "tiv_maybe": 60.0,
    "mcnr_yes": 8.0,
    "mcnr_maybe": 4.0,
    "mmr_yes": 6.0,
    "mmr_maybe": 3.0,
    "zmv_yes": 0.3,
    "zmv_maybe": 1.0,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class StatItem:
    name: str
    value: float
    verdict: str  # YES / MAYBE / NO / NA / UNCALIBRATED
    note: str = ""


@dataclass
class CheckReport:
    title: str
    tla: str
    stats: List[StatItem] = field(default_factory=list)
    images: Dict[str, str] = field(default_factory=dict)
    info: List[str] = field(default_factory=list)


def verdict_tiv(v: float) -> str:
    if v <= THRESH["tiv_yes"]:
        return "YES"
    if v <= THRESH["tiv_maybe"]:
        return "MAYBE"
    return "NO"


def verdict_mcnr(v: float) -> str:
    if v >= THRESH["mcnr_yes"]:
        return "YES"
    if v >= THRESH["mcnr_maybe"]:
        return "MAYBE"
    return "NO"


def verdict_mmr(v: float) -> str:
    if v >= THRESH["mmr_yes"]:
        return "YES"
    if v >= THRESH["mmr_maybe"]:
        return "MAYBE"
    return "NO"


def verdict_zmv(v: float) -> str:
    if v < THRESH["zmv_yes"]:
        return "YES"
    if v < THRESH["zmv_maybe"]:
        return "MAYBE"
    return "NO"


def estim_wiener(mcnr: float) -> float:
    """SoftWoRx Wiener 经验式（SIMcheck Raw_ModContrast）：w = 0.17 / MCNR^2"""
    if mcnr <= 1e-6:
        return float("nan")
    return 0.17 / (mcnr * mcnr)


# ---------------------------------------------------------------------------
# Load / reshape
# ---------------------------------------------------------------------------
def load_stack(path: Path) -> np.ndarray:
    """Load TIFF as float32 stack with shape (N, Y, X) or (Z,Y,X)."""
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 4:
        # (T,Z,Y,X) or (C,Z,Y,X) -> collapse first dims
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    if arr.ndim != 3:
        raise ValueError(f"Expect 2D/3D/4D TIFF, got shape {arr.shape}")
    return arr


def reshape_raw_cpzat(
    stack: np.ndarray,
    n_phases: int,
    n_angles: int,
    n_z: Optional[int] = None,
    n_c: int = 1,
    n_t: int = 1,
) -> np.ndarray:
    """
    Reshape flat stack into (C, P, Z, A, T, Y, X) assuming OMX CPZAT plane order:
      slice index grows as: for t for a for z for p for c
    SIMcheck stack indexing (1-based) uses the same layout.
    """
    n, ny, nx = stack.shape
    npa = n_phases * n_angles
    if n_z is None:
        if n % (n_c * npa * n_t) != 0:
            raise ValueError(
                f"stack size {n} not divisible by C*P*A*T="
                f"{n_c * npa * n_t}"
            )
        n_z = n // (n_c * npa * n_t)
    expect = n_c * n_phases * n_z * n_angles * n_t
    if n != expect:
        raise ValueError(
            f"stack frames={n}, expected C*P*Z*A*T={expect} "
            f"(C={n_c},P={n_phases},Z={n_z},A={n_angles},T={n_t})"
        )
    # ImageJ HyperStack for SIMcheck often stores planes in order C fastest,
    # then P, Z, A, T — matches SIMcheck I1l.sliceList.
    # Our flat tifffile order is typically sequential: [c-fast] as written.
    # We assume planes are stored as CPZAT with C fastest (channel-interleaved).
    out = stack.reshape(n_t, n_angles, n_z, n_phases, n_c, ny, nx)
    # → (C, P, Z, A, T, Y, X)
    out = np.transpose(out, (4, 3, 2, 1, 0, 5, 6))
    return np.ascontiguousarray(out)


def reshape_raw_pzat(
    stack: np.ndarray,
    n_phases: int,
    n_angles: int,
    n_z: Optional[int] = None,
) -> np.ndarray:
    """
    Order: for angle for Z for phase (single channel).
    Output shape: (C=1, P, Z, A, T=1, Y, X)
    """
    n, ny, nx = stack.shape
    npa = n_phases * n_angles
    if n_z is None:
        if n % npa != 0:
            raise ValueError(f"frames {n} not divisible by P*A={npa}")
        n_z = n // npa
    expect = n_phases * n_z * n_angles
    if n != expect:
        raise ValueError(f"frames={n} expected P*Z*A={expect}")
    # for a for z for p → (A,Z,P,Y,X)
    tmp = stack.reshape(n_angles, n_z, n_phases, ny, nx)
    out = np.transpose(tmp, (2, 1, 0, 3, 4))  # P,Z,A,Y,X
    out = out[None, :, :, :, None, :, :]  # C,P,Z,A,T,Y,X
    return np.ascontiguousarray(out)


def reshape_raw_zap(
    stack: np.ndarray,
    n_phases: int,
    n_angles: int,
    n_z: Optional[int] = None,
) -> np.ndarray:
    """
    Order Z, A, P (legacy SIMcheck.py Transfer2NPD input):
      total = Z * A * P, reshape (Z,A,P,Y,X) then swap to (P,A,Z,...)
    Stored as for z for a for p OR for z for (a,p interleaved)?
    SIMcheck.py used: reshape(nz, Nangles, Nshifts) → (Z,A,P)
    """
    n, ny, nx = stack.shape
    npa = n_phases * n_angles
    if n_z is None:
        if n % npa != 0:
            raise ValueError(f"frames {n} not divisible by A*P={npa}")
        n_z = n // npa
    expect = n_z * n_angles * n_phases
    if n != expect:
        raise ValueError(f"frames={n} expected Z*A*P={expect}")
    # (Z, A, P, Y, X) as in SIMcheck.py
    tmp = stack.reshape(n_z, n_angles, n_phases, ny, nx)
    # → C,P,Z,A,T,Y,X
    out = np.transpose(tmp, (2, 0, 1, 3, 4))  # P,Z,A,Y,X
    out = out[None, :, :, :, None, :, :]
    return np.ascontiguousarray(out)


def load_raw(
    path: Path,
    order: str,
    n_phases: int,
    n_angles: int,
    n_z: Optional[int],
    n_c: int,
    n_t: int,
) -> np.ndarray:
    stack = load_stack(path)
    order = order.upper()
    if order == "CPZAT":
        return reshape_raw_cpzat(stack, n_phases, n_angles, n_z, n_c, n_t)
    if order == "PZAT":
        if n_c != 1 or n_t != 1:
            raise ValueError("PZAT loader currently supports C=1,T=1")
        return reshape_raw_pzat(stack, n_phases, n_angles, n_z)
    if order == "ZAP":
        if n_c != 1 or n_t != 1:
            raise ValueError("ZAP loader currently supports C=1,T=1")
        return reshape_raw_zap(stack, n_phases, n_angles, n_z)
    raise ValueError(f"Unknown order {order}; use CPZAT|PZAT|ZAP")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_out_dir_for_input(inp: Path) -> Path:
    """
    Put SIMcheck results under the sample's **Dr** directory when possible.

    Example input:
      .../新建文件夹 (9)/08、8/Dr/01-488_padding/xxx_0003.tif
    Output:
      .../新建文件夹 (9)/08、8/Dr/SIMcheck_V2/01-488_padding_t0003/

    If path has no ``Dr`` segment, fall back to ``<parent>/<stem>_SIMcheck_V2``.
    """
    import re

    p = inp.resolve()
    parts = list(p.parts)
    # Windows drive-safe: find index of 'Dr'
    dr_idx = None
    for i, part in enumerate(parts):
        if part == "Dr":
            dr_idx = i
            break
    if dr_idx is None:
        return p.parent / f"{p.stem}_SIMcheck_V2"

    dr_dir = Path(*parts[: dr_idx + 1])
    # subfolder under Dr (e.g. 01-488_padding / 01-488)
    sub = parts[dr_idx + 1] if len(parts) > dr_idx + 1 else "input"
    stem = p.stem
    m = re.search(r"_(\d{3,5})$", stem)
    if m:
        run_name = f"{sub}_t{m.group(1)}"
    else:
        # keep name short but unique
        safe = re.sub(r'[<>:"/\\|?*]', "_", stem)
        run_name = f"{sub}_{safe[:48]}"
    return dr_dir / "SIMcheck_V2" / run_name


def _save_tif(path: Path, arr: np.ndarray) -> None:
    tifffile.imwrite(
        str(path),
        np.asarray(arr),
        imagej=True,
        photometric="minisblack",
    )


def _save_fig(path: Path, fig) -> None:
    if fig is None:
        return
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)


def otsu_threshold(values: np.ndarray) -> float:
    """Otsu on flattened finite values."""
    v = values[np.isfinite(values)].ravel()
    if v.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(v, bins=256)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    inter = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(inter))
    return float(bin_mids[idx])


def anscombe(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.maximum(x, 0.0) + 3.0 / 8.0)


def mcnr_policy_for_phases(
    n_phases: int,
) -> Tuple[Tuple[int, ...], str, bool]:
    if n_phases == 3:
        return (1,), "background_modulation_std", False
    if n_phases == 5:
        return (1, 2), "highest_frequency_spatial_std", True
    if n_phases == 7:
        return (1, 2, 3), "background_modulation_std", False
    if n_phases >= 5:
        return (1, 2), "highest_frequency_spatial_std", False
    raise ValueError(f"MCNR is not defined for {n_phases} phases")


def _mcnr_volume_for_angle(
    data_paz: np.ndarray,
    harmonic_orders: Sequence[int] = (1, 2),
    noise_method: str = "background_modulation_std",
    zw: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute MCNR (Z,Y,X) for one angle.
    data_paz: (P, Z, Y, X) float32
    harmonic_orders: positive phase harmonics combined in quadrature
    Uses torch FFT on CUDA if available (batched by window length).
    """
    p, z, ny, nx = data_paz.shape
    orders = tuple(dict.fromkeys(int(order) for order in harmonic_orders))
    if not orders or min(orders) < 1 or max(orders) > (p - 1) // 2:
        raise ValueError(
            f"harmonic orders {orders} are invalid for {p} uniformly spaced phases"
        )
    if noise_method not in {
        "background_modulation_std",
        "highest_frequency_spatial_std",
    }:
        raise ValueError(f"unsupported MCNR noise method: {noise_method}")
    out = np.zeros((z, ny, nx), dtype=np.float32)
    noise_by_z = np.zeros(z, dtype=np.float32)

    # group z-indices by window length L = n_z_in_window * n_phases
    groups: Dict[int, List[Tuple[int, int, int]]] = {}
    for zi in range(z):
        z0 = max(0, zi - zw)
        z1 = min(z, zi + zw + 1)
        L = (z1 - z0) * p
        groups.setdefault(L, []).append((zi, z0, z1))

    torch = _TORCH
    device = _DEVICE

    for L, items in groups.items():
        # build batch (B, L, Y, X) — order: for zz for phase (matches original loop)
        B = len(items)
        batch = np.empty((B, L, ny, nx), dtype=np.float32)
        zis = []
        for bi, (zi, z0, z1) in enumerate(items):
            # (P, Zw, Y, X) -> (Zw, P, Y, X) -> (Zw*P, Y, X)
            chunk = data_paz[:, z0:z1, :, :]
            v = np.transpose(chunk, (1, 0, 2, 3)).reshape(L, ny, nx)
            batch[bi] = v
            zis.append(zi)
        mean_intensity = batch.mean(axis=1).reshape(B, -1)
        varying = np.ptp(batch, axis=1).reshape(B, -1) > 0
        min_valid_count = int(varying.sum(axis=1).min())
        if min_valid_count < 2:
            raise ValueError(
                "MCNR background noise requires at least two spatial pixels "
                "that vary across the local phase/Z sequence"
            )
        background_count = min(
            min_valid_count,
            max(16, int(np.ceil(min_valid_count * 0.2))),
        )
        background_indices = np.empty((B, background_count), dtype=np.int64)
        for bi in range(B):
            valid_indices = np.flatnonzero(varying[bi])
            selected = np.argpartition(
                mean_intensity[bi, valid_indices],
                background_count - 1,
            )[:background_count]
            background_indices[bi] = valid_indices[selected]

        if torch is not None and device is not None:
            t = torch.from_numpy(batch).to(device, non_blocking=True)
            t = 2.0 * torch.sqrt(torch.clamp(t, min=0.0) + 3.0 / 8.0)
            t = t - t.mean(dim=1, keepdim=True)
            F = torch.fft.fft(t, dim=1)
            amp = torch.abs(F)
            order_positions = [(L * order) // p for order in orders]
            modulation = torch.sqrt(
                torch.stack([amp[:, pos] ** 2 for pos in order_positions]).sum(dim=0)
            )
            if noise_method == "highest_frequency_spatial_std":
                noise_std = amp[:, L // 2].reshape(B, -1).std(
                    dim=1,
                    correction=0,
                )
            else:
                index = torch.from_numpy(background_indices).to(device)
                background = torch.gather(modulation.reshape(B, -1), 1, index)
                noise_std = background.std(dim=1, correction=0)
            noise_std = noise_std.clamp_min(1e-8)
            mcnr = modulation / noise_std.view(B, 1, 1)
            mcnr_np = mcnr.detach().float().cpu().numpy()
            noise_np = noise_std.detach().float().cpu().numpy()
            for bi, zi in enumerate(zis):
                out[zi] = mcnr_np[bi]
                noise_by_z[zi] = noise_np[bi]
        else:
            for bi, zi in enumerate(zis):
                v = batch[bi]
                v = anscombe(v)
                v = v - v.mean(axis=0, keepdims=True)
                F = np.fft.fft(v, axis=0)
                amp = np.abs(F)
                modulation = np.sqrt(
                    sum(amp[(L * order) // p] ** 2 for order in orders)
                )
                if noise_method == "highest_frequency_spatial_std":
                    noise_values = amp[L // 2]
                else:
                    noise_values = modulation.reshape(-1)[background_indices[bi]]
                noise_std = max(float(np.std(noise_values)), 1e-8)
                out[zi] = (modulation / noise_std).astype(np.float32)
                noise_by_z[zi] = noise_std
    return out, noise_by_z


def _fpj_power_batch(
    planes: np.ndarray,
    batch_size: int = 24,
) -> np.ndarray:
    """
    planes: (N, H, W) already windowed+padded square.
    Returns max-reduced log1p(|FFT|^2) over N → (H, W)
    """
    n, h, w = planes.shape
    torch = _TORCH
    device = _DEVICE
    acc = None
    if torch is not None and device is not None:
        for i0 in range(0, n, batch_size):
            chunk = planes[i0 : i0 + batch_size]
            t = torch.from_numpy(chunk).to(device, non_blocking=True)
            F = torch.fft.fft2(t)
            F = torch.fft.fftshift(F, dim=(-2, -1))
            power = torch.log1p(torch.abs(F) ** 2)
            # max over batch dim
            pmax = power.amax(dim=0)
            pmax_np = pmax.detach().float().cpu().numpy()
            acc = pmax_np if acc is None else np.maximum(acc, pmax_np)
        return acc.astype(np.float32)
    # numpy fallback
    for i in range(n):
        F = np.fft.fftshift(np.fft.fft2(planes[i]))
        power = np.log1p(np.abs(F) ** 2).astype(np.float32)
        acc = power if acc is None else np.maximum(acc, power)
    return acc


# ---------------------------------------------------------------------------
# RAW checks
# ---------------------------------------------------------------------------
def check_cip(
    raw: np.ndarray,
    out_dir: Path,
    zwin: int = 9,
) -> CheckReport:
    """
    Channel Intensity Profiles (CIP).
    raw: (C,P,Z,A,T,Y,X)
    Metrics mirror SIMcheck Raw_IntensityProfiles.
    """
    c, p, z, a, t, ny, nx = raw.shape
    report = CheckReport(
        title="Channel Intensity Profiles",
        tla="CIP",
        info=[
            "TIV: total intensity variation (%) over central 9-z window × all phases/angles.",
            "SIMcheck: TIV<=40 YES, <=60 MAYBE, else NO.",
        ],
    )
    # mean intensity per plane in order P,Z,A,T for each channel
    means = raw.mean(axis=(-1, -2))  # (C,P,Z,A,T)
    # plot series: for each c, flatten order P,Z,A,T
    series = np.transpose(means, (0, 1, 2, 3, 4)).reshape(c, -1)  # wrong order
    # correct: P fastest then Z, A, T → already C,P,Z,A,T → reshape C, P*Z*A*T with order P,Z,A,T
    series = means.reshape(c, p, z, a, t)
    series_flat = []
    for ci in range(c):
        flat = []
        for ti in range(t):
            for ai in range(a):
                for zi in range(z):
                    for pi in range(p):
                        flat.append(float(series[ci, pi, zi, ai, ti]))
        series_flat.append(np.asarray(flat, dtype=np.float64))
    series_flat = np.stack(series_flat, axis=0)

    # figures
    if plt is not None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        x = np.arange(1, series_flat.shape[1] + 1)
        colors = ["r", "g", "b", "k", "m", "c"]
        for ci in range(c):
            col = colors[ci % len(colors)]
            axes[0].plot(x, series_flat[ci], color=col, lw=0.8, label=f"C{ci+1}")
            mx = series_flat[ci].max()
            axes[1].plot(
                x,
                series_flat[ci] / mx if mx > 0 else series_flat[ci],
                color=col,
                lw=0.8,
                label=f"C{ci+1}",
            )
        axes[0].set_ylabel("Mean intensity")
        axes[1].set_ylabel("Normalized mean")
        axes[1].set_xlabel("Slices (order: phase, Z, angle, time)")
        axes[0].set_title("CIP absolute")
        axes[1].set_title("CIP relative")
        axes[0].legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        p_png = out_dir / "CIP_profiles.png"
        _save_fig(p_png, fig)
        report.images["profiles"] = str(p_png)

    z_first = max(0, z // 2 - zwin // 2)
    z_last = min(z, z_first + zwin)

    for ci in range(c):
        av = series_flat[ci]
        # (0) TIV over central Z window, first timepoint all angles/phases
        intens = []
        for ai in range(a):
            for zi in range(z_first, z_last):
                for pi in range(p):
                    # index in flat: (((t*a+a)*z+z)*p+p) with t=0
                    # flat order: for t for a for z for p
                    idx = (((0 * a + ai) * z + zi) * p + pi)
                    intens.append(av[idx])
        intens = np.asarray(intens, dtype=np.float64)
        if intens.size and intens.max() > 0:
            tiv = 100.0 * (intens.max() - intens.min()) / intens.max()
        else:
            tiv = float("nan")
        report.stats.append(
            StatItem(
                f"C{ci+1} total intensity variation (%)",
                float(tiv),
                verdict_tiv(tiv) if np.isfinite(tiv) else "NA",
            )
        )

        # (1) decay per angle via linear fit, average
        decays = []
        for ai in range(a):
            ya = []
            for zi in range(z):
                for pi in range(p):
                    idx = (((0 * a + ai) * z + zi) * p + pi)
                    ya.append(av[idx])
            ya = np.asarray(ya, dtype=np.float64)
            xa = np.arange(ya.size, dtype=np.float64)
            if ya.size >= 2 and np.std(xa) > 0:
                coef = np.polyfit(xa, ya, 1)
                # (slope * n) * -100 / intercept
                if abs(coef[1]) > 1e-12:
                    d = (coef[0] * ya.size * -100.0) / coef[1]
                else:
                    d = 0.0
            else:
                d = 0.0
            decays.append(max(0.0, float(d)))
        report.stats.append(
            StatItem(
                f"C{ci+1} estimated intensity decay (%)",
                float(np.mean(decays)),
                "NA",
            )
        )

        # (2) max angle mean difference (%)
        angle_means = []
        for ai in range(a):
            vals = []
            for zi in range(z):
                for pi in range(p):
                    idx = (((0 * a + ai) * z + zi) * p + pi)
                    vals.append(av[idx])
            angle_means.append(float(np.mean(vals)))
        am = np.asarray(angle_means)
        if am.max() > 0:
            largest = 0.0
            for i in range(a):
                for j in range(a):
                    largest = max(largest, abs(am[i] - am[j]))
            largest = 100.0 * largest / am.max()
        else:
            largest = float("nan")
        report.stats.append(
            StatItem(
                f"C{ci+1} maximum intensity difference between angles (%)",
                float(largest),
                "NA",
            )
        )

        # (3) relative intensity fluctuations (%)
        ranges = []
        for ai in range(a):
            for pi in range(p):
                zs = []
                for zi in range(z_first, z_last):
                    idx = (((0 * a + ai) * z + zi) * p + pi)
                    zs.append(av[idx])
                zs = np.asarray(zs)
                if zs.size and zs.max() > 0:
                    ranges.append((zs.max() - zs.min()) / zs.max())
        fluc = 100.0 * float(np.mean(ranges)) if ranges else float("nan")
        report.stats.append(
            StatItem(
                f"C{ci+1} relative intensity fluctuations (%)",
                float(fluc),
                "NA",
            )
        )

    return report


def check_miv(raw: np.ndarray, out_dir: Path) -> CheckReport:
    """
    Motion & Illumination Variation (MIV): phase-average per angle,
    normalize angles, CMY composite. Only A==3 fully colored.
    raw: (C,P,Z,A,T,Y,X)
    """
    c, p, z, a, t, ny, nx = raw.shape
    report = CheckReport(
        title="Motion & Illumination Variation",
        tla="MIV",
        info=[
            "Gray/white = consistent across angles; color patches = drift/illumination differences.",
            "Requires 3 angles for full CMY mapping (SIMcheck).",
        ],
    )
    # phase average → (C,Z,A,T,Y,X)
    phase_avg = raw.mean(axis=1)
    # take first time
    vol = phase_avg[:, :, :, 0]  # C,Z,A,Y,X

    # normalize angles to max angle intensity per channel
    total = vol.mean(axis=(1, 3, 4))  # C,A
    for ci in range(c):
        mx = total[ci].max()
        if mx <= 0:
            continue
        for ai in range(a):
            if total[ci, ai] > 0:
                vol[ci, :, ai] *= mx / total[ci, ai]

    # central z RGB composite per channel
    zc = z // 2
    for ci in range(c):
        if a >= 3:
            a1, a2, a3 = vol[ci, zc, 0], vol[ci, zc, 1], vol[ci, zc, 2]
            half = 0.5
            r = half * (a2 + a3)
            g = half * (a1 + a3)
            b = half * (a1 + a2)

            def to_u8(x):
                x = x.astype(np.float64)
                lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
                if hi <= lo:
                    hi = lo + 1
                y = np.clip((x - lo) / (hi - lo), 0, 1)
                return (y * 255).astype(np.uint8)

            rgb = np.stack([to_u8(r), to_u8(g), to_u8(b)], axis=-1)
            path = out_dir / f"MIV_C{ci+1}_z{zc+1}.png"
            if plt is not None:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(rgb)
                ax.set_title(f"MIV C{ci+1} (A1 cyan, A2 magenta, A3 yellow)")
                ax.axis("off")
                _save_fig(path, fig)
            else:
                tifffile.imwrite(str(path.with_suffix(".tif")), rgb)
            report.images[f"C{ci+1}"] = str(path)

            # crude colorfulness metric: chroma mean (higher → more angle mismatch)
            rf, gf, bf = r.astype(np.float64), g.astype(np.float64), b.astype(np.float64)
            mx = np.maximum(np.maximum(rf, gf), bf) + 1e-6
            chroma = (np.abs(rf - gf) + np.abs(gf - bf) + np.abs(bf - rf)) / mx
            report.stats.append(
                StatItem(
                    f"C{ci+1} MIV mean chroma (proxy)",
                    float(np.mean(chroma)),
                    "NA",
                    note="not official SIMcheck statistic; lower is more uniform",
                )
            )
        else:
            report.info.append(f"C{ci+1}: angles={a} < 3, skip CMY composite")
    return report


def _feature_mcnr_mean(plane: np.ndarray) -> float:
    """Otsu feature mean on a single MCNR plane (mid-z)."""
    thr = otsu_threshold(plane)
    mask = plane >= thr
    return float(plane[mask].mean()) if mask.any() else float(plane.mean())


def _save_mcnr_preview(
    plane: np.ndarray,
    path_png: Path,
    title: str,
    vmax: float = 24.0,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(plane, cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")
    _save_fig(path_png, fig)


def check_mcnr(
    raw: np.ndarray,
    out_dir: Path,
    harmonic_orders: Sequence[int] = (1, 2),
    noise_method: str = "highest_frequency_spatial_std",
    calibrated: bool = True,
    zw: int = 1,
) -> CheckReport:
    """
    Modulation Contrast-to-Noise Ratio (MCNR), **per illumination angle**.

    Aligned with SIMcheck Raw_ModContrast:
      for each angle independently:
        Anscombe → DFT along phase*(z-window) → harmonic quadrature/noise_std
      then average angle maps for an overall score (SIMcheck default).

    raw: (C,P,Z,A,T,Y,X)

    Image outputs (lean):
      - MCNR_C{c}_z{mid}_angles_stack.tif  — mid-z plane per data angle (A,Y,X)
      - MCNR_C{c}_per_angle_compare_z{mid}.png — multi-panel A1..A + avg
    Metrics (CSV/MD) still include per-angle / avg feature MCNR and Wiener.
    """
    c, p, z, a, t, ny, nx = raw.shape
    report = CheckReport(
        title="Modulation Contrast",
        tla="MCN",
        info=[
            "MCNR is computed **per direction (angle)** on that angle's phase series, "
            "then averaged for the overall score (SIMcheck default).",
            f"MCNR harmonics: {','.join(str(order) for order in harmonic_orders)}.",
            (
                "Noise = highest-frequency amplitude std over XY."
                if noise_method == "highest_frequency_spatial_std"
                else "Noise = modulation-amplitude std in the lowest-intensity 20% "
                "of varying pixels; constant padding is excluded."
            ),
            (
                "Five-phase calibration: feature MCNR >=8 YES, >=4 MAYBE, else NO."
                if calibrated
                else "Extended MCNR is UNCALIBRATED; five-phase thresholds are not applied."
            ),
            "Wiener estimate (SoftWoRx): w = 0.17 / MCNR^2",
            "If one angle is much weaker than others, reconstruction may be directionally biased.",
            "Images written: mid-z per-angle stack TIFF + one compare PNG only.",
        ],
    )
    # use first time only
    data = raw[:, :, :, :, 0]  # C,P,Z,A,Y,X
    report.info.append(f"Compute device: {get_device_name()}")

    for ci in range(c):
        maps_a: List[np.ndarray] = []
        feat_per_angle: List[float] = []
        zc = z // 2

        for ai in range(a):
            # data for this angle: (P,Z,Y,X)
            data_paz = np.ascontiguousarray(data[ci, :, :, ai, :, :])
            mz, noise_by_z = _mcnr_volume_for_angle(
                data_paz,
                harmonic_orders=harmonic_orders,
                noise_method=noise_method,
                zw=zw,
            )
            maps_a.append(mz)

            feat_a = _feature_mcnr_mean(mz[zc])
            feat_per_angle.append(feat_a)
            verdict = verdict_mcnr(feat_a) if calibrated else "UNCALIBRATED"
            report.stats.append(
                StatItem(
                    f"C{ci+1} D{ai+1}(dataAngleIdx={ai}) feature MCNR",
                    feat_a,
                    verdict,
                    note="data angle index = recon direction index (not geometric sort)",
                )
            )
            if not calibrated:
                report.stats.append(
                    StatItem(
                        f"C{ci+1} D{ai+1}(dataAngleIdx={ai}) background noise std",
                        float(noise_by_z[zc]),
                        "NA",
                        note=f"middle-Z; method={noise_method}",
                    )
                )
            if calibrated:
                report.stats.append(
                    StatItem(
                        f"C{ci+1} D{ai+1}(dataAngleIdx={ai}) estimated Wiener",
                        estim_wiener(feat_a),
                        "NA",
                        note="from this data direction's MCNR",
                    )
                )

        # mid-z planes of each data direction → stack (A, Y, X) only
        mid_stack = np.stack([m[zc] for m in maps_a], axis=0).astype(np.float32)
        path_mid = out_dir / f"MCNR_C{ci+1}_z{zc+1}_angles_stack.tif"
        tifffile.imwrite(
            str(path_mid),
            mid_stack,
            imagej=True,
            photometric="minisblack",
            metadata={
                "axes": "TYX",
                "Labels": [f"D{i+1}_dataAngleIdx{i}_z{zc+1}" for i in range(a)],
            },
        )
        report.images[f"C{ci+1}_z{zc+1}_angles_stack"] = str(path_mid)

        # angle-average map (for stats + compare PNG; not written as full-volume TIFF)
        m_avg = np.mean(np.stack(maps_a, axis=0), axis=0)
        feat_avg = _feature_mcnr_mean(m_avg[zc])
        # also mean of per-angle feature scores (can differ slightly from map-average Otsu)
        feat_mean_angles = float(np.mean(feat_per_angle)) if feat_per_angle else float("nan")
        feat_min = float(np.min(feat_per_angle)) if feat_per_angle else float("nan")
        feat_max = float(np.max(feat_per_angle)) if feat_per_angle else float("nan")
        imbalance = (
            feat_max / feat_min if (np.isfinite(feat_min) and feat_min > 1e-6) else float("nan")
        )

        report.stats.append(
            StatItem(
                f"C{ci+1} average feature MCNR (angle-avg map)",
                feat_avg,
                verdict_mcnr(feat_avg) if calibrated else "UNCALIBRATED",
                note="SIMcheck-style overall score on angle-averaged MCNR map",
            )
        )
        report.stats.append(
            StatItem(
                f"C{ci+1} mean of per-angle feature MCNR",
                feat_mean_angles,
                verdict_mcnr(feat_mean_angles) if calibrated else "UNCALIBRATED",
                note="mean of angle1..N feature MCNR values",
            )
        )
        report.stats.append(
            StatItem(
                f"C{ci+1} min per-angle feature MCNR",
                feat_min,
                verdict_mcnr(feat_min) if calibrated else "UNCALIBRATED",
                note="weakest direction",
            )
        )
        report.stats.append(
            StatItem(
                f"C{ci+1} max per-angle feature MCNR",
                feat_max,
                verdict_mcnr(feat_max) if calibrated else "UNCALIBRATED",
                note="strongest direction",
            )
        )
        report.stats.append(
            StatItem(
                f"C{ci+1} max/min angle MCNR ratio",
                imbalance,
                "NA",
                note=">>1 means large directional imbalance",
            )
        )
        if calibrated:
            report.stats.append(
                StatItem(
                    f"C{ci+1} estimated Wiener filter optimum",
                    estim_wiener(feat_avg),
                    "NA",
                    note="from angle-avg map MCNR",
                )
            )

        # multi-panel comparison figure (only PNG kept for MCNR)
        if plt is not None and a >= 1:
            ncols = a + 1
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3.6))
            if ncols == 1:
                axes = [axes]
            vmax = 24.0
            for ai in range(a):
                im = axes[ai].imshow(maps_a[ai][zc], cmap="magma", vmin=0, vmax=vmax)
                axes[ai].set_title(f"A{ai+1} MCNR={feat_per_angle[ai]:.2f}")
                axes[ai].axis("off")
            im = axes[-1].imshow(m_avg[zc], cmap="magma", vmin=0, vmax=vmax)
            axes[-1].set_title(f"avg MCNR={feat_avg:.2f}")
            axes[-1].axis("off")
            fig.colorbar(im, ax=axes, fraction=0.02)
            fig.suptitle(f"MCNR per angle — C{ci+1} z={zc+1}", fontsize=11)
            p_cmp = out_dir / f"MCNR_C{ci+1}_per_angle_compare_z{zc+1}.png"
            _save_fig(p_cmp, fig)
            report.images[f"C{ci+1}_compare"] = str(p_cmp)

    return report


def _next_pow2(n: int) -> int:
    """Smallest power of 2 >= n (ImageJ FHT pad size convention)."""
    s = 2
    while s < int(n):
        s *= 2
    return s


def _pad_square_fht(plane: np.ndarray) -> np.ndarray:
    """
    Pad to square of side next power-of-2 of max(H,W), insert at (0,0).
    Matches SIMcheck FFT2D.pad / ImageJ FFT.java so FPJ is square like Fiji.
    """
    plane = np.asarray(plane, dtype=np.float32)
    ny, nx = plane.shape
    pad = _next_pow2(max(ny, nx))
    out = np.zeros((pad, pad), dtype=np.float32)
    out[:ny, :nx] = plane
    return out


def _to_u8_gray01(img: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    g = (img - lo) / (hi - lo)
    return (np.clip(g, 0, 1) * 255.0).astype(np.uint8)


def _save_square_rgb_png(rgb: np.ndarray, path: Path) -> None:
    """Save HxWx3 uint8 as PNG; H must equal W for true square output."""
    from PIL import Image

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expect HxWx3, got {rgb.shape}")
    if rgb.shape[0] != rgb.shape[1]:
        raise ValueError(f"PNG must be square, got {rgb.shape[0]}x{rgb.shape[1]}")
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(str(path))


def _draw_fpj_markers_on_square(
    gray_u8: np.ndarray,
    peak_records: List[Dict[str, Any]],
    cy: float,
    cx: float,
    only_angle: Optional[int] = None,
) -> np.ndarray:
    """
    Burn direction markers onto a square gray image → RGB uint8, same H=W.
    only_angle: if set (1-based), only draw that data direction's peaks.
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w = gray_u8.shape
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # center cross
    r0 = 6
    draw.line((cx - r0, cy, cx + r0, cy), fill=(255, 255, 255), width=1)
    draw.line((cx, cy - r0, cx, cy + r0), fill=(255, 255, 255), width=1)

    color_map = {
        "A1": (255, 68, 68),
        "A2": (51, 204, 102),
        "A3": (51, 153, 255),
        "A4": (255, 204, 0),
        "A5": (204, 102, 255),
    }
    for rec in peak_records:
        if only_angle is not None and rec["angle_idx"] != only_angle:
            continue
        col = color_map.get(rec["name"], (255, 255, 0))
        x, y = float(rec["x"]), float(rec["y"])
        rad = 7 if rec["order"] == "1st" else 5
        # circle
        draw.ellipse((x - rad, y - rad, x + rad, y + rad), outline=col, width=2)
        # line to center
        draw.line((cx, cy, x, y), fill=col, width=1)
        # label: D1-1st  (data direction index, not geometric angle)
        label = f"D{rec['angle_idx']}-{rec['order']}"
        tx, ty = x + 8, y - 10
        draw.text((tx, ty), label, fill=col, font=font)

    # legend strip top-left (inside square, small)
    legend_y = 4
    for ai in sorted({r["angle_idx"] for r in peak_records}):
        name = f"A{ai}"
        col = color_map.get(name, (255, 255, 255))
        text = f"D{ai}=data angle index {ai-1} (recon dir {ai-1})"
        draw.rectangle((4, legend_y, 12, legend_y + 8), outline=col, fill=col)
        draw.text((16, legend_y - 1), text, fill=col, font=font)
        legend_y += 12

    return np.asarray(im)


def _edge_gauss_window(ny: int, nx: int, win_frac: float = 0.06) -> np.ndarray:
    """
    SIMcheck-like edge soft window: hard-zero border band then Gaussian blur
    approximation via smooth ramp (pure numpy, no extra deps).
    win_frac is fraction of width/height for the border band.
    """
    winx = max(1, int(win_frac * nx))
    winy = max(1, int(win_frac * ny))
    yy, xx = np.mgrid[0:ny, 0:nx]
    # distance into valid interior (0 at outer edge of keep-region)
    dx = np.minimum(xx, nx - 1 - xx).astype(np.float32)
    dy = np.minimum(yy, ny - 1 - yy).astype(np.float32)
    # inside band width: weight ramps 0→1; interior = 1
    wx = np.clip(dx / float(winx), 0.0, 1.0)
    wy = np.clip(dy / float(winy), 0.0, 1.0)
    # smoothstep for softer edge (similar to blurred rect mask)
    def smooth(u: np.ndarray) -> np.ndarray:
        return u * u * (3.0 - 2.0 * u)

    return (smooth(wx) * smooth(wy)).astype(np.float32)


def _fpj_center_mask_radius(n_side: int) -> float:
    """SIMcheck FPJ central mask radius (diameter ≈ 0.125 * width)."""
    mask_r = 0.0625 * n_side
    if n_side % 2 == 0:
        mask_r += 0.5
    return float(mask_r)


def _find_sim_order_peaks(
    power: np.ndarray,
    n_pair: int = 2,
    min_r_frac: float = 0.08,
    max_r_frac: float = 0.45,
    min_sep: int = 8,
) -> List[Tuple[float, float, float, str]]:
    """
    Find bright SIM spots on one angle's FPJ map (center already useful for masking).

    Returns list of (y, x, radius_from_center, order_label) sorted by strength.
    For each radial shell we pick the strongest peak and its opposite partner,
    labeling closer shell as 1st order and farther as 2nd when n_pair>=2.
    """
    n = power.shape[0]
    cy = cx = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_min = min_r_frac * n
    r_max = max_r_frac * n
    mask = (r >= r_min) & (r <= r_max)

    work = power.astype(np.float64).copy()
    work[~mask] = -np.inf

    # local maxima via 5x5 neighborhood
    try:
        from scipy.ndimage import maximum_filter

        local_max = maximum_filter(work, size=5, mode="nearest") == work
        local_max &= np.isfinite(work)
    except Exception:
        local_max = np.zeros_like(work, dtype=bool)
        for y in range(2, n - 2):
            for x in range(2, n - 2):
                if not np.isfinite(work[y, x]):
                    continue
                patch = work[y - 2 : y + 3, x - 2 : x + 3]
                if work[y, x] >= patch.max():
                    local_max[y, x] = True

    ys, xs = np.where(local_max)
    if ys.size == 0:
        return []

    vals = work[ys, xs]
    order = np.argsort(-vals)
    candidates: List[Tuple[int, int, float, float]] = []  # y,x,r,val
    for k in order:
        y, x = int(ys[k]), int(xs[k])
        rr = float(r[y, x])
        vv = float(vals[k])
        # suppress near-duplicates
        if any((y - y0) ** 2 + (x - x0) ** 2 < min_sep**2 for y0, x0, _, _ in candidates):
            continue
        candidates.append((y, x, rr, vv))
        if len(candidates) >= 24:
            break

    # Cluster by radius into order shells (1st ~ smaller r, 2nd ~ larger)
    # Greedy: take strongest peak, pair with opposite hemisphere partner, assign order by r
    used = set()
    shells: List[Tuple[float, List[Tuple[float, float, float]]]] = []  # mean_r, peaks

    for i, (y, x, rr, vv) in enumerate(candidates):
        if i in used:
            continue
        # opposite partner: similar radius, roughly opposite direction
        best_j = None
        best_score = -1.0
        for j, (y2, x2, rr2, vv2) in enumerate(candidates):
            if j == i or j in used:
                continue
            if abs(rr2 - rr) > 0.12 * n:
                continue
            # angle between vectors from center
            v1 = np.array([y - cy, x - cx])
            v2 = np.array([y2 - cy, x2 - cx])
            n1 = np.linalg.norm(v1) + 1e-9
            n2 = np.linalg.norm(v2) + 1e-9
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            if cosang > -0.5:  # want roughly opposite
                continue
            score = vv + vv2 - abs(rr - rr2)
            if score > best_score:
                best_score = score
                best_j = j
        peaks = [(float(y), float(x), rr)]
        used.add(i)
        if best_j is not None:
            y2, x2, rr2, _ = candidates[best_j]
            peaks.append((float(y2), float(x2), rr2))
            used.add(best_j)
        mean_r = float(np.mean([pk[2] for pk in peaks]))
        shells.append((mean_r, peaks))
        if len(shells) >= n_pair:
            break

    shells.sort(key=lambda s: s[0])  # smaller r = 1st order typically
    out: List[Tuple[float, float, float, str]] = []
    for oi, (mean_r, peaks) in enumerate(shells):
        label = "1st" if oi == 0 else ("2nd" if oi == 1 else f"o{oi+1}")
        for y, x, rr in peaks:
            out.append((y, x, rr, label))
    return out


def check_fpj(
    raw: np.ndarray,
    out_dir: Path,
    win_frac: float = 0.06,
    peak_orders: int = 2,
) -> CheckReport:
    """
    Raw Fourier Projection (FPJ): edge window → pad to square (next power-of-2)
    → 2D FFT per plane → max-project log|FFT|^2 over phases, Z, angles (first T).

    Square output matches Fiji SIMcheck / ImageJ FHT (e.g. 152×408 → 512×512).

    Image output (lean, one multi-page TIFF per channel):
      FPJ_C{c}.tif page0 = max over all angles; page1..A = per-angle maps.
    Peak metrics still go to stats/CSV.
    """
    c, p, z, a, t, ny, nx = raw.shape
    pad = _next_pow2(max(ny, nx))
    report = CheckReport(
        title="Raw Fourier Projection",
        tla="FPJ",
        info=[
            "Maximum intensity projection of log(amplitude^2) 2D FFT stack.",
            "Each slice is edge-windowed then zero-padded to a square power-of-2 "
            f"({ny}x{nx} -> {pad}x{pad}), matching Fiji SIMcheck / ImageJ FHT.",
            "Central disk masked (~1/8 of width).",
            (
                f"Automatically label {peak_orders} radial order shell(s)."
                if peak_orders
                else "Peak-order labeling disabled; interpret the FPJ visually."
            ),
            "Direction indexing follows **data order only** (not geometric FFT angle sort): "
            "with order=ZAP, stack is reshape(Z, Angle, Phase); "
            "D1/A1 = angle index 0, D2/A2 = index 1, D3/A3 = index 2 "
            "(same as recon Transfer2NPD -> reshape(3,Nshift,nz) direction 0/1/2).",
            "TIFF pages: [0]=max_all_angles, [1..A]=per data angle D1..DA.",
            "Sparse samples may lack clear spots even when data are OK.",
        ],
    )
    win = _edge_gauss_window(ny, nx, win_frac)
    report.info.append(f"Compute device: {get_device_name()}")

    for ci in range(c):
        per_angle: List[np.ndarray] = []
        count = 0
        # Batch all P×Z planes per angle → one GPU/CPU FFT batch (much faster)
        for ai in range(a):
            planes = []
            for zi in range(z):
                for pi in range(p):
                    plane = raw[ci, pi, zi, ai, 0] * win
                    planes.append(_pad_square_fht(plane))
            planes_np = np.stack(planes, axis=0).astype(np.float32, copy=False)
            acc_a = _fpj_power_batch(planes_np, batch_size=32)
            count += planes_np.shape[0]
            per_angle.append(acc_a)

        acc = per_angle[0].copy()
        for ai in range(1, a):
            acc = np.maximum(acc, per_angle[ai])
        n_side = acc.shape[0]
        yy, xx = np.mgrid[0:n_side, 0:n_side]
        cy = cx = (n_side - 1) / 2.0
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        mask_r = _fpj_center_mask_radius(n_side)

        def apply_center_mask(img: np.ndarray) -> np.ndarray:
            out = img.copy()
            out[r < mask_r] = float(out.min())
            return out

        acc = apply_center_mask(acc)
        per_angle = [apply_center_mask(m) for m in per_angle]
        stack_a = np.stack(per_angle, axis=0).astype(np.float32)

        # one multi-page TIFF: page0 = max-all-angles, then per-angle maps
        combined = np.concatenate(
            [acc.astype(np.float32)[None, ...], stack_a],
            axis=0,
        )
        labels = ["max_all_angles"] + [
            f"D{ai+1}_dataAngleIdx{ai}" for ai in range(a)
        ]
        path = out_dir / f"FPJ_C{ci+1}.tif"
        tifffile.imwrite(
            str(path),
            combined,
            imagej=True,
            photometric="minisblack",
            metadata={"axes": "TYX", "Labels": labels},
        )
        report.images[f"C{ci+1}"] = str(path)

        # detect peaks per **data** angle index (stats only; no PNG)
        peak_n = 0
        for ai, mapa in enumerate(per_angle):
            if a > 1:
                others = np.max(np.delete(stack_a, ai, axis=0), axis=0)
                residual = mapa - others
                residual = residual - float(np.median(residual))
                residual = np.maximum(residual, 0.0)
                detect_map = 0.65 * residual + 0.35 * (mapa - float(np.median(mapa)))
            else:
                detect_map = mapa
            peaks = (
                _find_sim_order_peaks(detect_map, n_pair=peak_orders)
                if peak_orders
                else []
            )
            for y, x, rr, order_lab in peaks:
                peak_n += 1
                ang = float(np.degrees(np.arctan2(-(y - cy), (x - cx))))
                report.stats.append(
                    StatItem(
                        f"C{ci+1} D{ai+1}(dataAngleIdx={ai}) {order_lab} peak r(px)",
                        float(rr),
                        "NA",
                        note=(
                            f"pos=({x:.1f},{y:.1f}), FourierOrient≈{ang:.1f}deg; "
                            f"label follows data angle order not orientation sort"
                        ),
                    )
                )

        report.stats.append(
            StatItem(f"C{ci+1} FPJ planes used", float(count), "NA")
        )
        report.stats.append(
            StatItem(
                f"C{ci+1} FPJ size (px)",
                float(n_side),
                "NA",
                note=f"square pad from {ny}x{nx}; TIFF is {1+a}x{n_side}x{n_side}",
            )
        )
        report.stats.append(
            StatItem(
                f"C{ci+1} FPJ peaks labeled",
                float(peak_n),
                "NA",
                note="direction labels follow acquisition order",
            )
        )
    return report


def util_pseudo_widefield(raw: np.ndarray, out_dir: Path) -> CheckReport:
    """Average all phases and angles → pseudo-widefield (C,Z,T,Y,X)."""
    report = CheckReport(
        title="Raw SI to Pseudo-Widefield",
        tla="PWF",
        info=["Average over phases and angles for comparison with reconstruction."],
    )
    # (C,Z,T,Y,X)
    pwf = raw.mean(axis=(1, 3))
    path = out_dir / "PWF.tif"
    # save as (C*T*Z, Y, X) or hyperstack-like Z for C0 T0
    _save_tif(path, pwf[0, :, 0])
    report.images["pwf"] = str(path)
    report.stats.append(StatItem("PWF z-slices", float(pwf.shape[1]), "NA"))
    return report


# ---------------------------------------------------------------------------
# RECON checks
# ---------------------------------------------------------------------------
def check_rih(recon: np.ndarray, out_dir: Path, percentile: float = 0.0001) -> CheckReport:
    """
    Reconstructed Intensity Histogram / MMR.
    recon: (Z,Y,X) or (C,Z,Y,X)
    """
    if recon.ndim == 3:
        recon = recon[None, ...]
    c = recon.shape[0]
    report = CheckReport(
        title="Reconstructed Intensity Histogram",
        tla="RIH",
        info=[
            "MMR = (Max* - Mode) / |Min* - Mode| over extreme histogram tails.",
            "SIMcheck: MMR>=6 YES, >=3 MAYBE, else NO. Needs unclipped 32-bit data.",
        ],
    )
    for ci in range(c):
        vol = recon[ci].astype(np.float64)
        flat = vol.ravel()
        mode = float(np.median(flat))  # robust proxy; SIMcheck uses hist mode
        # histogram mode
        hist, edges = np.histogram(flat, bins=512)
        mode = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))
        n = flat.size
        n_tail = max(100, int(np.ceil(n * percentile / 100.0 * 100)))  # paper 0.0001 fraction
        # paper: percentile is 0.0001 as fraction of pixels at each extreme (0.01%)
        n_tail = max(100, int(np.ceil(n * 0.0001)))
        sorted_v = np.sort(flat)
        neg = sorted_v[:n_tail]
        pos = sorted_v[-n_tail:]
        min_star = float(neg.mean())
        max_star = float(pos.mean())
        denom = abs(min_star - mode)
        mmr = (max_star - mode) / denom if denom > 1e-12 else float("nan")
        report.stats.append(
            StatItem(f"C{ci+1} max-to-min intensity ratio", float(mmr), verdict_mmr(mmr))
        )
        report.stats.append(
            StatItem(f"C{ci+1} histogram mode", mode, "NA")
        )
        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.hist(flat, bins=200, color="k", alpha=0.7, density=True, label="linear")
            # log density hard — show counts on log y
            ax.set_yscale("log")
            ax.axvline(mode, color="r", ls="--", label="mode")
            ax.set_title(f"RIH C{ci+1}  MMR={mmr:.2f}")
            ax.legend(fontsize=8)
            p_png = out_dir / f"RIH_C{ci+1}.png"
            _save_fig(p_png, fig)
            report.images[f"C{ci+1}"] = str(p_png)
    return report


def check_sam(recon: np.ndarray, out_dir: Path) -> CheckReport:
    """Spherical Aberration Mismatch / ZMV."""
    if recon.ndim == 3:
        recon = recon[None, ...]
    c, z, ny, nx = recon.shape
    report = CheckReport(
        title="Spherical Aberration Mismatch",
        tla="SAM",
        info=[
            "ZMV = std(z-minima) / mean(feature intensity). High ZMV → OTF/sample RI mismatch.",
            "SIMcheck: ZMV<0.3 YES, <1.0 MAYBE, else NO. Best on beads; needs unclipped data.",
        ],
    )
    for ci in range(c):
        vol = recon[ci]
        zmin = vol.reshape(z, -1).min(axis=1)
        thr = otsu_threshold(vol)
        feat_means = []
        for zi in range(z):
            plane = vol[zi]
            m = plane >= thr
            feat_means.append(float(plane[m].mean()) if m.any() else float(plane.mean()))
        feat_means = np.asarray(feat_means, dtype=np.float64)
        zmv = float(np.std(zmin) / (np.mean(feat_means) + 1e-12))
        report.stats.append(
            StatItem(f"C{ci+1} Z-minimum variation", zmv, verdict_zmv(zmv))
        )
        if plt is not None:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x = np.arange(1, z + 1)
            ax.plot(x, zmin, "k-", lw=1.5, label="z-min")
            ax.plot(x, feat_means, color="0.6", lw=1.5, label="feature mean")
            ax.set_xlabel("Z")
            ax.set_ylabel("Intensity")
            ax.set_title(f"SAM C{ci+1}  ZMV={zmv:.3f}")
            ax.legend(fontsize=8)
            p_png = out_dir / f"SAM_C{ci+1}.png"
            _save_fig(p_png, fig)
            report.images[f"C{ci+1}"] = str(p_png)
    return report


def check_ftr(recon: np.ndarray, out_dir: Path) -> CheckReport:
    """Lateral FFT of mid-z + radial profile (FTR approximation)."""
    if recon.ndim == 3:
        recon = recon[None, ...]
    c, z, ny, nx = recon.shape
    report = CheckReport(
        title="Reconstructed Fourier Plots (lateral + radial)",
        tla="FTR",
        info=[
            "Mid-z lateral FFT (gamma=0.2 display) and radial average profile.",
            "Inflection of radial profile approximates effective resolution support.",
        ],
    )
    for ci in range(c):
        plane = recon[ci, z // 2].astype(np.float64)
        mode = float(np.median(plane))
        plane = plane - mode  # rough noise cut
        plane = np.clip(plane, 0, None)
        # window
        yy, xx = np.mgrid[0:ny, 0:nx]
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        win = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.06 * max(ny, nx)) ** 2))
        F = np.fft.fftshift(np.fft.fft2(plane * win))
        amp = np.abs(F)
        disp = np.power(amp / (amp.max() + 1e-12), 0.2)
        # radial profile
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = int(min(ny, nx) // 2)
        radial = np.zeros(rmax, dtype=np.float64)
        for ri in range(rmax):
            m = (r >= ri) & (r < ri + 1)
            if m.any():
                radial[ri] = amp[m].mean()
        path = out_dir / f"FTL_C{ci+1}.tif"
        _save_tif(path, disp.astype(np.float32))
        report.images[f"C{ci+1}_ftl"] = str(path)
        if plt is not None:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            axes[0].imshow(disp, cmap="magma")
            axes[0].set_title(f"FTL C{ci+1}")
            axes[0].axis("off")
            axes[1].plot(radial, "k-")
            axes[1].set_xlabel("radius (px)")
            axes[1].set_ylabel("mean |FFT|")
            axes[1].set_title("FTR")
            fig.tight_layout()
            p_png = out_dir / f"FTR_C{ci+1}.png"
            _save_fig(p_png, fig)
            report.images[f"C{ci+1}_png"] = str(p_png)
        # half-max radius as crude resolution proxy in px
        if radial.max() > 0:
            half = radial.max() * 0.5
            idx = np.where(radial < half)[0]
            r_half = float(idx[0]) if idx.size else float(rmax)
        else:
            r_half = float("nan")
        report.stats.append(
            StatItem(f"C{ci+1} FTR half-max radius (px)", r_half, "NA")
        )
    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_markdown(
    out_path: Path,
    source: Path,
    mode: str,
    reports: Sequence[CheckReport],
    meta: Dict[str, Any],
) -> None:
    lines = [
        f"# SIMcheck_V2 report",
        "",
        f"- source: `{source}`",
        f"- mode: `{mode}`",
        f"- meta: `{json.dumps(meta, ensure_ascii=False)}`",
        "",
        "Implementation notes: pure-Python port of Ball et al. 2015 / SIMcheck 1.3 core metrics.",
        f"Official plugin jar (optional): `{DEFAULT_SIMCHECK_JAR}`",
        "",
    ]
    for r in reports:
        lines.append(f"## {r.title} (`{r.tla}`)")
        lines.append("")
        if r.stats:
            lines.append("| metric | value | verdict | note |")
            lines.append("| --- | ---: | --- | --- |")
            for s in r.stats:
                lines.append(
                    f"| {s.name} | {s.value:.6g} | **{s.verdict}** | {s.note} |"
                )
            lines.append("")
        for info in r.info:
            lines.append(f"- {info}")
        if r.images:
            lines.append("")
            lines.append("### Images")
            for k, v in r.images.items():
                lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(out_path: Path, reports: Sequence[CheckReport]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "tla", "metric", "value", "verdict", "note"])
        for r in reports:
            for s in r.stats:
                w.writerow([r.title, r.tla, s.name, s.value, s.verdict, s.note])


# ---------------------------------------------------------------------------
# Batch MCNR summary (same MD/CSV copied into every sample's SIMcheck_V2)
# ---------------------------------------------------------------------------
BATCH_MCNR_MD_NAME = "SIMcheck_V2_batch_MCNR.md"
BATCH_MCNR_CSV_NAME = "SIMcheck_V2_batch_MCNR.csv"

# User-facing risk bands (finer than SIMcheck YES/MAYBE/NO alone)
_RISK_RANK = {"ERROR": -1, "HIGH": 0, "MED": 1, "LOW": 2, "OK": 3, "NA": 4}


def risk_level_mcnr(mcnr: float) -> str:
    """HIGH/MED/LOW/OK for feature MCNR (project convention for P-spot risk)."""
    if mcnr != mcnr:  # NaN
        return "NA"
    if mcnr < THRESH["mcnr_maybe"]:
        return "HIGH"
    if mcnr < 6.0:
        return "MED"
    if mcnr < THRESH["mcnr_yes"]:
        return "LOW"
    return "OK"


def overall_risk_from_levels(levels: Sequence[str]) -> str:
    if not levels:
        return "NA"
    return min(levels, key=lambda x: _RISK_RANK.get(x, 99))


@dataclass
class SampleMcnrSummary:
    """One sample row for batch MCNR tables."""

    sample: str
    track: str = "A"  # A=bright+reg (recon); B=regOnly
    dr_sub: str = ""
    source: str = ""
    out_dir: str = ""
    d_mcnr: Dict[int, float] = field(default_factory=dict)  # 1-based D1/D2/D3
    mean: float = float("nan")
    avg_map: float = float("nan")
    min_a: float = float("nan")
    max_a: float = float("nan")
    ratio: float = float("nan")
    elapsed_s: float = float("nan")
    error: str = ""


def track_label_from_dr_sub(dr_sub: str) -> str:
    """Map Dr subfolder name to A/B track tag for tables."""
    name = str(dr_sub or "")
    if "regOnly" in name or "reg_only" in name.lower():
        return "B_regOnly"
    return "A_bright+reg"


def extract_mcnr_from_stats_csv(stats_csv: Path) -> Dict[str, Any]:
    """Parse SIMcheck_V2_stats.csv → angle MCNR dict + aggregates."""
    import re

    out: Dict[str, Any] = {"d": {}}
    if not stats_csv.is_file():
        return out
    with stats_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            m = row.get("metric", "")
            try:
                v = float(row["value"])
            except Exception:
                continue
            ma = re.search(r"(?:D|angle)(\d+).*feature MCNR", m)
            if (
                ma
                and "mean of per-angle" not in m
                and "min per-angle" not in m
                and "max per-angle" not in m
                and "average feature MCNR" not in m
            ):
                out["d"][int(ma.group(1))] = v
            if "average feature MCNR (angle-avg map)" in m:
                out["avg_map"] = v
            if "mean of per-angle feature MCNR" in m:
                out["mean"] = v
            if "min per-angle feature MCNR" in m:
                out["min_a"] = v
            if "max per-angle feature MCNR" in m:
                out["max_a"] = v
            if "max/min angle MCNR ratio" in m:
                out["ratio"] = v
    return out


def extract_mcnr_from_reports(reports: Sequence[CheckReport]) -> Dict[str, Any]:
    """Same as CSV parse but from in-memory CheckReport list."""
    import re

    out: Dict[str, Any] = {"d": {}}
    for r in reports:
        if r.tla != "MCN":
            continue
        for s in r.stats:
            m = s.name
            v = float(s.value)
            ma = re.search(r"(?:D|angle)(\d+).*feature MCNR", m)
            if (
                ma
                and "mean of per-angle" not in m
                and "min per-angle" not in m
                and "max per-angle" not in m
                and "average feature MCNR" not in m
            ):
                out["d"][int(ma.group(1))] = v
            if "average feature MCNR (angle-avg map)" in m:
                out["avg_map"] = v
            if "mean of per-angle feature MCNR" in m:
                out["mean"] = v
            if "min per-angle feature MCNR" in m:
                out["min_a"] = v
            if "max per-angle feature MCNR" in m:
                out["max_a"] = v
            if "max/min angle MCNR ratio" in m:
                out["ratio"] = v
    return out


def sample_summary_from_mcnr_dict(
    sample: str,
    m: Dict[str, Any],
    *,
    track: str = "A",
    dr_sub: str = "",
    source: str = "",
    out_dir: str = "",
    elapsed_s: float = float("nan"),
    error: str = "",
) -> SampleMcnrSummary:
    d = dict(m.get("d") or {})
    mean = float(m["mean"]) if "mean" in m else (
        float(np.mean(list(d.values()))) if d else float("nan")
    )
    return SampleMcnrSummary(
        sample=sample,
        track=track or track_label_from_dr_sub(dr_sub),
        dr_sub=dr_sub,
        source=source,
        out_dir=out_dir,
        d_mcnr=d,
        mean=mean,
        avg_map=float(m.get("avg_map", float("nan"))),
        min_a=float(m.get("min_a", min(d.values()) if d else float("nan"))),
        max_a=float(m.get("max_a", max(d.values()) if d else float("nan"))),
        ratio=float(m.get("ratio", float("nan"))),
        elapsed_s=elapsed_s,
        error=error,
    )


def _fmt_mcnr(v: float) -> str:
    if v != v:
        return "—"
    return f"{v:.3f}"


def build_batch_mcnr_markdown(
    rows: Sequence[SampleMcnrSummary],
    meta: Dict[str, Any],
) -> str:
    """
    Full-batch MCNR document. Same text is written under every sample's SIMcheck_V2.
    """
    batch = meta.get("batch", "")
    dr_sub = meta.get("dr_sub", "")
    device = meta.get("device", "")
    checks = meta.get("checks", "mcnr,fpj")
    generated = meta.get("generated", "")
    note = meta.get("note", "")
    calibrated = bool(meta.get("mcnr_calibrated", True))

    def risk(value: float) -> str:
        return risk_level_mcnr(value) if calibrated else "UNCALIBRATED"

    def overall(levels: Sequence[str]) -> str:
        return overall_risk_from_levels(levels) if calibrated else "UNCALIBRATED"

    n_angles = int(meta.get("n_angles", 3) or 3)
    angle_ids = list(range(1, n_angles + 1))

    lines: List[str] = [
        "# SIMcheck 批次 MCNR 汇总",
        "",
        "同一批次内**全部样本**的 MCNR 分方向表。本文件会复制到该批**每个样本**的 "
        "`Dr/SIMcheck_V2/`（以及当次 run 子目录）下，内容相同，便于在任一目录打开即见全批。",
        "",
        f"- **batch**: `{batch}`",
        f"- **Dr_sub**: `{dr_sub}`",
        f"- **timepoint**: `{meta.get('timepoint', 't0003')}`",
        f"- **device**: `{device}`",
        f"- **checks**: `{checks}`",
        f"- **order**: `{meta.get('order', 'ZAP')}`",
        f"- **generated**: `{generated}`",
        f"- **samples_n**: {len(rows)}",
    ]
    if note:
        lines.append(f"- **note**: {note}")
    lines += [
        "",
        "## 方向与判读",
        "",
        "| 标签 | 含义 |",
        "| --- | --- |",
        "| D1 / D2 / D3 | data angle index 0/1/2 = recon 方向 0/1/2（非几何角排序） |",
        *(
            [
                "| HIGH | feature MCNR &lt; 4（SIMcheck NO），P 级难 |",
                "| MED | 4–6，偏弱 |",
                "| LOW | 6–8 |",
                "| OK | ≥ 8 |",
            ]
            if calibrated
            else [
                "| UNCALIBRATED | 扩展 MCNR；不应用五相位阈值 |",
            ]
        ),
        "",
        "总判 = 各方向风险中最差一档。",
        "",
        "## 数据轨",
        "",
        "| track | 含义 | recon |",
        "| --- | --- | --- |",
        "| A_bright+reg | 背景扣除 + 配准 + 区域/方向亮度矫正 → `01-488_padding` | **是** |",
        "| B_regOnly | 背景扣除 + 配准，**无**亮度矫正 → `01-488_regOnly_padding` | 否（仅质检） |",
        "",
        "## 总表",
        "",
    ]

    head = ["样本", "track", "Dr_sub"]
    for ai in angle_ids:
        head += [f"D{ai}", "风险"]
    head += ["mean", "avg-map", "总判", "耗时s", "详情目录"]
    lines.append("| " + " | ".join(head) + " |")
    seps: List[str] = []
    for h in head:
        if h in ("样本", "track", "Dr_sub", "详情目录"):
            seps.append("---")
        elif h in ("风险", "总判"):
            seps.append(":---:")
        else:
            seps.append("---:")
    lines.append("| " + " | ".join(seps) + " |")

    for row in rows:
        track = row.track or track_label_from_dr_sub(row.dr_sub)
        if row.error and not row.d_mcnr:
            cells = [row.sample, track, row.dr_sub or "—"] + ["—", "ERROR"] * n_angles + [
                "—",
                "—",
                "ERROR",
                "—",
                row.error,
            ]
            lines.append("| " + " | ".join(cells) + " |")
            continue
        risks: List[str] = []
        cells = [row.sample, track, row.dr_sub or "—"]
        for ai in angle_ids:
            v = row.d_mcnr.get(ai, float("nan"))
            rk = risk(v)
            risks.append(rk)
            cells += [_fmt_mcnr(v), rk]
        overall_level = overall(risks)
        el = f"{row.elapsed_s:.2f}" if row.elapsed_s == row.elapsed_s else "—"
        cells += [
            _fmt_mcnr(row.mean),
            _fmt_mcnr(row.avg_map),
            f"**{overall_level}**",
            el,
            f"`{row.out_dir}`" if row.out_dir else "—",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## 分样本明细", ""]
    for row in rows:
        track = row.track or track_label_from_dr_sub(row.dr_sub)
        lines.append(f"### {row.sample} · `{track}` · `{row.dr_sub or '—'}`")
        lines.append("")
        if row.error and not row.d_mcnr:
            lines.append(f"- **error**: `{row.error}`")
            lines.append("")
            continue
        lines.append(f"- **source**: `{row.source}`")
        lines.append(f"- **out_dir**: `{row.out_dir}`")
        if row.elapsed_s == row.elapsed_s:
            lines.append(f"- **elapsed_s**: {row.elapsed_s:.2f}")
        lines.append("")
        lines.append("| 方向 | MCNR | 风险 |")
        lines.append("| :---: | ---: | :---: |")
        risks = []
        for ai in angle_ids:
            v = row.d_mcnr.get(ai, float("nan"))
            rk = risk(v)
            risks.append(rk)
            lines.append(f"| D{ai} | {_fmt_mcnr(v)} | {rk} |")
        lines.append(f"| mean | {_fmt_mcnr(row.mean)} | — |")
        lines.append(f"| avg-map | {_fmt_mcnr(row.avg_map)} | — |")
        lines.append(f"| **总判** | — | **{overall(risks)}** |")
        lines.append("")

    lines += [
        "## 文件说明",
        "",
        f"- 本汇总文件名：`{BATCH_MCNR_MD_NAME}` / `{BATCH_MCNR_CSV_NAME}`",
        "- 单样本详细报告仍在各 `.../SIMcheck_V2/<Dr_sub>_t####/SIMcheck_V2_report.md`",
        "- 批次汇总写入：每个样本的 `Dr/SIMcheck_V2/` **以及** 当次 run 子目录（内容一致，含全批样本与全部 track）",
        "- recon 仅使用 A 轨 `01-488_padding`；B 轨仅 SIMcheck 对比",
        "",
    ]
    return "\n".join(lines)


def build_batch_mcnr_csv_rows(
    rows: Sequence[SampleMcnrSummary],
    n_angles: int = 3,
    calibrated: bool = True,
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        risks = [
            (
                risk_level_mcnr(row.d_mcnr.get(ai, float("nan")))
                if calibrated
                else "UNCALIBRATED"
            )
            for ai in range(1, n_angles + 1)
        ]
        rec: Dict[str, Any] = {
            "sample": row.sample,
            "track": row.track or track_label_from_dr_sub(row.dr_sub),
            "dr_sub": row.dr_sub,
            "source": row.source,
            "out_dir": row.out_dir,
            "mean": row.mean,
            "avg_map": row.avg_map,
            "overall_risk": (
                overall_risk_from_levels(risks)
                if calibrated
                else "UNCALIBRATED"
            )
            if row.d_mcnr
            else ("ERROR" if row.error else "NA"),
            "elapsed_s": row.elapsed_s,
            "error": row.error,
        }
        for ai in range(1, n_angles + 1):
            v = row.d_mcnr.get(ai, float("nan"))
            rec[f"D{ai}"] = v
            rec[f"D{ai}_risk"] = (
                risk_level_mcnr(v) if calibrated else "UNCALIBRATED"
            )
        out.append(rec)
    return out


def write_batch_mcnr_documents(
    rows: Sequence[SampleMcnrSummary],
    target_dirs: Sequence[Path],
    meta: Dict[str, Any],
) -> List[Path]:
    """
    Write identical batch MCNR MD + CSV into every target directory.

    Typical targets (both):
      - <sample>/Dr/SIMcheck_V2/
      - <sample>/Dr/SIMcheck_V2/<Dr_sub>_t0003/
    so opening either folder shows the full batch table.
    """
    from datetime import datetime

    meta = dict(meta)
    meta.setdefault("generated", datetime.now().isoformat(timespec="seconds"))
    n_angles = int(meta.get("n_angles", 3) or 3)
    md_text = build_batch_mcnr_markdown(rows, meta)
    csv_rows = build_batch_mcnr_csv_rows(
        rows,
        n_angles=n_angles,
        calibrated=bool(meta.get("mcnr_calibrated", True)),
    )
    fieldnames: List[str] = ["sample", "track", "dr_sub"]
    for ai in range(1, n_angles + 1):
        fieldnames += [f"D{ai}", f"D{ai}_risk"]
    fieldnames += [
        "mean",
        "avg_map",
        "overall_risk",
        "elapsed_s",
        "source",
        "out_dir",
        "error",
    ]

    written: List[Path] = []
    seen: set = set()
    for d in target_dirs:
        d = Path(d)
        key = str(d.resolve()) if d.exists() or True else str(d)
        try:
            key = str(d.resolve())
        except Exception:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        _ensure_dir(d)
        md_path = d / BATCH_MCNR_MD_NAME
        csv_path = d / BATCH_MCNR_CSV_NAME
        md_path.write_text(md_text, encoding="utf-8")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        written.append(md_path)
        written.append(csv_path)
    return written


_DR_TP_RE = re.compile(r"_(\d{3,5})\.(?:tif|tiff)$", re.I)


def _dr_timepoint_index(path: Path) -> Optional[int]:
    m = _DR_TP_RE.search(path.name)
    return int(m.group(1)) if m else None


def find_dr_timepoint_tif(
    pad_dir: Path,
    *,
    start_timepoint: int = 3,
) -> Optional[Path]:
    """
    Pick one Dr stack under a subfolder for SIMcheck.

    Default **start_timepoint=3**: prefer ``*_0003.tif`` (skip warm-up 0001/0002,
    same convention as recon Main_recon_MP). If no ``_0003+`` exists, fall back
    to ``_0001``, then to the first sorted TIFF.
    """
    if not pad_dir.is_dir():
        return None
    tifs = sorted(
        [p for p in pad_dir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}],
        key=lambda p: p.name,
    )
    if not tifs:
        return None
    try:
        start_tp = int(start_timepoint)
    except (TypeError, ValueError):
        start_tp = 3
    if start_tp < 1:
        start_tp = 1

    # exact preferred index first
    for t in tifs:
        ti = _dr_timepoint_index(t)
        if ti is not None and ti == start_tp:
            print(f"[timepoint] prefer _{start_tp:04d}: {t.name}")
            return t
    # any >= start_tp
    for t in tifs:
        ti = _dr_timepoint_index(t)
        if ti is not None and ti >= start_tp:
            print(f"[timepoint] no exact _{start_tp:04d}; use _{ti:04d}: {t.name}")
            return t
    # fallback 0001
    for t in tifs:
        ti = _dr_timepoint_index(t)
        if ti is not None and ti == 1:
            print(f"[timepoint] no _{start_tp:04d}+; fallback _0001: {t.name}")
            return t
    print(f"[timepoint] fallback first file: {tifs[0].name}")
    return tifs[0]


def find_t0_tif(pad_dir: Path, *, start_timepoint: int = 3) -> Optional[Path]:
    """Backward-compatible alias of :func:`find_dr_timepoint_tif` (default tp=3)."""
    return find_dr_timepoint_tif(pad_dir, start_timepoint=start_timepoint)


def list_batch_samples(batch_root: Path, *, skip_peizhun: bool = True) -> List[Path]:
    """Sorted sample dirs under a dataOut batch folder."""
    samples = []
    for p in sorted(batch_root.iterdir(), key=lambda x: x.name):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        if skip_peizhun and (
            p.name.startswith("99") or "peizhun" in p.name.lower()
        ):
            continue
        samples.append(p)
    return samples


def run_batch_simcheck(
    batch_root: Path,
    *,
    dr_sub: str = "01-488_padding",
    dr_subs: Optional[Sequence[str]] = None,
    order: str = "ZAP",
    n_phases: int = 5,
    n_angles: int = 3,
    n_z: Optional[int] = 8,
    n_c: int = 1,
    n_t: int = 1,
    checks: Optional[set] = None,
    mcnr_zw: int = 1,
    prefer_gpu: bool = True,
    sample_filter: Optional[Sequence[str]] = None,
    start_timepoint: int = 3,
) -> Tuple[List[SampleMcnrSummary], List[Path]]:
    """
    Run SIMcheck on each sample under batch_root, then write batch MCNR MD/CSV
    into every sample's SIMcheck_V2 (and each run subdir).

    dr_subs: e.g. ["01-488_padding", "01-488_regOnly_padding"] for A+B tracks.
    If None, uses single dr_sub.

    start_timepoint: default 3 → prefer *_0003.tif (skip warm-up 0001/0002);
    fall back to 0001 if missing (same as recon).

    Returns (summaries, written_paths).
    """
    checks = checks or {"mcnr", "fpj"}
    mcnr_orders, mcnr_noise_method, mcnr_calibrated = (
        mcnr_policy_for_phases(n_phases)
        if "mcnr" in checks
        else ((), "", False)
    )
    dev = _init_device(prefer_gpu=prefer_gpu)
    print(f"[device] {dev}")
    print(f"[start_timepoint] {start_timepoint}")
    batch_root = Path(batch_root)
    samples = list_batch_samples(batch_root)
    if sample_filter:
        want = set(sample_filter)
        samples = [s for s in samples if s.name in want or any(w in s.name for w in want)]

    sub_list = list(dr_subs) if dr_subs else [dr_sub]
    print(f"[dr_subs] {sub_list}")

    rows: List[SampleMcnrSummary] = []
    target_dirs: List[Path] = []
    used_tps: List[str] = []

    for s in samples:
        target_dirs.append(s / "Dr" / "SIMcheck_V2")
        for dr_sub_i in sub_list:
            track = track_label_from_dr_sub(dr_sub_i)
            pad = s / "Dr" / dr_sub_i
            t0 = find_dr_timepoint_tif(pad, start_timepoint=start_timepoint)
            if t0 is None:
                print(f"[skip] {s.name} / {dr_sub_i}: no tif")
                rows.append(
                    SampleMcnrSummary(
                        sample=s.name,
                        track=track,
                        dr_sub=dr_sub_i,
                        error=f"no tif under Dr/{dr_sub_i}",
                    )
                )
                continue

            ti = _dr_timepoint_index(t0)
            used_tps.append(f"t{ti:04d}" if ti is not None else t0.stem[-5:])
            out_dir = default_out_dir_for_input(t0)
            _ensure_dir(out_dir)
            simcheck_root = out_dir.parent  # .../Dr/SIMcheck_V2
            target_dirs.append(simcheck_root)
            target_dirs.append(out_dir)

            print(f"[run] {s.name} [{track}] <- {dr_sub_i} / {t0.name}")
            print(f"  out -> {out_dir}")
            t_sample = time.perf_counter()
            try:
                raw = load_raw(t0, order, n_phases, n_angles, n_z, n_c, n_t)
                print(f"  shape {raw.shape}")
                raw_dir = _ensure_dir(out_dir / "raw")
                reports: List[CheckReport] = []
                if "mcnr" in checks:
                    t1 = time.perf_counter()
                    reports.append(
                        check_mcnr(
                            raw,
                            raw_dir,
                            harmonic_orders=mcnr_orders,
                            noise_method=mcnr_noise_method,
                            calibrated=mcnr_calibrated,
                            zw=mcnr_zw,
                        )
                    )
                    print(f"  MCNR {time.perf_counter()-t1:.2f}s")
                if "fpj" in checks:
                    t1 = time.perf_counter()
                    reports.append(check_fpj(raw, raw_dir))
                    print(f"  FPJ {time.perf_counter()-t1:.2f}s")
                if "cip" in checks:
                    reports.append(check_cip(raw, raw_dir))
                if "miv" in checks:
                    reports.append(check_miv(raw, raw_dir))
                if "pwf" in checks:
                    reports.append(util_pseudo_widefield(raw, raw_dir))

                meta_one = {
                    "sample": s.name,
                    "track": track,
                    "dr_sub": dr_sub_i,
                    "order": order,
                    "n_phases": n_phases,
                    "n_angles": n_angles,
                    "n_z": n_z,
                    "device": dev,
                    "checks": sorted(checks),
                    "source": str(t0),
                    "out_dir": str(out_dir),
                }
                write_markdown(
                    out_dir / "SIMcheck_V2_report.md", t0, "raw", reports, meta_one
                )
                write_csv(out_dir / "SIMcheck_V2_stats.csv", reports)
                m = extract_mcnr_from_reports(reports)
                elapsed = time.perf_counter() - t_sample
                summary = sample_summary_from_mcnr_dict(
                    s.name,
                    m,
                    track=track,
                    dr_sub=dr_sub_i,
                    source=str(t0),
                    out_dir=str(out_dir),
                    elapsed_s=elapsed,
                )
                rows.append(summary)
                d = summary.d_mcnr
                print(
                    "  "
                    + " ".join(
                        f"D{ai}={d.get(ai, float('nan')):.3f}({risk_level_mcnr(d.get(ai, float('nan')))})"
                        for ai in range(1, n_angles + 1)
                    )
                    + f"  total={elapsed:.2f}s"
                )
            except Exception as e:
                print(f"  ERROR {s.name} / {dr_sub_i}: {e}")
                rows.append(
                    SampleMcnrSummary(
                        sample=s.name,
                        track=track,
                        dr_sub=dr_sub_i,
                        source=str(t0),
                        out_dir=str(out_dir),
                        error=str(e),
                        elapsed_s=time.perf_counter() - t_sample,
                    )
                )

    # Prefer actual used tp(s); default label when empty
    if used_tps:
        tp_label = used_tps[0] if len(set(used_tps)) == 1 else ",".join(used_tps)
    else:
        tp_label = f"t{int(start_timepoint):04d}"
    meta = {
        "batch": batch_root.name,
        "batch_path": str(batch_root),
        "dr_sub": ",".join(sub_list),
        "timepoint": tp_label,
        "start_timepoint": start_timepoint,
        "device": dev,
        "checks": ",".join(sorted(checks)),
        "order": order,
        "n_angles": n_angles,
        "n_phases": n_phases,
        "n_z": n_z,
        "mcnr_calibrated": mcnr_calibrated,
    }
    written = write_batch_mcnr_documents(rows, target_dirs, meta)

    # CLI table
    print("\n=== Batch MCNR ===")
    print(
        f"{'sample':<16} {'track':<14} "
        + " ".join(f"{'D'+str(i):>8}" for i in range(1, n_angles + 1))
        + f" {'mean':>8} {'risk':>6}"
    )
    for row in rows:
        track = row.track or track_label_from_dr_sub(row.dr_sub)
        if row.error and not row.d_mcnr:
            print(f"{row.sample:<16} {track:<14} ERROR: {row.error}")
            continue
        risks = [
            risk_level_mcnr(row.d_mcnr.get(i, float("nan")))
            for i in range(1, n_angles + 1)
        ]
        vals = " ".join(
            f"{row.d_mcnr.get(i, float('nan')):8.3f}" for i in range(1, n_angles + 1)
        )
        print(
            f"{row.sample:<16} {track:<14} {vals} {row.mean:8.3f} "
            f"{overall_risk_from_levels(risks):>6}"
        )
    print(
        f"[batch] wrote {BATCH_MCNR_MD_NAME} × "
        f"{len(set(str(p.parent) for p in written if p.suffix=='.md'))} dirs"
    )
    for p in written:
        if p.suffix == ".md":
            print(f"  → {p}")

    return rows, written


# ---------------------------------------------------------------------------
# Optional Fiji launcher
# ---------------------------------------------------------------------------
def open_in_fiji(
    image_path: Path,
    fiji_exe: Path = DEFAULT_FIJI,
    simcheck_jar: Path = DEFAULT_SIMCHECK_JAR,
) -> None:
    """
    Launch Fiji with the image open. User can then run Plugins > SIMcheck.
    Ensures jar is present under Fiji.app/plugins if possible.
    """
    if not fiji_exe.is_file():
        raise FileNotFoundError(f"Fiji not found: {fiji_exe}")
    plugins_dir = fiji_exe.parent / "plugins"
    target = plugins_dir / simcheck_jar.name
    if simcheck_jar.is_file() and not target.is_file():
        try:
            import shutil

            shutil.copy2(simcheck_jar, target)
            print(f"[FIJI] copied SIMcheck jar → {target}")
        except Exception as e:
            print(f"[FIJI] could not copy jar into plugins: {e}")
            print(f"       ensure {simcheck_jar} is on Fiji plugin path")
    # headless-friendly open still needs GUI for SIMcheck dialogs
    cmd = [str(fiji_exe), str(image_path)]
    print("[FIJI] launching:", " ".join(cmd))
    subprocess.Popen(cmd, cwd=str(fiji_exe.parent))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SIMcheck_V2: Python SIM quality checks (Ball 2015 / SIMcheck 1.3)"
    )
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="raw or reconstructed TIFF path (omit when using --batch-root)",
    )
    p.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help="batch dataOut folder: run SIMcheck on each sample Dr/<dr-sub> "
        "(default timepoint *_0003, skip warm-up 0001/0002), "
        "then write identical SIMcheck_V2_batch_MCNR.md into every sample SIMcheck_V2",
    )
    p.add_argument(
        "--start-tp",
        type=int,
        default=3,
        help="preferred Dr timepoint index (default 3 → *_0003.tif; "
        "fallback to 0001 if missing). Aligns with recon dr_start_timepoint",
    )
    p.add_argument(
        "--dr-sub",
        type=str,
        default="01-488_padding",
        help="Dr subfolder name under each sample (batch mode; default 01-488_padding)",
    )
    p.add_argument(
        "--dr-subs",
        type=str,
        default="",
        help="batch mode: comma-separated Dr subfolders, e.g. "
        "01-488_padding,01-488_regOnly_padding (A+B tracks; overrides --dr-sub)",
    )
    p.add_argument(
        "--samples",
        type=str,
        default="",
        help="batch mode: comma-separated sample name filters (optional)",
    )
    p.add_argument(
        "--mode",
        choices=["raw", "recon", "both"],
        default="raw",
        help="raw checks / recon checks / both (same file used twice if both)",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="output directory")
    p.add_argument(
        "--order",
        choices=["CPZAT", "PZAT", "ZAP"],
        default="ZAP",
        help="raw stack dimension order (default ZAP matches legacy SIMcheck.py)",
    )
    p.add_argument("--n-phases", type=int, default=5)
    p.add_argument("--n-angles", type=int, default=3)
    p.add_argument("--n-z", type=int, default=None, help="Z count; inferred if omitted")
    p.add_argument("--n-c", type=int, default=1)
    p.add_argument("--n-t", type=int, default=1)
    p.add_argument("--zwin", type=int, default=9, help="CIP Z window size")
    p.add_argument("--mcnr-zw", type=int, default=1, help="MCNR Z half-window")
    p.add_argument(
        "--checks",
        type=str,
        default="mcnr,fpj",
        help="comma list: mcnr,fpj,cip,miv,pwf,rih,sam,ftr,all  (default: mcnr,fpj)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="compute device: auto=GPU if available else CPU (default auto)",
    )
    p.add_argument("--open-fiji", action="store_true", help="also open input in Fiji")
    p.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    p.add_argument("--simcheck-jar", type=Path, default=DEFAULT_SIMCHECK_JAR)
    return p


def parse_checks(s: str) -> set:
    s = s.strip().lower()
    if s == "all":
        return {"cip", "miv", "mcnr", "fpj", "pwf", "rih", "sam", "ftr"}
    if s in ("core", "fast"):
        return {"mcnr", "fpj"}
    return {x.strip() for x in s.split(",") if x.strip()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # --- batch mode: full-batch MCNR MD into every sample SIMcheck_V2 ---
    if args.batch_root is not None:
        br = Path(args.batch_root)
        if not br.is_dir():
            print(f"ERROR: --batch-root not a directory: {br}", file=sys.stderr)
            return 2
        prefer_gpu = args.device in ("auto", "cuda")
        filt = [x.strip() for x in args.samples.split(",") if x.strip()] or None
        subs = [x.strip() for x in args.dr_subs.split(",") if x.strip()] or None
        nz = args.n_z if args.n_z is not None else 8
        run_batch_simcheck(
            br,
            dr_sub=args.dr_sub,
            dr_subs=subs,
            order=args.order,
            n_phases=args.n_phases,
            n_angles=args.n_angles,
            n_z=nz,
            n_c=args.n_c,
            n_t=args.n_t,
            checks=parse_checks(args.checks),
            mcnr_zw=args.mcnr_zw,
            prefer_gpu=prefer_gpu,
            sample_filter=filt,
            start_timepoint=args.start_tp,
        )
        return 0

    if args.input is None:
        print("ERROR: provide input TIFF or --batch-root", file=sys.stderr)
        return 2

    inp: Path = args.input
    if not inp.is_file():
        print(f"ERROR: input not found: {inp}", file=sys.stderr)
        return 2

    prefer_gpu = args.device in ("auto", "cuda")
    dev_name = _init_device(prefer_gpu=prefer_gpu)
    if args.device == "cuda" and not str(dev_name).startswith("cuda"):
        print(
            "WARNING: --device cuda requested but CUDA unavailable; using",
            dev_name,
            file=sys.stderr,
        )
    print(f"[device] {dev_name}")

    out_dir = args.out_dir if args.out_dir is not None else default_out_dir_for_input(inp)
    _ensure_dir(out_dir)
    print(f"[out] {out_dir}")

    checks = parse_checks(args.checks)
    mcnr_orders, mcnr_noise_method, mcnr_calibrated = (
        mcnr_policy_for_phases(args.n_phases)
        if "mcnr" in checks
        else ((), "", False)
    )
    reports: List[CheckReport] = []
    meta = {
        "order": args.order,
        "n_phases": args.n_phases,
        "n_angles": args.n_angles,
        "n_z": args.n_z,
        "n_c": args.n_c,
        "n_t": args.n_t,
        "device": dev_name,
        "checks": sorted(checks),
        "mcnr_orders": list(mcnr_orders),
        "mcnr_noise_method": mcnr_noise_method,
        "mcnr_calibrated": mcnr_calibrated,
        "simcheck_jar": str(args.simcheck_jar),
        "fiji": str(args.fiji),
    }
    t0 = time.perf_counter()

    if args.mode in ("raw", "both"):
        print(f"[raw] load {inp} order={args.order}")
        raw = load_raw(
            inp,
            args.order,
            args.n_phases,
            args.n_angles,
            args.n_z,
            args.n_c,
            args.n_t,
        )
        meta["raw_shape_CPZATYX"] = list(raw.shape)
        print(f"[raw] shape (C,P,Z,A,T,Y,X) = {raw.shape}")
        raw_dir = _ensure_dir(out_dir / "raw")
        if "cip" in checks:
            print("[raw] CIP ...")
            reports.append(check_cip(raw, raw_dir, zwin=args.zwin))
        if "miv" in checks:
            print("[raw] MIV ...")
            reports.append(check_miv(raw, raw_dir))
        if "mcnr" in checks:
            t1 = time.perf_counter()
            print("[raw] MCNR ...")
            reports.append(
                check_mcnr(
                    raw,
                    raw_dir,
                    harmonic_orders=mcnr_orders,
                    noise_method=mcnr_noise_method,
                    calibrated=mcnr_calibrated,
                    zw=args.mcnr_zw,
                )
            )
            print(f"[raw] MCNR done in {time.perf_counter()-t1:.2f}s")
        if "fpj" in checks:
            t1 = time.perf_counter()
            print("[raw] FPJ ...")
            reports.append(check_fpj(raw, raw_dir))
            print(f"[raw] FPJ done in {time.perf_counter()-t1:.2f}s")
        if "pwf" in checks:
            print("[raw] PWF ...")
            reports.append(util_pseudo_widefield(raw, raw_dir))

    if args.mode in ("recon", "both"):
        print(f"[recon] load {inp}")
        recon = load_stack(inp)
        if recon.ndim == 3:
            recon_c = recon[None, ...]
        else:
            recon_c = recon
        meta["recon_shape"] = list(recon_c.shape)
        recon_dir = _ensure_dir(out_dir / "recon")
        if "rih" in checks:
            print("[recon] RIH ...")
            reports.append(check_rih(recon_c, recon_dir))
        if "sam" in checks:
            print("[recon] SAM ...")
            reports.append(check_sam(recon_c, recon_dir))
        if "ftr" in checks:
            print("[recon] FTR ...")
            reports.append(check_ftr(recon_c, recon_dir))

    md_path = out_dir / "SIMcheck_V2_report.md"
    csv_path = out_dir / "SIMcheck_V2_stats.csv"
    write_markdown(md_path, inp, args.mode, reports, meta)
    write_csv(csv_path, reports)
    print(f"[done] report → {md_path}")
    print(f"[done] stats  → {csv_path}")
    print(f"[done] total {time.perf_counter()-t0:.2f}s on {dev_name}")

    # summary table
    print("\n=== Summary ===")
    for r in reports:
        for s in r.stats:
            if s.verdict != "NA":
                print(f"  [{s.verdict:5s}] {r.tla:3s} | {s.name}: {s.value:.4g}")

    if args.open_fiji:
        try:
            open_in_fiji(inp, args.fiji, args.simcheck_jar)
        except Exception as e:
            print(f"[FIJI] failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
