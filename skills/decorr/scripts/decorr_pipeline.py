"""Headless Python decorrelation analysis for reconstructed 3D MP-SIM TIFFs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import fft as sp_fft


@dataclass(frozen=True)
class DecorrParameters:
    nr: int = 50
    ng: int = 10
    dt: float = 0.01
    na: int = 20
    rmin: float = 0.2
    rmax: float = 0.3

    @property
    def radii(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.nr, dtype=np.float64)


@dataclass(frozen=True)
class MontageParameters:
    num_groups: int = 12
    num_points: int = 32
    end_replication: int = 4
    apodization: int = 5
    canvas_size: int = 1024
    seed: int = 20250124


SIGNAL_CONTRAST_RATIO = 0.05
MIN_AXIAL_PLANES = 8
OUTPUT_SCHEMA_VERSION = 2
CUDA_CURVE_BATCH_SIZE = 16


def read_tiff_stack_float32(path: str | Path) -> np.ndarray:
    stack = np.asarray(tifffile.imread(path), dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    if stack.ndim != 3:
        raise ValueError(f"Expected a 2D/3D grayscale TIFF, got {stack.shape}: {path}")
    return stack


def write_tiff_float32(path: str | Path, stack: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.asarray(stack, dtype=np.float32),
        photometric="minisblack",
        compression=None,
        metadata=None,
    )


def select_xy_plane(
    stack: np.ndarray,
    *,
    plane_index: int | None = None,
    mode: str = "tenengrad",
    top_count: int = 5,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    if stack.ndim != 3:
        raise ValueError(f"Expected Z x Y x X data, got {stack.shape}")
    tenengrad = np.empty(stack.shape[0], dtype=np.float64)
    means = np.empty(stack.shape[0], dtype=np.float64)
    for z, image in enumerate(np.asarray(stack, dtype=np.float32)):
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)
        gx[:, 1:-1] = 0.5 * (image[:, 2:] - image[:, :-2])
        gy[1:-1, :] = 0.5 * (image[2:, :] - image[:-2, :])
        tenengrad[z] = np.nanmean(
            (gx.astype(np.float64) ** 2) + (gy.astype(np.float64) ** 2)
        )
        means[z] = np.nanmean(image, dtype=np.float64)

    mode = mode.lower()
    if mode not in {"tenengrad", "mean"}:
        raise ValueError("mode must be 'tenengrad' or 'mean'")
    selection_score = tenengrad if mode == "tenengrad" else means
    order = np.argsort(-np.nan_to_num(selection_score, nan=-np.inf), kind="stable")
    if plane_index is None:
        selected = int(order[0])
    else:
        selected = int(plane_index) - 1
        if not 0 <= selected < stack.shape[0]:
            raise IndexError(f"Plane index {plane_index} outside 1..{stack.shape[0]}")
        mode = "manual"

    ranks = np.empty(stack.shape[0], dtype=np.int64)
    ranks[order] = np.arange(1, stack.shape[0] + 1)
    top_count = min(top_count, stack.shape[0])
    scores = pd.DataFrame(
        {
            "plane_index": np.arange(1, stack.shape[0] + 1),
            "tenengrad": tenengrad,
            "mean_value": means,
            "selection_score": selection_score,
            "rank": ranks,
            "is_top_candidate": ranks <= top_count,
            "is_selected": np.arange(stack.shape[0]) == selected,
        }
    )
    result = {
        "selected_index": selected + 1,
        "mode": mode,
        "top_indices": (order[:top_count] + 1).tolist(),
        "top_scores": selection_score[order[:top_count]].tolist(),
    }
    return stack[selected], result, scores


def apodize_rect(image: np.ndarray, width: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    ny, nx = image.shape
    n = min(width, nx // 2, ny // 2)
    if n <= 0:
        return image.copy()
    ramp = (0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))).astype(
        np.float32
    )
    wx = np.ones(nx, dtype=np.float32)
    wy = np.ones(ny, dtype=np.float32)
    wx[:n] = ramp
    wx[-n:] = ramp[::-1]
    wy[:n] = ramp
    wy[-n:] = ramp[::-1]
    return image * wy[:, None] * wx[None, :]


def _odd_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    return image[: height - int(height % 2 == 0), : width - int(width % 2 == 0)]


def _fft2_centered(image: np.ndarray) -> np.ndarray:
    return sp_fft.fftshift(
        sp_fft.fftn(sp_fft.fftshift(np.asarray(image, dtype=np.float32)), workers=-1)
    )


def _local_max(curve: np.ndarray, dt: float) -> tuple[int, float]:
    working = np.asarray(curve, dtype=np.float64).copy()
    if working.size < 2:
        return 0, float(working[0])
    while working.size > 1:
        index = int(np.argmax(working))
        amplitude = float(working[index])
        if index == working.size - 1:
            working = working[:-1]
        elif index == 0 or amplitude - float(np.min(working[index:])) >= dt:
            break
        else:
            working = working[:-1]
    index = int(np.argmax(working))
    return index, float(working[index])


class _RadialSampler:
    def __init__(self, r2: np.ndarray):
        flat = np.asarray(r2, dtype=np.float32).ravel()
        self.order = np.argsort(flat, kind="stable")
        self.sorted_r2 = flat[self.order]

    def curve(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        energy1: float,
        radii: np.ndarray,
    ) -> np.ndarray:
        first = np.asarray(image1).ravel()[self.order]
        second = np.asarray(image2).ravel()[self.order]
        numerator = np.cumsum(np.real(first * np.conj(second)), dtype=np.float64)
        second_energy = np.cumsum(np.abs(second) ** 2, dtype=np.float64)
        counts = np.searchsorted(self.sorted_r2, radii**2, side="left")
        result = np.zeros(len(radii), dtype=np.float64)
        valid = counts > 0
        positions = counts[valid] - 1
        denominator = energy1 * np.sqrt(second_energy[positions])
        values = np.divide(
            numerator[positions],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        result[valid] = np.floor(1000.0 * values) / 1000.0
        result[~np.isfinite(result)] = 0.0
        return result


def _frequency_geometry(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, width, dtype=np.float32),
        np.linspace(-1.0, 1.0, height, dtype=np.float32),
    )
    return x * x + y * y, np.arctan2(y, x).astype(np.float32)


def imdecorr_global(
    image: np.ndarray, params: DecorrParameters | None = None
) -> dict[str, object]:
    params = params or DecorrParameters()
    image = _odd_crop(np.asarray(image, dtype=np.float32))
    r2, _ = _frequency_geometry(image.shape)
    sampler = _RadialSampler(r2)
    mask0 = r2 < 1.0
    spectrum = _fft2_centered(image) * mask0
    phase = np.divide(
        spectrum,
        np.abs(spectrum),
        out=np.zeros_like(spectrum),
        where=np.abs(spectrum) > 0,
    )
    energy = float(np.sqrt(np.sum(np.abs(spectrum) ** 2, dtype=np.float64)))
    radii = params.radii
    d0 = sampler.curve(spectrum, phase, energy, radii)
    index0, snr0 = _local_max(d0, params.dt)
    kc0 = float(radii[index0])
    gmax = 2.0 / max(kc0, np.finfo(float).eps)
    if not np.isfinite(gmax):
        gmax = max(image.shape) / 2.0
    g = np.r_[
        image.shape[0] / 4.0,
        np.exp(np.linspace(np.log(gmax), np.log(0.15), params.ng - 1)),
    ]
    search_r = radii.copy()
    curves = np.zeros((params.nr, params.ng * 2), dtype=np.float64)
    for refinement in range(2):
        stage_kc = np.zeros(params.ng, dtype=np.float64)
        stage_snr = np.zeros(params.ng, dtype=np.float64)
        for h in range(params.ng):
            highpass = 1.0 - np.exp(-2.0 * g[h] ** 2 * r2)
            filtered = spectrum * highpass
            filtered_energy = float(
                np.sqrt(np.sum(np.abs(filtered) ** 2, dtype=np.float64))
            )
            curve = sampler.curve(filtered, phase, filtered_energy, search_r)
            curves[:, h + refinement * params.ng] = curve
            peak, stage_snr[h] = _local_max(curve, params.dt)
            stage_kc[h] = search_r[peak]
        if refinement == 0:
            best = int(np.argmax(stage_kc))
            low = max(0, best - 1)
            high = min(params.ng - 1, best + 1)
            g = np.exp(np.linspace(np.log(g[low]), np.log(g[high]), params.ng))
            center = float(np.max(stage_kc))
            search_r = np.linspace(
                max(0.0, center - 0.2), min(1.0, center + 0.2), params.nr
            )
    final_kc = np.r_[stage_kc, kc0]
    final_snr = np.r_[stage_snr, snr0]
    final_kc[final_snr < 0.05] = 0.0
    best = int(np.argmax(final_kc))
    kc = float(final_kc[best]) if final_kc[best] > 0 else math.nan
    return {
        "kc": kc,
        "A0": float(snr0),
        "peak_snr": float(final_snr[best]),
        "d0": d0,
        "curves": curves,
        "refined_radii": search_r,
        "device": "cpu",
    }


def _sector_mask(
    theta: np.ndarray, r2: np.ndarray, sector_count: int, sector: int
) -> np.ndarray:
    matlab_sector = sector + 1
    angle_min = -(3 + 2 * (sector_count - matlab_sector)) * np.pi / (
        2 * sector_count
    )
    angle_max = -(1 + 2 * (sector_count - matlab_sector)) * np.pi / (
        2 * sector_count
    )
    angle_min = (angle_min + np.pi) % (2 * np.pi) - np.pi
    angle_max = (angle_max + np.pi) % (2 * np.pi) - np.pi
    if angle_min < angle_max:
        mask = (theta >= angle_min) & (theta <= angle_max)
    else:
        mask = (theta >= angle_min) | (theta <= angle_max)
    return mask | np.rot90(mask, 2) | (r2 < 0.01**2)


def _imdecorr_sectorial_cpu(
    image: np.ndarray,
    params: DecorrParameters | None = None,
    *,
    selected_sector: int | None = None,
) -> dict[str, object]:
    params = params or DecorrParameters()
    if selected_sector is not None and not 0 <= selected_sector < params.na:
        raise IndexError(f"Sector {selected_sector} outside 0..{params.na - 1}")
    image = _odd_crop(np.asarray(image, dtype=np.float32))
    r2, theta = _frequency_geometry(image.shape)
    sampler = _RadialSampler(r2)
    spectrum = _fft2_centered(image)
    mask0 = r2 < 1.0
    radii = params.radii
    masks = [_sector_mask(theta, r2, params.na, s) for s in range(params.na)]
    d0 = np.zeros((params.nr, params.na), dtype=np.float64)
    initial_kc = np.zeros(params.na, dtype=np.float64)
    initial_snr = np.zeros(params.na, dtype=np.float64)
    for sector, sector_mask in enumerate(masks):
        sector_spectrum = spectrum * mask0 * sector_mask
        normalized = np.divide(
            sector_spectrum,
            np.abs(sector_spectrum),
            out=np.zeros_like(sector_spectrum),
            where=np.abs(sector_spectrum) > 0,
        )
        energy = float(
            np.sqrt(np.sum(np.abs(sector_spectrum) ** 2, dtype=np.float64))
        )
        d0[:, sector] = sampler.curve(
            sector_spectrum, normalized, energy, radii
        )
        peak, initial_snr[sector] = _local_max(d0[:, sector], params.dt)
        initial_kc[sector] = radii[peak]

    gmax = 1.0 / max(float(np.min(initial_kc)), np.finfo(float).eps)
    g = np.exp(np.linspace(np.log(gmax), np.log(0.14), params.ng))
    search_r = np.tile(radii, (params.na, 1))
    curves = np.full(
        (params.nr, params.na, params.ng * 2),
        np.nan if selected_sector is not None else 0.0,
        dtype=np.float64,
    )
    middle = params.na // 2
    active_sectors = (
        range(params.na) if selected_sector is None else (selected_sector,)
    )
    for refinement in range(2):
        stage_kc = np.zeros((params.na, params.ng), dtype=np.float64)
        stage_snr = np.zeros((params.na, params.ng), dtype=np.float64)
        highpasses = [1.0 - np.exp(-2.0 * value**2 * r2) for value in g]
        for sector in active_sectors:
            sector_mask = masks[sector]
            for h, highpass in enumerate(highpasses):
                filtered = spectrum * mask0 * sector_mask * highpass
                normalized = np.divide(
                    filtered,
                    np.abs(filtered),
                    out=np.zeros_like(filtered),
                    where=np.abs(filtered) > 0,
                )
                energy = float(
                    np.sqrt(np.sum(np.abs(filtered) ** 2, dtype=np.float64))
                )
                curve = sampler.curve(
                    filtered, normalized, energy, search_r[sector]
                )
                curves[:, sector, h + refinement * params.ng] = curve
                peak, stage_snr[sector, h] = _local_max(curve, params.dt)
                stage_kc[sector, h] = search_r[sector, peak]
        if refinement == 0:
            metric = np.sqrt(np.maximum(stage_kc, 0) * np.maximum(stage_snr, 0))
            if selected_sector is not None:
                best = int(np.argmax(metric[selected_sector]))
                low_g = max(0, best - 1)
                high_g = min(params.ng - 1, best + 1)
                g = np.exp(
                    np.linspace(np.log(g[low_g]), np.log(g[high_g]), params.ng)
                )
                center = stage_kc[selected_sector, best]
                search_r[selected_sector] = np.linspace(
                    max(0.0, center - params.rmin),
                    min(1.0, center + params.rmax),
                    params.nr,
                )
                continue
            best_by_sector = np.argmax(metric, axis=1)
            middle_best = int(best_by_sector[middle])
            low_g = max(0, middle_best - 1)
            high_g = min(params.ng - 1, middle_best + 1)
            g = np.exp(
                np.linspace(np.log(g[low_g]), np.log(g[high_g]), params.ng)
            )
            for sector in range(params.na):
                center = stage_kc[sector, best_by_sector[sector]]
                search_r[sector] = np.linspace(
                    max(0.0, center - params.rmin),
                    min(1.0, center + params.rmax),
                    params.nr,
                )

    if selected_sector is not None:
        sector_kc = stage_kc[selected_sector].copy()
        sector_snr = stage_snr[selected_sector]
        sector_kc[sector_snr < 0.05] = 0.0
        best = int(
            np.argmax(
                np.sqrt(np.maximum(sector_kc, 0) * np.maximum(sector_snr, 0))
            )
        )
        chosen = sector_kc[best]
        valid_value = (
            chosen > search_r[selected_sector, 0]
            and chosen < search_r[selected_sector, -1]
            and chosen > 0
        )
        kc = np.full(params.na, np.nan, dtype=np.float64)
        peak_snr = np.full(params.na, np.nan, dtype=np.float64)
        valid = np.zeros(params.na, dtype=bool)
        kc[selected_sector] = chosen if valid_value else np.nan
        peak_snr[selected_sector] = sector_snr[best]
        valid[selected_sector] = valid_value
        return {
            "kcGM": kc,
            "A0": initial_snr,
            "peak_snr": peak_snr,
            "interior_valid": valid,
            "d0": d0,
            "curves": curves,
            "device": "cpu",
        }

    final_low = search_r[:, 0]
    final_high = search_r[:, -1]
    stage_kc[stage_snr < 0.05] = 0.0
    metric = np.sqrt(np.maximum(stage_kc, 0) * np.maximum(stage_snr, 0))
    best = np.argmax(metric, axis=1)
    kc = stage_kc[np.arange(params.na), best]
    peak_snr = stage_snr[np.arange(params.na), best]
    valid = (kc > final_low) & (kc < final_high) & (kc > 0)
    kc = np.where(valid, kc, np.nan)
    return {
        "kcGM": kc,
        "A0": initial_snr,
        "peak_snr": peak_snr,
        "interior_valid": valid,
        "d0": d0,
        "curves": curves,
        "device": "cpu",
    }


class _TorchRadialSampler:
    def __init__(self, r2: np.ndarray, torch):
        flat = np.asarray(r2, dtype=np.float32).ravel()
        order = np.argsort(flat, kind="stable")
        self.sorted_r2 = flat[order]
        self.order = torch.as_tensor(order, device="cuda", dtype=torch.int64)
        self.torch = torch

    def curve(
        self,
        image1,
        image2,
        energy1: float,
        radii: np.ndarray,
    ) -> np.ndarray:
        return self.curves(
            image1[None],
            image2[None],
            self.torch.as_tensor([energy1], device="cuda"),
            np.asarray(radii)[None],
        )[0]

    def curves(
        self,
        image1,
        image2,
        energy1,
        radii: np.ndarray,
    ) -> np.ndarray:
        torch = self.torch
        batch_size = image1.shape[0]
        first = image1.reshape(batch_size, -1)[:, self.order]
        second = image2.reshape(batch_size, -1)[:, self.order]
        numerator = torch.cumsum(
            torch.real(first * torch.conj(second)), dim=1, dtype=torch.float64
        )
        second_energy = torch.cumsum(
            torch.abs(second) ** 2, dim=1, dtype=torch.float64
        )
        radii = np.asarray(radii, dtype=np.float64)
        counts = np.searchsorted(self.sorted_r2, radii**2, side="left")
        valid = counts > 0
        positions = torch.as_tensor(
            np.maximum(counts - 1, 0), device="cuda", dtype=torch.int64
        )
        numerator_at_radii = torch.gather(numerator, 1, positions)
        energy_at_radii = torch.gather(second_energy, 1, positions)
        denominator = energy1.reshape(-1, 1) * torch.sqrt(energy_at_radii)
        values = torch.where(
            denominator > 0,
            numerator_at_radii / denominator,
            torch.zeros_like(denominator),
        )
        valid_tensor = torch.as_tensor(valid, device="cuda")
        result = torch.where(
            valid_tensor,
            torch.floor(1000.0 * values).div_(1000.0),
            torch.zeros_like(values),
        )
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
        return result.cpu().numpy()


# ponytail: formal runs use fixed shapes; add LRU eviction for arbitrary-size services.
_TORCH_SECTORIAL_GEOMETRY_CACHE: dict[tuple[object, ...], tuple[object, ...]] = {}


def _torch_sectorial_geometry(
    shape: tuple[int, int],
    sector_count: int,
    torch,
) -> tuple[object, ...]:
    key = (torch.cuda.current_device(), shape, sector_count)
    cached = _TORCH_SECTORIAL_GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    r2_cpu, theta_cpu = _frequency_geometry(shape)
    r2 = torch.as_tensor(r2_cpu, device="cuda")
    masks = torch.stack(
        [
            torch.as_tensor(
                _sector_mask(theta_cpu, r2_cpu, sector_count, sector),
                device="cuda",
            )
            for sector in range(sector_count)
        ]
    )
    cached = (
        r2,
        r2 < 1.0,
        masks,
        _TorchRadialSampler(r2_cpu, torch),
    )
    _TORCH_SECTORIAL_GEOMETRY_CACHE[key] = cached
    return cached


def _imdecorr_sectorial_cuda(
    image: np.ndarray,
    params: DecorrParameters,
    selected_sector: int | None,
) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in PyTorch")
    image = _odd_crop(np.asarray(image, dtype=np.float32))
    r2, mask0, masks, sampler = _torch_sectorial_geometry(
        image.shape, params.na, torch
    )
    spectrum = torch.fft.fftshift(
        torch.fft.fftn(
            torch.fft.fftshift(torch.as_tensor(image, device="cuda"))
        )
    )
    radii = params.radii
    d0 = np.zeros((params.nr, params.na), dtype=np.float64)
    initial_kc = np.zeros(params.na, dtype=np.float64)
    initial_snr = np.zeros(params.na, dtype=np.float64)
    for first_sector in range(0, params.na, CUDA_CURVE_BATCH_SIZE):
        sectors = list(
            range(
                first_sector,
                min(first_sector + CUDA_CURVE_BATCH_SIZE, params.na),
            )
        )
        sector_spectra = spectrum * mask0 * masks[sectors]
        magnitude = torch.abs(sector_spectra)
        normalized = torch.where(
            magnitude > 0,
            sector_spectra / magnitude,
            torch.zeros_like(sector_spectra),
        )
        energies = torch.sqrt(
            torch.sum(
                torch.abs(sector_spectra) ** 2,
                dim=(1, 2),
                dtype=torch.float64,
            )
        )
        batch_curves = sampler.curves(
            sector_spectra,
            normalized,
            energies,
            np.tile(radii, (len(sectors), 1)),
        )
        for batch_index, sector in enumerate(sectors):
            d0[:, sector] = batch_curves[batch_index]
            peak, initial_snr[sector] = _local_max(
                d0[:, sector], params.dt
            )
            initial_kc[sector] = radii[peak]

    gmax = 1.0 / max(float(np.min(initial_kc)), np.finfo(float).eps)
    g = np.exp(np.linspace(np.log(gmax), np.log(0.14), params.ng))
    search_r = np.tile(radii, (params.na, 1))
    curves = np.full(
        (params.nr, params.na, params.ng * 2),
        np.nan if selected_sector is not None else 0.0,
        dtype=np.float64,
    )
    middle = params.na // 2
    active_sectors = (
        range(params.na) if selected_sector is None else (selected_sector,)
    )
    for refinement in range(2):
        stage_kc = np.zeros((params.na, params.ng), dtype=np.float64)
        stage_snr = np.zeros((params.na, params.ng), dtype=np.float64)
        highpasses = [
            1.0 - torch.exp(-2.0 * float(value) ** 2 * r2) for value in g
        ]
        pairs = [
            (sector, h)
            for sector in active_sectors
            for h in range(params.ng)
        ]
        for first_pair in range(0, len(pairs), CUDA_CURVE_BATCH_SIZE):
            batch_pairs = pairs[
                first_pair : first_pair + CUDA_CURVE_BATCH_SIZE
            ]
            filtered = torch.stack(
                [
                    spectrum
                    * mask0
                    * masks[sector]
                    * highpasses[h]
                    for sector, h in batch_pairs
                ]
            )
            magnitude = torch.abs(filtered)
            normalized = torch.where(
                magnitude > 0,
                filtered / magnitude,
                torch.zeros_like(filtered),
            )
            energies = torch.sqrt(
                torch.sum(
                    torch.abs(filtered) ** 2,
                    dim=(1, 2),
                    dtype=torch.float64,
                )
            )
            batch_curves = sampler.curves(
                filtered,
                normalized,
                energies,
                np.stack([search_r[sector] for sector, _ in batch_pairs]),
            )
            for batch_index, (sector, h) in enumerate(batch_pairs):
                curve = batch_curves[batch_index]
                curves[:, sector, h + refinement * params.ng] = curve
                peak, stage_snr[sector, h] = _local_max(curve, params.dt)
                stage_kc[sector, h] = search_r[sector, peak]
        if refinement == 0:
            metric = np.sqrt(
                np.maximum(stage_kc, 0) * np.maximum(stage_snr, 0)
            )
            if selected_sector is not None:
                best = int(np.argmax(metric[selected_sector]))
                low_g = max(0, best - 1)
                high_g = min(params.ng - 1, best + 1)
                g = np.exp(
                    np.linspace(np.log(g[low_g]), np.log(g[high_g]), params.ng)
                )
                center = stage_kc[selected_sector, best]
                search_r[selected_sector] = np.linspace(
                    max(0.0, center - params.rmin),
                    min(1.0, center + params.rmax),
                    params.nr,
                )
                continue
            best_by_sector = np.argmax(metric, axis=1)
            middle_best = int(best_by_sector[middle])
            low_g = max(0, middle_best - 1)
            high_g = min(params.ng - 1, middle_best + 1)
            g = np.exp(
                np.linspace(np.log(g[low_g]), np.log(g[high_g]), params.ng)
            )
            for sector in range(params.na):
                center = stage_kc[sector, best_by_sector[sector]]
                search_r[sector] = np.linspace(
                    max(0.0, center - params.rmin),
                    min(1.0, center + params.rmax),
                    params.nr,
                )

    if selected_sector is not None:
        sector_kc = stage_kc[selected_sector].copy()
        sector_snr = stage_snr[selected_sector]
        sector_kc[sector_snr < 0.05] = 0.0
        best = int(
            np.argmax(
                np.sqrt(np.maximum(sector_kc, 0) * np.maximum(sector_snr, 0))
            )
        )
        chosen = sector_kc[best]
        valid_value = (
            chosen > search_r[selected_sector, 0]
            and chosen < search_r[selected_sector, -1]
            and chosen > 0
        )
        kc = np.full(params.na, np.nan, dtype=np.float64)
        peak_snr = np.full(params.na, np.nan, dtype=np.float64)
        valid = np.zeros(params.na, dtype=bool)
        kc[selected_sector] = chosen if valid_value else np.nan
        peak_snr[selected_sector] = sector_snr[best]
        valid[selected_sector] = valid_value
    else:
        final_low = search_r[:, 0]
        final_high = search_r[:, -1]
        stage_kc[stage_snr < 0.05] = 0.0
        metric = np.sqrt(
            np.maximum(stage_kc, 0) * np.maximum(stage_snr, 0)
        )
        best = np.argmax(metric, axis=1)
        kc = stage_kc[np.arange(params.na), best]
        peak_snr = stage_snr[np.arange(params.na), best]
        valid = (kc > final_low) & (kc < final_high) & (kc > 0)
        kc = np.where(valid, kc, np.nan)
    return {
        "kcGM": kc,
        "A0": initial_snr,
        "peak_snr": peak_snr,
        "interior_valid": valid,
        "d0": d0,
        "curves": curves,
        "device": "cuda",
    }


def imdecorr_sectorial(
    image: np.ndarray,
    params: DecorrParameters | None = None,
    *,
    selected_sector: int | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    params = params or DecorrParameters()
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if selected_sector is not None and not 0 <= selected_sector < params.na:
        raise IndexError(f"Sector {selected_sector} outside 0..{params.na - 1}")
    if device == "cpu":
        return _imdecorr_sectorial_cpu(
            image, params, selected_sector=selected_sector
        )
    try:
        return _imdecorr_sectorial_cuda(image, params, selected_sector)
    except (ImportError, RuntimeError) as exc:
        if device == "cuda":
            raise
        warnings.warn(f"CUDA decorrelation unavailable; using CPU: {exc}")
        return _imdecorr_sectorial_cpu(
            image, params, selected_sector=selected_sector
        )


def resolution_from_kc(pixel_size_nm: float, kc: np.ndarray | float):
    values = np.asarray(kc, dtype=np.float64)
    result = np.divide(
        2.0 * pixel_size_nm,
        values,
        out=np.full_like(values, np.nan),
        where=np.isfinite(values) & (values > 0),
    )
    return float(result) if result.ndim == 0 else result


def _resize_rows_centered(strip: np.ndarray, target: int) -> np.ndarray:
    rows = strip.shape[0]
    if rows == target:
        return strip
    if rows < target:
        before = (target - rows) // 2
        result = np.zeros((target, strip.shape[1]), dtype=np.float32)
        result[before : before + rows] = strip
        return result
    first = (rows - target) // 2
    return strip[first : first + target]


def build_axial_montages(
    stack: np.ndarray, params: MontageParameters | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    params = params or MontageParameters()
    stack = np.asarray(stack, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        raise ValueError("Expected a Z x Y x X stack with at least two planes")
    depth, height, width = stack.shape
    strip_depth = depth + 2 * params.end_replication
    if min(height, width) < params.num_groups * params.num_points:
        raise ValueError("XY dimensions are too small for the montage sampling grid")
    if strip_depth > params.canvas_size:
        raise ValueError("Replicated Z width exceeds the montage canvas")
    rng = np.random.RandomState(params.seed)

    def build(axis: str):
        axis_length = height if axis == "xz" else width
        center_start = axis_length // 2 - params.num_points // 2
        centers = center_start + np.arange(params.num_groups)
        montages = np.zeros(
            (params.num_groups, params.canvas_size, params.canvas_size),
            dtype=np.float32,
        )
        samples = np.zeros((params.num_groups, params.num_points), dtype=np.int64)
        offsets = np.zeros_like(samples)
        half_span = params.num_groups * (params.num_points - 1) / 2
        for group, center in enumerate(centers):
            points = np.rint(
                center
                - half_span
                + np.arange(params.num_points) * params.num_groups
            ).astype(int)
            if np.any(points < 1) or np.any(points > axis_length):
                raise ValueError("Montage sampling indices are outside the stack")
            samples[group] = points
            canvas = np.zeros(
                (params.canvas_size, params.canvas_size), dtype=np.float32
            )
            for point_index, point in enumerate(points):
                if axis == "xz":
                    strip = stack[:, point - 1, :].T
                else:
                    strip = stack[:, :, point - 1].T
                strip = np.concatenate(
                    [
                        np.repeat(strip[:, :1], params.end_replication, axis=1),
                        strip,
                        np.repeat(strip[:, -1:], params.end_replication, axis=1),
                    ],
                    axis=1,
                )
                strip = _resize_rows_centered(
                    apodize_rect(strip, params.apodization), params.canvas_size
                )
                offset = int(
                    rng.randint(0, params.canvas_size - strip_depth + 1)
                )
                offsets[group, point_index] = offset + 1
                canvas[:, offset : offset + strip_depth] += strip
            montages[group] = np.rot90(canvas)
        return montages, samples, offsets

    xz, xz_samples, xz_offsets = build("xz")
    yz, yz_samples, yz_offsets = build("yz")
    info = {
        "seed": params.seed,
        "xz_sample_indices": xz_samples.tolist(),
        "yz_sample_indices": yz_samples.tolist(),
        "xz_offsets": xz_offsets.tolist(),
        "yz_offsets": yz_offsets.tolist(),
        "parameters": asdict(params),
    }
    return xz, yz, info


def local_grid(
    image: np.ndarray,
    pixel_size_nm: float,
    params: DecorrParameters | None = None,
    *,
    tile_size: int = 256,
    grid_count: int = 4,
) -> pd.DataFrame:
    params = params or DecorrParameters()
    height, width = image.shape
    step_y = (height - tile_size) // (grid_count - 1) - 1
    step_x = (width - tile_size) // (grid_count - 1) - 1
    if min(step_y, step_x) < 1:
        raise ValueError("Image is too small for the requested local grid")
    finite = image[np.isfinite(image)]
    full_contrast = (
        float(np.diff(np.percentile(finite, [0.5, 99.5]))[0])
        if finite.size
        else 0.0
    )
    rows = []
    for column in range(grid_count):
        for row in range(grid_count):
            y1 = row * step_y
            x1 = column * step_x
            y2 = y1 + tile_size + 1
            x2 = x1 + tile_size + 1
            tile = image[y1:y2, x1:x2]
            tile_finite = tile[np.isfinite(tile)]
            tile_contrast = (
                float(np.diff(np.percentile(tile_finite, [0.5, 99.5]))[0])
                if tile_finite.size
                else 0.0
            )
            included = (
                full_contrast > 0
                and tile_contrast >= SIGNAL_CONTRAST_RATIO * full_contrast
            )
            core = (
                imdecorr_global(apodize_rect(tile, 20), params)
                if included
                else {"kc": math.nan, "peak_snr": math.nan}
            )
            rows.append(
                {
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "y_first": y1 + 1,
                    "y_last": y2,
                    "x_first": x1 + 1,
                    "x_last": x2,
                    "kc": core["kc"],
                    "resolution_nm": resolution_from_kc(
                        pixel_size_nm, core["kc"]
                    ),
                    "peak_snr": core["peak_snr"],
                }
            )
    return pd.DataFrame(rows)


def analyze_file(
    input_file: str | Path,
    output_dir: str | Path,
    *,
    xy_pixel_size_nm: float,
    z_pixel_size_nm: float | None = None,
    seed: int = 20250124,
    audit_seeds: tuple[int, ...] = (),
    axial_device: str = "cpu",
) -> dict[str, object]:
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    lateral_dir = output_dir / "lateral"
    lateral_dir.mkdir(parents=True, exist_ok=True)
    stack = read_tiff_stack_float32(input_file)
    data_axes = "YX" if stack.shape[0] == 1 else "ZYX"
    if xy_pixel_size_nm <= 0:
        raise ValueError("XY sampling must be positive")
    if data_axes == "ZYX" and (z_pixel_size_nm is None or z_pixel_size_nm <= 0):
        raise ValueError("Positive Z sampling is required for ZYX input")
    params = DecorrParameters()

    plane, selection, scores = select_xy_plane(stack)
    write_tiff_float32(lateral_dir / "selected_plane.tif", plane)
    if data_axes == "ZYX":
        scores.to_csv(lateral_dir / "plane_scores.csv", index=False, encoding="utf-8")
    global_result = imdecorr_global(plane, params)
    np.savez_compressed(
        lateral_dir / "lateral_diagnostics.npz",
        d0=global_result["d0"],
        curves=global_result["curves"],
    )
    local = local_grid(plane, xy_pixel_size_nm, params)
    local.to_csv(
        lateral_dir / "lateral_local_results.csv", index=False, encoding="utf-8"
    )
    lateral = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "input_file": str(input_file),
        "data_axes": data_axes,
        "selected_plane": (
            selection["selected_index"] if data_axes == "ZYX" else None
        ),
        "selection_mode": selection["mode"] if data_axes == "ZYX" else "not_applicable",
        "top_candidate_indices": ";".join(map(str, selection["top_indices"])),
        "top_candidate_scores": ";".join(
            f"{value:.9g}" for value in selection["top_scores"]
        ),
        "pixel_size_nm": xy_pixel_size_nm,
        "kc": global_result["kc"],
        "resolution_nm": resolution_from_kc(
            xy_pixel_size_nm, global_result["kc"]
        ),
        "local_valid_tiles": int(local["resolution_nm"].notna().sum()),
        "local_median_resolution_nm": float(
            local["resolution_nm"].median(skipna=True)
        ),
        "device": "cpu",
        "axial_status": "pending" if data_axes == "ZYX" else "not_applicable",
        "status": "ok",
    }
    pd.DataFrame([lateral]).to_csv(
        lateral_dir / "lateral_result.csv", index=False, encoding="utf-8"
    )

    if data_axes == "YX":
        return lateral
    if stack.shape[0] < MIN_AXIAL_PLANES:
        lateral["axial_status"] = "insufficient_native_z"
        return lateral

    axial_dir = output_dir / "axial"
    axial_dir.mkdir(parents=True, exist_ok=True)

    def measure_axial(current_seed: int, *, save_diagnostics: bool):
        montage_params = MontageParameters(seed=current_seed)
        xz, yz, montage_info = build_axial_montages(stack, montage_params)
        sector_index = params.na // 2
        axial_rows = []
        xz_d0 = []
        yz_d0 = []
        xz_curves = []
        yz_curves = []
        for group in range(montage_params.num_groups):
            selected_sector = None if save_diagnostics else sector_index
            xz_core = imdecorr_sectorial(
                apodize_rect(xz[group], 20),
                params,
                selected_sector=selected_sector,
                device=axial_device,
            )
            yz_core = imdecorr_sectorial(
                apodize_rect(yz[group], 20),
                params,
                selected_sector=selected_sector,
                device=axial_device,
            )
            xz_kc = float(xz_core["kcGM"][sector_index])
            yz_kc = float(yz_core["kcGM"][sector_index])
            axial_rows.append(
                {
                    "input_file": str(input_file),
                    "group_index": group + 1,
                    "axial_pixel_size_nm": z_pixel_size_nm,
                    "seed": current_seed,
                    "xz_kc": xz_kc,
                    "xz_resolution_nm": resolution_from_kc(z_pixel_size_nm, xz_kc),
                    "xz_valid": bool(xz_core["interior_valid"][sector_index]),
                    "yz_kc": yz_kc,
                    "yz_resolution_nm": resolution_from_kc(z_pixel_size_nm, yz_kc),
                    "yz_valid": bool(yz_core["interior_valid"][sector_index]),
                    "device": (
                        xz_core["device"]
                        if xz_core["device"] == yz_core["device"]
                        else "mixed"
                    ),
                }
            )
            if save_diagnostics:
                xz_d0.append(xz_core["d0"])
                yz_d0.append(yz_core["d0"])
                xz_curves.append(xz_core["curves"])
                yz_curves.append(yz_core["curves"])
        return (
            pd.DataFrame(axial_rows),
            xz,
            yz,
            montage_info,
            np.asarray(xz_d0),
            np.asarray(yz_d0),
            np.asarray(xz_curves),
            np.asarray(yz_curves),
        )

    (
        axial,
        xz,
        yz,
        montage_info,
        xz_d0,
        yz_d0,
        xz_curves,
        yz_curves,
    ) = measure_axial(seed, save_diagnostics=True)
    write_tiff_float32(axial_dir / "xz_montage.tif", xz)
    write_tiff_float32(axial_dir / "yz_montage.tif", yz)
    axial.to_csv(axial_dir / "axial_result.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        axial_dir / "axial_diagnostics.npz",
        xz_d0=xz_d0,
        yz_d0=yz_d0,
        xz_curves=xz_curves,
        yz_curves=yz_curves,
    )
    with (axial_dir / "montage_info.json").open("w", encoding="utf-8") as handle:
        json.dump(montage_info, handle, ensure_ascii=False, indent=2)
    main_xz = float(axial["xz_resolution_nm"].median(skipna=True))
    main_yz = float(axial["yz_resolution_nm"].median(skipna=True))
    stability_rows = [
        {
            "seed": seed,
            "xz_median_resolution_nm": main_xz,
            "yz_median_resolution_nm": main_yz,
            "max_relative_deviation": 0.0,
            "status": "main",
        }
    ]
    for audit_seed in audit_seeds:
        audit = measure_axial(audit_seed, save_diagnostics=False)[0]
        audit_xz = float(audit["xz_resolution_nm"].median(skipna=True))
        audit_yz = float(audit["yz_resolution_nm"].median(skipna=True))
        deviation = max(
            abs(audit_xz - main_xz) / main_xz,
            abs(audit_yz - main_yz) / main_yz,
        )
        stability_rows.append(
            {
                "seed": audit_seed,
                "xz_median_resolution_nm": audit_xz,
                "yz_median_resolution_nm": audit_yz,
                "max_relative_deviation": deviation,
                "status": (
                    "stable"
                    if deviation <= 0.10
                    else "caution"
                    if deviation <= 0.20
                    else "unstable"
                ),
            }
        )
    pd.DataFrame(stability_rows).to_csv(
        axial_dir / "axial_seed_stability.csv", index=False, encoding="utf-8"
    )
    return {
        **lateral,
        "axial_status": "ok",
        "axial_device": (
            axial["device"].iloc[0]
            if axial["device"].nunique() == 1
            else "mixed"
        ),
        "xz_valid_groups": int(axial["xz_valid"].sum()),
        "xz_median_resolution_nm": main_xz,
        "yz_valid_groups": int(axial["yz_valid"].sum()),
        "yz_median_resolution_nm": main_yz,
    }


def default_output_path(
    input_path: str | Path,
    now: datetime | None = None,
) -> Path:
    source = Path(input_path).resolve()
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    result_parent = source.parent.parent / "decorr-results"
    candidate = result_parent / f"{source.stem}-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = result_parent / f"{source.stem}-{timestamp}-{suffix:02d}"
        suffix += 1
    return candidate


def prepare_output_path(
    input_path: str | Path,
    requested_output: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> Path:
    output = (
        Path(requested_output).resolve()
        if requested_output is not None
        else default_output_path(input_path, now)
    )
    output.mkdir(parents=True, exist_ok=False)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="One reconstructed TIFF file")
    parser.add_argument(
        "--output",
        type=Path,
        help="New result directory; defaults beside the input data folder",
    )
    parser.add_argument("--xy-pixel-nm", type=float, required=True)
    parser.add_argument("--z-pixel-nm", type=float)
    parser.add_argument("--seed", type=int, default=20250124)
    parser.add_argument("--audit-seeds", type=int, nargs="*", default=())
    parser.add_argument(
        "--axial-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = prepare_output_path(args.input, args.output)
    print(f"input={args.input.resolve()}", flush=True)
    print(f"output={output}", flush=True)
    started = time.perf_counter()
    result = analyze_file(
        args.input,
        output,
        xy_pixel_size_nm=args.xy_pixel_nm,
        z_pixel_size_nm=args.z_pixel_nm,
        seed=args.seed,
        audit_seeds=tuple(args.audit_seeds),
        axial_device=args.axial_device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"elapsed_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
