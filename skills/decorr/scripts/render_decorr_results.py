"""Render decorrelation result figures from existing TIFF and CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Circle, Rectangle
from scipy.io import loadmat

from analyze_axial_peak_qc import (
    SPECTRAL_SNR_MIN_DB,
    analyze_batch as analyze_axial_peak_batch,
)
from decorr_pipeline import DecorrParameters, apodize_rect, imdecorr_global


PEAK_QC_SHEET = "轴向峰值QC"


def _single_match(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.8])
    if high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def _finite_limits(values: list[np.ndarray]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    finite = np.concatenate(
        [np.asarray(value, dtype=np.float64).ravel() for value in values]
    )
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(finite)), float(np.max(finite))
    return (low, high if high > low else low + 1.0)


def _spectrum(image: np.ndarray) -> np.ndarray:
    return np.log1p(
        np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(image))))
    )


def _render_axial_spectrum(
    axis: plt.Axes,
    montage: np.ndarray,
    kc: float,
    direction: str,
) -> None:
    spectrum = _spectrum(montage)
    axis.imshow(
        spectrum,
        cmap="magma",
        origin="lower",
        extent=(-1, 1, -1, 1),
    )
    axis.axhline(kc, color="#4DE3FF", linewidth=1.4, label=f"{direction} kc")
    axis.axhline(-kc, color="#4DE3FF", linewidth=1.4)
    axis.set_title(f"{direction} Fourier spectrum")
    axis.set_xlabel("Transverse frequency")
    axis.set_ylabel("Normalized z frequency")
    axis.legend(loc="upper right")


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


def _normalized_radial_average(
    spectrum: np.ndarray, sample_count: int
) -> np.ndarray:
    height, width = spectrum.shape
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, width),
        np.linspace(-1.0, 1.0, height),
    )
    radius = np.sqrt(x * x + y * y)
    edges = np.linspace(0.0, 1.0, sample_count + 1)
    radial = np.zeros(sample_count, dtype=np.float64)
    for index in range(sample_count):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        radial[index] = np.mean(spectrum[mask]) if np.any(mask) else np.nan
    valid = np.isfinite(radial)
    radial = np.interp(
        np.arange(sample_count),
        np.flatnonzero(valid),
        radial[valid],
    )
    if sample_count > 1:
        radial[0] = radial[1]
        radial[-1] = radial[-2]
    low, high = float(np.min(radial)), float(np.max(radial))
    if high <= low:
        return np.zeros_like(radial)
    return (radial - low) / (high - low)


def _paper_curve_data(
    core: object, sector: int, ng: int, dt: float, rmin: float, rmax: float
) -> dict[str, np.ndarray | float]:
    curves = np.asarray(core.curves, dtype=np.float64)
    d0 = np.asarray(core.d0, dtype=np.float64)
    radii = np.linspace(0.0, 1.0, curves.shape[0])
    first_curves = curves[:, sector, :ng]
    second_curves = curves[:, sector, ng : 2 * ng]

    first_kc = np.zeros(ng, dtype=np.float64)
    first_snr = np.zeros(ng, dtype=np.float64)
    for index in range(ng):
        peak, first_snr[index] = _local_max(first_curves[:, index], dt)
        first_kc[index] = radii[peak]
    metric = np.sqrt(np.maximum(first_kc, 0) * np.maximum(first_snr, 0))
    refinement_kc = float(first_kc[int(np.argmax(metric))])
    refined_radii = np.linspace(
        max(0.0, refinement_kc - rmin),
        min(1.0, refinement_kc + rmax),
        curves.shape[0],
    )
    return {
        "radii": radii,
        "refined_radii": refined_radii,
        "first_curves": first_curves,
        "second_curves": second_curves,
        "d0": d0[:, sector],
        "refinement_kc": refinement_kc,
    }


def render_axial_paper(
    sample_root: Path,
    figure_dir: Path,
    result: pd.DataFrame,
    montage_stack: np.ndarray,
    representative: int,
    direction: str,
) -> None:
    diagnostics_path = sample_root / "axial" / "axial_diagnostics.npz"
    if diagnostics_path.is_file():
        diagnostics = np.load(diagnostics_path)
        defaults = DecorrParameters()
        core = SimpleNamespace(
            curves=diagnostics[f"{direction.lower()}_curves"][representative],
            d0=diagnostics[f"{direction.lower()}_d0"][representative],
        )
        params = SimpleNamespace(
            Ng=defaults.ng,
            Na=defaults.na,
            Nr=defaults.nr,
            dt=defaults.dt,
            rmin=defaults.rmin,
            rmax=defaults.rmax,
        )
    else:
        mat = loadmat(
            _single_match(sample_root / "axial", "axial_result.mat"),
            squeeze_me=True,
            struct_as_record=False,
        )
        cores = np.atleast_1d(mat[f"{direction.lower()}Core"])
        params = mat["params"]
        core = cores[representative]
    ng = int(params.Ng)
    na = int(params.Na)
    nr = int(params.Nr)
    sector = na // 2
    curve_data = _paper_curve_data(
        core,
        sector=sector,
        ng=ng,
        dt=float(params.dt),
        rmin=float(params.rmin),
        rmax=float(params.rmax),
    )

    montage = montage_stack[representative]
    spectrum = _spectrum(montage)
    kc = float(result.loc[representative, f"{direction.lower()}_kc"])
    pixel_size_nm = float(result.loc[representative, "axial_pixel_size_nm"])
    resolution_nm = 2.0 * pixel_size_nm / kc
    radial_average = _normalized_radial_average(spectrum, nr)

    figure = plt.figure(figsize=(13.4, 4.35), constrained_layout=False)
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.05, 1.05, 1.0),
        left=0.065,
        right=0.985,
        bottom=0.16,
        top=0.83,
        wspace=0.38,
    )
    image_axis = figure.add_subplot(grid[0, 0])
    spectrum_axis = figure.add_subplot(grid[0, 1])
    curve_axis = figure.add_subplot(grid[0, 2])
    figure.text(0.035, 0.88, "e", fontsize=20, fontweight="bold")
    figure.text(
        0.73,
        0.98,
        "Normalized spatial frequencies",
        ha="center",
        va="top",
        fontsize=12,
    )

    low, high = _display_limits(montage)
    image_axis.imshow(montage, cmap="gray", vmin=low, vmax=high)
    image_axis.text(
        0.05,
        0.96,
        f"3D-MP-SIM {direction}",
        transform=image_axis.transAxes,
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
    )
    image_axis.plot(
        [0.88, 0.88],
        [0.08, 0.28],
        transform=image_axis.transAxes,
        color="white",
        linewidth=2.2,
    )
    image_axis.text(
        0.88,
        0.31,
        "z",
        transform=image_axis.transAxes,
        color="white",
        fontsize=12,
        fontstyle="italic",
        ha="center",
        va="bottom",
    )
    image_axis.set_axis_off()

    spectrum_axis.imshow(
        spectrum,
        cmap="viridis",
        extent=(-1, 1, 1, -1),
        aspect="equal",
        vmin=float(np.percentile(spectrum, 5)),
        vmax=float(np.percentile(spectrum, 99)),
    )
    spectrum_axis.annotate(
        "",
        xy=(0, -kc),
        xytext=(0, 0),
        arrowprops={
            "arrowstyle": "<->",
            "color": "white",
            "linewidth": 1.8,
            "mutation_scale": 14,
        },
    )
    spectrum_axis.text(
        0,
        0.22,
        f"kcGM = {kc:.3f}",
        color="white",
        fontsize=12,
        ha="center",
        va="center",
    )
    spectrum_axis.plot([0.72, 0.72], [0.69, 0.9], color="white", linewidth=2.2)
    spectrum_axis.text(
        0.72,
        0.64,
        r"$k_z$",
        color="white",
        fontsize=12,
        ha="center",
        va="bottom",
    )
    spectrum_axis.set_xlim(-1, 1)
    spectrum_axis.set_ylim(1, -1)
    spectrum_axis.set_xticks([-1, -0.5, 0, 0.5, 1])
    spectrum_axis.set_yticks([-1, -0.5, 0, 0.5, 1])
    spectrum_axis.tick_params(labelsize=8)

    radii = np.asarray(curve_data["radii"])
    refined_radii = np.asarray(curve_data["refined_radii"])
    for curve in np.asarray(curve_data["first_curves"]).T:
        curve_axis.plot(radii, curve, color="black", linewidth=0.9)
    for curve in np.asarray(curve_data["second_curves"]).T:
        curve_axis.plot(refined_radii, curve, color="blue", linewidth=0.9)
    curve_axis.plot(
        radii,
        np.asarray(curve_data["d0"]),
        color="#00D000",
        linewidth=1.7,
    )
    curve_axis.plot(
        np.linspace(0.0, 1.0, nr),
        radial_average,
        color="magenta",
        linewidth=1.7,
    )
    curve_axis.axvline(kc, color="black", linewidth=1.25)
    curve_axis.axvline(
        float(curve_data["refinement_kc"]),
        color="#F0D000",
        linewidth=1.25,
    )
    curve_axis.set_xlim(0, 1)
    curve_axis.set_ylim(0, 1)
    curve_axis.set_xticks([0, 1])
    curve_axis.set_yticks([0, 1])
    curve_axis.set_xlabel("Normalized spatial frequencies", fontsize=11)
    curve_axis.set_ylabel("Cross-correlation coefficients", fontsize=11)
    curve_axis.set_title(
        f"kc: {kc:.3f}, resolution: {resolution_nm:.0f} nm",
        fontsize=11,
        pad=6,
    )
    curve_axis.tick_params(labelsize=10)

    output = figure_dir / "axial_decorrelation_paper.png"
    figure.savefig(output, dpi=300, facecolor="white")
    plt.close(figure)


def render_lateral_paper(
    image: np.ndarray,
    local: pd.DataFrame,
    pixel_size_nm: float,
    figure_dir: Path,
) -> str:
    valid = local.dropna(subset=["resolution_nm"])
    if valid.empty:
        raise RuntimeError("No signal-supported lateral tile is available")
    median = float(valid["resolution_nm"].median())
    row = valid.loc[(valid["resolution_nm"] - median).abs().idxmin()]
    tile = image[
        int(row.y_first) - 1 : int(row.y_last),
        int(row.x_first) - 1 : int(row.x_last),
    ]
    core = imdecorr_global(apodize_rect(tile, 20))
    kc = float(core["kc"])
    resolution = 2.0 * pixel_size_nm / kc
    params = DecorrParameters()
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.35), constrained_layout=True)
    low, high = _display_limits(tile)
    axes[0].imshow(tile, cmap="gray", vmin=low, vmax=high)
    region = f"R{int(row.grid_row)}C{int(row.grid_column)}"
    axes[0].set_title(region)
    axes[0].set_axis_off()
    spectrum = _spectrum(tile)
    axes[1].imshow(
        spectrum,
        cmap="viridis",
        origin="lower",
        extent=(-1, 1, -1, 1),
        vmin=float(np.percentile(spectrum, 5)),
        vmax=float(np.percentile(spectrum, 99)),
    )
    axes[1].add_patch(Circle((0, 0), kc, fill=False, edgecolor="white", linewidth=1.5))
    axes[1].set_title(f"kc={kc:.3f}, {resolution:.0f} nm")
    curves = np.asarray(core["curves"])
    for curve in curves[:, : params.ng].T:
        axes[2].plot(params.radii, curve, color="black", linewidth=0.7)
    refined_radii = np.asarray(core.get("refined_radii", params.radii))
    for curve in curves[:, params.ng :].T:
        axes[2].plot(refined_radii, curve, color="blue", linewidth=0.7)
    axes[2].plot(params.radii, core["d0"], color="#00A83B", linewidth=1.4)
    axes[2].axvline(kc, color="black", linewidth=1.1)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized spatial frequency")
    axes[2].set_ylabel("Cross-correlation coefficient")
    figure.savefig(figure_dir / "lateral_decorrelation_paper.png", dpi=300)
    plt.close(figure)
    return region


def render_lateral(
    sample_root: Path,
    figure_dir: Path,
    *,
    resolution_limits: tuple[float, float],
) -> dict[str, object]:
    result = pd.read_csv(_single_match(sample_root / "lateral", "lateral_result.csv"))
    local = pd.read_csv(
        _single_match(sample_root / "lateral", "lateral_local_results.csv")
    )
    image = np.asarray(
        tifffile.imread(_single_match(sample_root / "lateral", "selected_plane.tif")),
        dtype=np.float32,
    )
    if image.ndim == 3:
        image = image[0]
    kc = float(result.loc[0, "kc"])
    resolution = float(result.loc[0, "resolution_nm"])
    selected_value = result.loc[0, "selected_plane"]
    selected_plane = int(selected_value) if pd.notna(selected_value) else None
    local_map = local.pivot(
        index="grid_row", columns="grid_column", values="resolution_nm"
    ).to_numpy()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    low, high = _display_limits(image)
    axes[0].imshow(image, cmap="gray", vmin=low, vmax=high)
    axes[0].set_title(
        f"Selected XY plane Z{selected_plane}" if selected_plane else "Input YX image"
    )
    axes[0].set_axis_off()
    for row in local.itertuples(index=False):
        axes[0].add_patch(
            Rectangle(
                (row.x_first - 1, row.y_first - 1),
                row.x_last - row.x_first + 1,
                row.y_last - row.y_first + 1,
                fill=False,
                edgecolor="#33D1C6",
                linewidth=0.45,
                alpha=0.55,
            )
        )

    spectrum = _spectrum(image)
    axes[1].imshow(
        spectrum,
        cmap="magma",
        origin="lower",
        extent=(-1, 1, -1, 1),
    )
    axes[1].add_patch(
        Circle((0, 0), kc, fill=False, edgecolor="#4DE3FF", linewidth=1.8)
    )
    axes[1].set_title(f"Fourier spectrum, kc={kc:.4f}\n{resolution:.2f} nm")
    axes[1].set_xlabel("Normalized frequency")
    axes[1].set_ylabel("Normalized frequency")

    heatmap = axes[2].imshow(
        local_map,
        cmap="viridis_r",
        origin="upper",
        vmin=resolution_limits[0],
        vmax=resolution_limits[1],
    )
    for row in range(local_map.shape[0]):
        for column in range(local_map.shape[1]):
            value = local_map[row, column]
            axes[2].text(
                column,
                row,
                f"{value:.1f}" if np.isfinite(value) else "N/A",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    axes[2].set_title(
        "Local resolution "
        f"(nm; batch scale {resolution_limits[0]:.1f}-{resolution_limits[1]:.1f})"
        f"\nmedian={np.nanmedian(local_map):.2f}"
    )
    axes[2].set_xlabel("Grid column")
    axes[2].set_ylabel("Grid row")
    axes[2].set_xticks(range(local_map.shape[1]), range(1, local_map.shape[1] + 1))
    axes[2].set_yticks(range(local_map.shape[0]), range(1, local_map.shape[0] + 1))
    figure.colorbar(heatmap, ax=axes[2], label="nm", shrink=0.82)
    output = figure_dir / "lateral_decorrelation_summary.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    paper_region = render_lateral_paper(
        image, local, float(result.loc[0, "pixel_size_nm"]), figure_dir
    )
    return {
        "selected_plane": selected_plane,
        "lateral_global_resolution_nm": resolution,
        "lateral_median_resolution_nm": float(np.nanmedian(local_map)),
        "lateral_valid_tiles": int(np.isfinite(local_map).sum()),
        "lateral_paper_region": paper_region,
    }


def render_axial(sample_root: Path, figure_dir: Path) -> dict[str, object]:
    result = pd.read_csv(_single_match(sample_root / "axial", "axial_result.csv"))
    xz = np.asarray(
        tifffile.imread(_single_match(sample_root / "axial", "xz_montage.tif")),
        dtype=np.float32,
    )
    yz = np.asarray(
        tifffile.imread(_single_match(sample_root / "axial", "yz_montage.tif")),
        dtype=np.float32,
    )
    if xz.ndim == 2:
        xz = xz[np.newaxis, ...]
    if yz.ndim == 2:
        yz = yz[np.newaxis, ...]
    candidates = pd.concat(
        [
            pd.DataFrame(
                {
                    "direction": "XZ",
                    "group": result.index,
                    "resolution": result["xz_resolution_nm"],
                }
            ),
            pd.DataFrame(
                {
                    "direction": "YZ",
                    "group": result.index,
                    "resolution": result["yz_resolution_nm"],
                }
            ),
        ],
        ignore_index=True,
    ).dropna()
    axial_median = float(candidates["resolution"].median())
    chosen = candidates.loc[(candidates["resolution"] - axial_median).abs().idxmin()]
    direction = str(chosen["direction"])
    representative = int(chosen["group"])
    render_axial_paper(
        sample_root,
        figure_dir,
        result,
        xz if direction == "XZ" else yz,
        representative,
        direction,
    )
    paper_figure_written = True

    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    axes = np.empty((2, 2), dtype=object)
    axes[0, 0] = figure.add_subplot(grid[0, 0])
    axes[0, 1] = figure.add_subplot(grid[0, 1])
    axes[1, 0] = figure.add_subplot(grid[1, 0])
    spectrum_grid = grid[1, 1].subgridspec(1, 2)
    axes[1, 1] = figure.add_subplot(spectrum_grid[0, 0])
    yz_spectrum_axis = figure.add_subplot(spectrum_grid[0, 1])
    for axis, montage, label in (
        (axes[0, 0], xz[representative], "XZ"),
        (axes[0, 1], yz[representative], "YZ"),
    ):
        low, high = _display_limits(montage)
        axis.imshow(montage, cmap="gray", vmin=low, vmax=high)
        axis.set_title(f"{label} montage, group {representative + 1}")
        axis.set_axis_off()

    groups = result["group_index"].to_numpy()
    axes[1, 0].plot(
        groups,
        result["xz_resolution_nm"],
        marker="o",
        linewidth=1.4,
        label="XZ",
        color="#157A6E",
    )
    axes[1, 0].plot(
        groups,
        result["yz_resolution_nm"],
        marker="s",
        linewidth=1.4,
        label="YZ",
        color="#C65D35",
    )
    axes[1, 0].axhline(
        result["xz_resolution_nm"].median(),
        color="#157A6E",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 0].axhline(
        result["yz_resolution_nm"].median(),
        color="#C65D35",
        linestyle="--",
        linewidth=1,
    )
    axes[1, 0].set_title("Axial resolution across 12 montage groups")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Resolution (nm)")
    axes[1, 0].set_xticks(groups)
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend()

    xz_kc = float(result.loc[representative, "xz_kc"])
    yz_kc = float(result.loc[representative, "yz_kc"])
    _render_axial_spectrum(
        axes[1, 1], xz[representative], xz_kc, "XZ"
    )
    _render_axial_spectrum(
        yz_spectrum_axis, yz[representative], yz_kc, "YZ"
    )

    output = figure_dir / "axial_decorrelation_summary.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return {
        "xz_median_resolution_nm": float(result["xz_resolution_nm"].median()),
        "yz_median_resolution_nm": float(result["yz_resolution_nm"].median()),
        "axial_median_resolution_nm": axial_median,
        "representative_group": representative + 1,
        "representative_direction": direction,
        "paper_figure_written": paper_figure_written,
    }


def render_contact_pages(
    figures: list[tuple[str, Path]],
    output_dir: Path,
    *,
    rows: int,
    columns: int,
    prefix: str,
) -> int:
    page_size = rows * columns
    for page, first in enumerate(range(0, len(figures), page_size), start=1):
        figure, axes = plt.subplots(
            rows, columns, figsize=(columns * 8, rows * 5), constrained_layout=True
        )
        flat = np.asarray(axes).reshape(-1)
        for axis in flat:
            axis.set_axis_off()
        for axis, (sample, path) in zip(flat, figures[first : first + page_size]):
            axis.imshow(plt.imread(path))
            axis.set_title(sample.replace("、", "-"), fontsize=9)
            axis.set_axis_off()
        figure.savefig(output_dir / f"{prefix}-{page:02d}.png", dpi=120)
        plt.close(figure)
    return (len(figures) + page_size - 1) // page_size


def write_index(output_root: Path, summary: pd.DataFrame) -> Path:
    lateral = summary["lateral_median_resolution_nm"].dropna()
    axial = summary["axial_median_resolution_nm"].dropna()
    axial_6db = summary["axial_6db_median_resolution_nm"].dropna()
    lines = [
        "# Decorrelation Results",
        "",
        "![Batch summary](batch_decorrelation_summary.png)",
        "",
        "| Metric | n | Median (nm) | IQR (nm) |",
        "|---|---:|---:|---:|",
        f"| Lateral | {len(lateral)} | {lateral.median():.2f} | {(lateral.quantile(.75)-lateral.quantile(.25)):.2f} |",
        f"| Axial raw | {len(axial)} | {axial.median():.2f} | {(axial.quantile(.75)-axial.quantile(.25)):.2f} |",
        (
            f"| Axial after 6 dB | {len(axial_6db)} | "
            f"{axial_6db.median():.2f} | "
            f"{(axial_6db.quantile(.75)-axial_6db.quantile(.25)):.2f} |"
            if not axial_6db.empty
            else "| Axial after 6 dB | 0 | N/A | N/A |"
        ),
        "",
        (
            "[Excel results](decorr_results.xlsx) | "
            if (output_root / "decorr_results.xlsx").is_file()
            else ""
        )
        + "[Lateral QC images](lateral_qc_pages/) | "
        "[Axial QC images](axial_qc_pages/) | "
        "[Axial peak QC](axial-peak-qc/)",
        "",
        "| Sample | Lateral (nm) | Axial raw (nm) | Axial after 6 dB (nm) | Valid groups | Figures |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        axial_value = row.axial_median_resolution_nm
        axial_6db_value = row.axial_6db_median_resolution_nm
        axial_text = f"{axial_value:.2f}" if np.isfinite(axial_value) else "N/A"
        axial_6db_text = (
            f"{axial_6db_value:.2f}"
            if np.isfinite(axial_6db_value)
            else "N/A"
        )
        links = (
            f"[Lateral QC]({row.lateral_qc_path}), "
            f"[Lateral paper]({row.lateral_paper_path})"
        )
        if np.isfinite(axial_value):
            links += (
                f", [Axial QC]({row.axial_qc_path}), "
                f"[Axial paper]({row.axial_paper_path})"
            )
        lines.append(
            f"| {row.sample_name} | {row.lateral_median_resolution_nm:.2f} | "
            f"{axial_text} | {axial_6db_text} | "
            f"{row.axial_6db_valid_groups} | {links} |"
        )
    output = output_root / "INDEX.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _append_peak_qc_sheet(output_root: Path, table: pd.DataFrame) -> None:
    workbook = output_root / "decorr_results.xlsx"
    if not workbook.is_file():
        return
    with pd.ExcelWriter(
        workbook,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        table.to_excel(writer, sheet_name=PEAK_QC_SHEET, index=False)


def _add_six_db_summary(
    summary: pd.DataFrame, peak_qc: pd.DataFrame
) -> pd.DataFrame:
    supported = peak_qc[
        peak_qc["spectral_snr_db"] >= SPECTRAL_SNR_MIN_DB
    ]
    result = summary.copy()
    for direction in ("XZ", "YZ"):
        medians = (
            supported[supported["direction"] == direction]
            .groupby("sample")["baseline_resolution_nm"]
            .median()
        )
        result[f"{direction.lower()}_6db_median_resolution_nm"] = (
            result["sample_name"].map(medians)
        )
    axial_medians = supported.groupby("sample")["baseline_resolution_nm"].median()
    valid_groups = supported.groupby("sample").size()
    result["axial_6db_median_resolution_nm"] = result["sample_name"].map(
        axial_medians
    )
    result["axial_6db_valid_groups"] = (
        result["sample_name"].map(valid_groups).fillna(0).astype(int)
    )
    return result


def render_batch(output_root: Path) -> pd.DataFrame:
    rows = []
    lateral_figures = []
    axial_figures = []
    lateral_qc_dir = output_root / "lateral_qc_pages"
    axial_qc_dir = output_root / "axial_qc_pages"
    lateral_paper_dir = output_root / "lateral_decorrelation_paper"
    axial_paper_dir = output_root / "axial_decorrelation_paper"
    for directory in (
        lateral_qc_dir,
        axial_qc_dir,
        lateral_paper_dir,
        axial_paper_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.png"):
            path.unlink()
    for pattern in ("lateral_qc_page-*.png", "axial_qc_page-*.png"):
        for path in output_root.glob(pattern):
            path.unlink()
    if (output_root / "lateral" / "lateral_result.csv").is_file():
        sample_roots = [output_root]
    else:
        sample_roots = sorted(
            (output_root / "samples").iterdir(), key=lambda path: path.name
        )
    lateral_values = [
        pd.read_csv(
            _single_match(sample_root / "lateral", "lateral_local_results.csv")
        )["resolution_nm"].to_numpy()
        for sample_root in sample_roots
        if sample_root.is_dir()
    ]
    lateral_resolution_limits = _finite_limits(lateral_values)
    for index, sample_root in enumerate(sample_roots, start=1):
        if not sample_root.is_dir():
            continue
        print(f"[{index}/{len(sample_roots)}] {sample_root.name}", flush=True)
        figure_dir = sample_root / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "lateral_decorrelation_summary.png",
            "lateral_decorrelation_paper.png",
            "axial_decorrelation_summary.png",
            "axial_decorrelation_paper.png",
        ):
            (figure_dir / name).unlink(missing_ok=True)
        sample_figure_name = f"{sample_root.name}.png"
        row = {
            "sample_name": sample_root.name,
            "lateral_qc_path": f"lateral_qc_pages/{sample_figure_name}",
            "lateral_paper_path": (
                f"lateral_decorrelation_paper/{sample_figure_name}"
            ),
        }
        row.update(
            render_lateral(
                sample_root,
                figure_dir,
                resolution_limits=lateral_resolution_limits,
            )
        )
        lateral_qc = lateral_qc_dir / sample_figure_name
        lateral_paper = lateral_paper_dir / sample_figure_name
        (figure_dir / "lateral_decorrelation_summary.png").replace(lateral_qc)
        (figure_dir / "lateral_decorrelation_paper.png").replace(lateral_paper)
        lateral_figures.append(
            (sample_root.name, lateral_qc)
        )
        if (sample_root / "axial" / "axial_result.csv").is_file():
            row.update(render_axial(sample_root, figure_dir))
            axial_qc = axial_qc_dir / sample_figure_name
            axial_paper = axial_paper_dir / sample_figure_name
            (figure_dir / "axial_decorrelation_summary.png").replace(axial_qc)
            (figure_dir / "axial_decorrelation_paper.png").replace(axial_paper)
            row.update(
                {
                    "axial_qc_path": f"axial_qc_pages/{sample_figure_name}",
                    "axial_paper_path": (
                        f"axial_decorrelation_paper/{sample_figure_name}"
                    ),
                }
            )
            axial_figures.append(
                (sample_root.name, axial_qc)
            )
        else:
            row.update(
                {
                    "xz_median_resolution_nm": np.nan,
                    "yz_median_resolution_nm": np.nan,
                    "axial_median_resolution_nm": np.nan,
                    "paper_figure_written": False,
                    "axial_qc_path": "",
                    "axial_paper_path": "",
                }
            )
        try:
            figure_dir.rmdir()
        except OSError:
            pass
        rows.append(row)

    summary = pd.DataFrame(rows)
    is_batch = (output_root / "samples").is_dir()
    if is_batch and axial_figures:
        peak_qc_dir = output_root / "axial-peak-qc"
        analyze_axial_peak_batch(output_root, peak_qc_dir)
        peak_qc = pd.read_csv(peak_qc_dir / "axial_peak_qc.csv")
        _append_peak_qc_sheet(output_root, peak_qc)
        summary = _add_six_db_summary(summary, peak_qc)
    else:
        summary["xz_6db_median_resolution_nm"] = np.nan
        summary["yz_6db_median_resolution_nm"] = np.nan
        summary["axial_6db_median_resolution_nm"] = np.nan
        summary["axial_6db_valid_groups"] = 0
    summary.to_csv(
        output_root / "decorrelation_figure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    x = np.arange(len(summary))
    plot_labels = summary["sample_name"].str.replace("、", "-", regex=False)
    figure = plt.figure(figsize=(20, 12), constrained_layout=True)
    figure.suptitle(
        "Batch decorrelation summary\n"
        "6 dB spectral-support filter: "
        "10 log10[P(kc)/Pnoise] >= 6 (P ratio >= 3.98). "
        "Workflow QC gate, not an original decorrelation-paper threshold.",
        fontsize=11,
    )
    grid = figure.add_gridspec(2, 2)
    lateral_axis = figure.add_subplot(grid[0, 0])
    box_axis = figure.add_subplot(grid[0, 1])
    axial_raw_axis = figure.add_subplot(grid[1, 0])
    axial_filtered_axis = figure.add_subplot(
        grid[1, 1], sharey=axial_raw_axis
    )
    lateral_axis.plot(
        x,
        summary["lateral_median_resolution_nm"],
        marker="o",
        color="#157A6E",
        linewidth=1.3,
    )
    lateral_axis.set_title("Lateral resolution by sample")
    lateral_axis.set_ylabel("Resolution (nm)")
    lateral_axis.grid(axis="y", alpha=0.25)
    axial_raw_axis.plot(
        x,
        summary["xz_median_resolution_nm"],
        marker="o",
        label="XZ",
        color="#3564A7",
    )
    axial_raw_axis.plot(
        x,
        summary["yz_median_resolution_nm"],
        marker="s",
        label="YZ",
        color="#C65D35",
    )
    axial_filtered_axis.plot(
        x,
        summary["xz_6db_median_resolution_nm"],
        marker="o",
        markerfacecolor="none",
        linestyle="--",
        label="XZ after 6 dB",
        color="#3564A7",
    )
    axial_filtered_axis.plot(
        x,
        summary["yz_6db_median_resolution_nm"],
        marker="s",
        markerfacecolor="none",
        linestyle="--",
        label="YZ after 6 dB",
        color="#C65D35",
    )
    axial_raw_axis.set_title("Axial resolution before 6 dB filtering")
    axial_filtered_axis.set_title("Axial resolution after 6 dB filtering")
    axial_raw_axis.set_ylabel("Resolution (nm)")
    for axis in (axial_raw_axis, axial_filtered_axis):
        axis.set_xlabel("Sample")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    box_data = [
        summary["lateral_median_resolution_nm"].dropna(),
        summary["axial_6db_median_resolution_nm"].dropna(),
        summary["axial_median_resolution_nm"].dropna(),
    ]
    box_input = [
        values.to_numpy() if not values.empty else np.array([np.nan])
        for values in box_data
    ]
    box_artists = box_axis.boxplot(
        box_input,
        tick_labels=["Lateral", "Axial after 6 dB", "Axial raw"],
        patch_artist=True,
    )
    box_artists["boxes"][0].set(facecolor="#157A6E", alpha=0.45)
    box_artists["boxes"][1].set(
        facecolor="none", edgecolor="#3564A7", linestyle="--", linewidth=1.5
    )
    box_artists["boxes"][2].set(facecolor="#C65D35", alpha=0.45)
    for artist in (
        box_artists["whiskers"][2:4]
        + box_artists["caps"][2:4]
        + [box_artists["medians"][1]]
    ):
        artist.set(linestyle="--", color="#3564A7")
    for position, values in enumerate(box_data, start=1):
        if values.empty:
            continue
        box_axis.scatter(
            np.full(len(values), position),
            values,
            s=22,
            alpha=0.65,
            facecolors=(
                "none"
                if position == 2
                else "#157A6E" if position == 1 else "#C65D35"
            ),
            edgecolors=(
                "#3564A7"
                if position == 2
                else "#157A6E" if position == 1 else "#C65D35"
            ),
        )
        iqr = values.quantile(0.75) - values.quantile(0.25)
        box_axis.text(
            position,
            float(values.max()) * 1.04,
            f"n={len(values)}\nMedian={values.median():.1f}\nIQR={iqr:.1f}",
            ha="center",
            fontsize=9,
        )
    box_max = max(float(values.max()) for values in box_data if not values.empty)
    box_axis.set_ylim(0, box_max * 1.14)
    box_axis.set_ylabel("Resolution (nm)")
    box_axis.set_title("Sample-level distributions")
    for axis in (lateral_axis, axial_raw_axis, axial_filtered_axis):
        axis.set_xticks(x, plot_labels, rotation=75, ha="right", fontsize=7)
    figure.savefig(output_root / "batch_decorrelation_summary.png", dpi=180)
    plt.close(figure)
    render_contact_pages(
        lateral_figures,
        output_root,
        rows=5,
        columns=2,
        prefix="lateral_qc_page",
    )
    render_contact_pages(
        axial_figures,
        output_root,
        rows=3,
        columns=2,
        prefix="axial_qc_page",
    )
    write_index(output_root, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    summary = render_batch(args.output_root)
    print(f"rendered_samples={len(summary)}")


if __name__ == "__main__":
    main()
