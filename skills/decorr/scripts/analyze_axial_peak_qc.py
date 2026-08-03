"""Render exploratory axial peak-support QC without changing reported results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy.io import loadmat

from decorr_pipeline import (
    DecorrParameters,
    _local_max,
    apodize_rect,
    resolution_from_kc,
)


CURVE_SUPPORT_MIN = 6
SPECTRAL_SNR_MIN_DB = 6.0
PEAK_TOLERANCE_KC = 0.03
SPECTRAL_BAND_HALF_WIDTH = 0.02
SPECTRAL_NOISE_START = 0.9


def curve_peak_diagnostics(core: object, params: object) -> dict[str, object]:
    curves = np.asarray(core.curves, dtype=np.float64)
    sector = int(params.Na) // 2
    ng = int(params.Ng)
    if curves.ndim != 3 or curves.shape[2] < 2 * ng:
        raise ValueError(f"Unexpected decorrelation curve shape: {curves.shape}")

    radii = np.linspace(0.0, 1.0, curves.shape[0])
    first_curves = curves[:, sector, :ng]
    second_curves = curves[:, sector, ng : 2 * ng]
    first_kc = np.zeros(ng, dtype=np.float64)
    first_snr = np.zeros(ng, dtype=np.float64)
    for index in range(ng):
        peak, first_snr[index] = _local_max(first_curves[:, index], float(params.dt))
        first_kc[index] = radii[peak]
    metric = np.sqrt(np.maximum(first_kc, 0) * np.maximum(first_snr, 0))
    refinement_center = float(first_kc[int(np.argmax(metric))])
    refined_radii = np.linspace(
        max(0.0, refinement_center - float(params.rmin)),
        min(1.0, refinement_center + float(params.rmax)),
        curves.shape[0],
    )

    second_kc = np.zeros(ng, dtype=np.float64)
    for index in range(ng):
        peak, _ = _local_max(second_curves[:, index], float(params.dt))
        second_kc[index] = refined_radii[peak]
    selected_kc = float(np.asarray(core.kcGM).reshape(-1)[sector])
    tolerance = max(
        PEAK_TOLERANCE_KC,
        2.0 * float(np.median(np.diff(refined_radii))),
    )
    support_count = int(np.sum(np.abs(second_kc - selected_kc) <= tolerance))
    return {
        "selected_kc": selected_kc,
        "curve_support_count": support_count,
        "refined_peak_kc": second_kc,
    }


def axial_spectral_diagnostics(
    image: np.ndarray, selected_kc: float
) -> dict[str, object]:
    image = apodize_rect(np.asarray(image, dtype=np.float32), 20)
    spectrum = np.fft.fftshift(np.fft.fft(image, axis=0), axes=0)
    power = np.mean(np.abs(spectrum) ** 2, axis=1)
    center = image.shape[0] // 2
    positive = power[center:]
    negative = power[: center + 1][::-1]
    size = min(positive.size, negative.size)
    profile = 0.5 * (positive[:size] + negative[:size])
    frequencies = np.linspace(0.0, 1.0, size)

    noise_values = profile[frequencies >= SPECTRAL_NOISE_START]
    noise_floor = float(np.median(noise_values))
    selected_band = np.abs(frequencies - selected_kc) <= SPECTRAL_BAND_HALF_WIDTH
    if not np.any(selected_band):
        raise ValueError(f"Selected kc outside spectral profile: {selected_kc}")
    band_power = float(np.median(profile[selected_band]))
    spectral_snr_db = 10.0 * np.log10(
        max(band_power, np.finfo(float).eps)
        / max(noise_floor, np.finfo(float).eps)
    )
    profile_db = 10.0 * np.log10(
        np.maximum(profile, np.finfo(float).eps)
        / max(noise_floor, np.finfo(float).eps)
    )
    return {
        "spectral_snr_db": spectral_snr_db,
        "spectral_frequencies": frequencies,
        "spectral_profile_db": profile_db,
    }


def qc_status(curve_support_count: int, spectral_snr_db: float) -> str:
    if (
        curve_support_count >= CURVE_SUPPORT_MIN
        and spectral_snr_db >= SPECTRAL_SNR_MIN_DB
    ):
        return "PASS"
    return "REVIEW"


def _heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    value_format: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = axis.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                format(values[row, column], value_format),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    axis.set_title(title, fontsize=10)
    axis.set_xlabel("Montage group")
    axis.set_xticks(np.arange(values.shape[1]), np.arange(1, values.shape[1] + 1))
    axis.set_yticks([0, 1], ["XZ", "YZ"])
    axis.figure.colorbar(image, ax=axis, shrink=0.72)


def render_sample_qc(
    sample_name: str,
    rows: list[dict[str, object]],
    output: Path,
    *,
    resolution_limits: tuple[float, float],
) -> None:
    table = pd.DataFrame(rows)
    table["direction_order"] = table["direction"].map({"XZ": 0, "YZ": 1})
    table = table.sort_values(["direction_order", "group"])
    shape = (2, int(table["group"].max()))

    def matrix(column: str) -> np.ndarray:
        result = np.full(shape, np.nan)
        for row in table.itertuples(index=False):
            result[int(row.direction_order), int(row.group) - 1] = getattr(row, column)
        return result

    display_name = sample_name.replace("、", "-")
    figure, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)
    figure.suptitle(f"{display_name} | exploratory axial peak QC", fontsize=15)
    _heatmap(
        axes[0, 0],
        matrix("baseline_resolution_nm"),
        title=(
            "Baseline axial resolution "
            f"(nm; batch scale {resolution_limits[0]:.0f}-{resolution_limits[1]:.0f})"
        ),
        cmap="viridis_r",
        value_format=".0f",
        vmin=resolution_limits[0],
        vmax=resolution_limits[1],
    )
    _heatmap(
        axes[0, 1],
        matrix("spectral_snr_db"),
        title=f"Spectral margin at selected kc (dB; review < {SPECTRAL_SNR_MIN_DB:g})",
        cmap="RdYlGn",
        value_format=".1f",
        vmin=0,
        vmax=15,
    )
    _heatmap(
        axes[0, 2],
        matrix("curve_support_count"),
        title=f"Refined curves supporting selected peak (of 10; review < {CURVE_SUPPORT_MIN})",
        cmap="YlGn",
        value_format=".0f",
        vmin=0,
        vmax=10,
    )

    suspect = table.sort_values(
        ["spectral_snr_db", "curve_support_count"]
    ).iloc[0]
    reference = table.sort_values(
        ["spectral_snr_db", "curve_support_count"], ascending=False
    ).iloc[0]
    for row, color, label in (
        (suspect, "#C43C39", "lowest margin"),
        (reference, "#157A6E", "highest margin"),
    ):
        axes[1, 0].plot(
            row["spectral_frequencies"],
            row["spectral_profile_db"],
            color=color,
            linewidth=1.2,
            label=(
                f"{label}: {row['direction']} G{int(row['group'])}, "
                f"{row['baseline_resolution_nm']:.0f} nm"
            ),
        )
        axes[1, 0].axvline(
            row["baseline_kc"], color=color, linewidth=1, linestyle="--"
        )
    axes[1, 0].axhline(SPECTRAL_SNR_MIN_DB, color="black", linestyle=":", linewidth=1)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(bottom=-5)
    axes[1, 0].set_title("Axial power spectrum relative to outer-frequency noise")
    axes[1, 0].set_xlabel("Normalized axial frequency")
    axes[1, 0].set_ylabel("Power / noise floor (dB)")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend(fontsize=8)

    colors = table["direction"].map({"XZ": "#3564A7", "YZ": "#C65D35"})
    axes[1, 1].scatter(
        table["baseline_resolution_nm"],
        table["spectral_snr_db"],
        c=colors,
        s=38,
        alpha=0.85,
    )
    axes[1, 1].axhline(
        SPECTRAL_SNR_MIN_DB, color="black", linestyle="--", linewidth=1
    )
    axes[1, 1].set_title("Resolution vs spectral margin")
    axes[1, 1].set_xlabel("Baseline resolution (nm)")
    axes[1, 1].set_ylabel("Spectral margin (dB)")
    axes[1, 1].grid(alpha=0.2)

    axes[1, 2].scatter(
        table["baseline_resolution_nm"],
        table["curve_support_count"],
        c=colors,
        s=38,
        alpha=0.85,
    )
    axes[1, 2].axhline(
        CURVE_SUPPORT_MIN, color="black", linestyle="--", linewidth=1
    )
    review_count = int((table["qc_status"] == "REVIEW").sum())
    axes[1, 2].set_title(
        f"Resolution vs curve support | REVIEW {review_count}/{len(table)}"
    )
    axes[1, 2].set_xlabel("Baseline resolution (nm)")
    axes[1, 2].set_ylabel("Supporting refined curves (of 10)")
    axes[1, 2].set_ylim(-0.5, 10.5)
    axes[1, 2].grid(alpha=0.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def render_batch_summary(table: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    figure.suptitle(
        "6 dB spectral support: "
        "10 log10[P(kc)/Pnoise] >= 6 (power ratio >= 3.98). "
        "Workflow QC gate; raw results are retained.",
        fontsize=11,
    )
    for status, marker, color in (
        ("PASS", "o", "#157A6E"),
        ("REVIEW", "x", "#C43C39"),
    ):
        subset = table[table["qc_status"] == status]
        axes[0, 0].scatter(
            subset["baseline_resolution_nm"],
            subset["spectral_snr_db"],
            label=status,
            marker=marker,
            color=color,
            s=38,
            alpha=0.75,
        )
        axes[0, 1].scatter(
            subset["baseline_resolution_nm"],
            subset["curve_support_count"],
            label=status,
            marker=marker,
            color=color,
            s=38,
            alpha=0.75,
        )
    axes[0, 0].axhline(
        SPECTRAL_SNR_MIN_DB, color="black", linestyle="--", linewidth=1
    )
    axes[0, 0].set_title("All axial groups: resolution vs spectral margin")
    axes[0, 0].set_xlabel("Baseline resolution (nm, unchanged)")
    axes[0, 0].set_ylabel("Spectral margin at selected kc (dB)")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend()

    axes[0, 1].axhline(
        CURVE_SUPPORT_MIN, color="black", linestyle="--", linewidth=1
    )
    axes[0, 1].set_title("All axial groups: resolution vs curve support")
    axes[0, 1].set_xlabel("Baseline resolution (nm, unchanged)")
    axes[0, 1].set_ylabel("Supporting refined curves (of 10)")
    axes[0, 1].set_ylim(-0.5, 10.5)
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend()

    sample_review = (
        table.assign(is_review=table["qc_status"].eq("REVIEW"))
        .groupby("sample", sort=True)["is_review"]
        .sum()
    )
    x = np.arange(len(sample_review))
    axes[1, 0].bar(x, sample_review, color="#C65D35")
    axes[1, 0].set_title("REVIEW groups by sample (24 XZ/YZ groups each)")
    axes[1, 0].set_ylabel("REVIEW group count")
    axes[1, 0].set_xticks(
        x,
        sample_review.index.str.replace("、", "-", regex=False),
        rotation=75,
        ha="right",
        fontsize=7,
    )
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].hist(
        table["spectral_snr_db"],
        bins=30,
        color="#3564A7",
        alpha=0.8,
    )
    axes[1, 1].axvline(
        SPECTRAL_SNR_MIN_DB, color="black", linestyle="--", linewidth=1.2
    )
    axes[1, 1].set_title(
        "Spectral-margin distribution\n"
        f"PASS={int((table['qc_status'] == 'PASS').sum())}, "
        f"REVIEW={int((table['qc_status'] == 'REVIEW').sum())}"
    )
    axes[1, 1].set_xlabel("Spectral margin at selected kc (dB)")
    axes[1, 1].set_ylabel("Group count")
    axes[1, 1].grid(axis="y", alpha=0.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def render_contact_pages(sample_figures: list[tuple[str, Path]], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_count = 0
    for first in range(0, len(sample_figures), 6):
        page_count += 1
        chunk = sample_figures[first : first + 6]
        figure, axes = plt.subplots(3, 2, figsize=(22, 18), constrained_layout=True)
        for axis in axes.flat:
            axis.set_axis_off()
        for axis, (sample, path) in zip(axes.flat, chunk):
            axis.imshow(plt.imread(path))
            axis.set_title(sample.replace("、", "-"), fontsize=11)
            axis.set_axis_off()
        figure.savefig(
            output_dir / f"axial_peak_qc_page-{page_count:02d}.png",
            dpi=120,
            facecolor="white",
        )
        plt.close(figure)
    return page_count


def analyze_batch(batch_root: Path, output_dir: Path) -> dict[str, object]:
    samples_root = batch_root / "samples"
    if not samples_root.is_dir():
        samples_root = batch_root
    diagnostic_paths = sorted(samples_root.glob("*/axial/axial_diagnostics.npz"))
    if not diagnostic_paths:
        diagnostic_paths = sorted(samples_root.glob("*/axial/*/axial_result.mat"))
    if not diagnostic_paths:
        raise RuntimeError(f"No Python or MATLAB axial diagnostics below {samples_root}")
    if output_dir.resolve() == batch_root.resolve():
        raise ValueError("QC output must not overwrite the baseline batch root")

    for path in (
        output_dir / "axial_peak_qc.csv",
        output_dir / "axial_peak_qc_summary.png",
        output_dir / "qc_run_info.json",
    ):
        path.unlink(missing_ok=True)
    for pattern in ("pages/*.png", "samples/*.png"):
        for path in output_dir.glob(pattern):
            path.unlink()
    for path in output_dir.glob("samples/*/axial_peak_qc.png"):
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            pass

    all_rows: list[dict[str, object]] = []
    sample_rows_by_name: list[tuple[str, list[dict[str, object]]]] = []
    for index, diagnostic_path in enumerate(diagnostic_paths, start=1):
        sample_name = diagnostic_path.relative_to(samples_root).parts[0]
        print(f"[{index}/{len(diagnostic_paths)}] {sample_name}", flush=True)
        csv = pd.read_csv(diagnostic_path.parent / "axial_result.csv")
        if diagnostic_path.suffix == ".npz":
            data = np.load(diagnostic_path)
            defaults = DecorrParameters()
            params = SimpleNamespace(
                Na=defaults.na,
                Ng=defaults.ng,
                dt=defaults.dt,
                rmin=defaults.rmin,
                rmax=defaults.rmax,
            )
        else:
            data = loadmat(
                diagnostic_path, squeeze_me=True, struct_as_record=False
            )
            params = data["params"]
        sample_rows: list[dict[str, object]] = []
        for direction in ("xz", "yz"):
            montages = np.asarray(
                tifffile.imread(diagnostic_path.parent / f"{direction}_montage.tif"),
                dtype=np.float32,
            )
            if diagnostic_path.suffix == ".npz":
                cores = []
                for group, curves in enumerate(data[f"{direction}_curves"]):
                    kc_values = np.full(defaults.na, np.nan)
                    kc_values[defaults.na // 2] = float(
                        csv.loc[group, f"{direction}_kc"]
                    )
                    cores.append(SimpleNamespace(curves=curves, kcGM=kc_values))
            else:
                cores = np.atleast_1d(data[f"{direction}Core"])
            if len(cores) != len(csv) or len(montages) != len(csv):
                raise RuntimeError(
                    f"Group-count mismatch for {sample_name} {direction.upper()}"
                )
            for group, (core, montage) in enumerate(zip(cores, montages), start=1):
                curve = curve_peak_diagnostics(core, params)
                baseline_kc = float(csv.loc[group - 1, f"{direction}_kc"])
                if not np.isclose(
                    curve["selected_kc"], baseline_kc, atol=1e-10, rtol=0
                ):
                    raise RuntimeError(
                        f"MAT/CSV kc mismatch for {sample_name} "
                        f"{direction.upper()} group {group}"
                    )
                spectral = axial_spectral_diagnostics(montage, baseline_kc)
                pixel_size_nm = float(csv.loc[group - 1, "axial_pixel_size_nm"])
                support_count = int(curve["curve_support_count"])
                spectral_snr_db = float(spectral["spectral_snr_db"])
                sample_rows.append(
                    {
                        "sample": sample_name,
                        "direction": direction.upper(),
                        "group": group,
                        "baseline_kc": baseline_kc,
                        "baseline_resolution_nm": resolution_from_kc(
                            pixel_size_nm, baseline_kc
                        ),
                        "curve_support_count": support_count,
                        "spectral_snr_db": spectral_snr_db,
                        "spectral_support_6db": (
                            spectral_snr_db >= SPECTRAL_SNR_MIN_DB
                        ),
                        "qc_status": qc_status(support_count, spectral_snr_db),
                        **spectral,
                    }
                )
        sample_rows_by_name.append((sample_name, sample_rows))
        all_rows.extend(sample_rows)

    full_table = pd.DataFrame(all_rows)
    resolution_min = float(full_table["baseline_resolution_nm"].min())
    resolution_max = float(full_table["baseline_resolution_nm"].max())
    if resolution_max <= resolution_min:
        resolution_max = resolution_min + 1.0
    resolution_limits = (resolution_min, resolution_max)
    sample_figures: list[tuple[str, Path]] = []
    for sample_name, sample_rows in sample_rows_by_name:
        sample_output = output_dir / "samples" / f"{sample_name}.png"
        render_sample_qc(
            sample_name,
            sample_rows,
            sample_output,
            resolution_limits=resolution_limits,
        )
        sample_figures.append((sample_name, sample_output))
    export_columns = [
        "sample",
        "direction",
        "group",
        "baseline_resolution_nm",
        "curve_support_count",
        "spectral_snr_db",
        "spectral_support_6db",
        "qc_status",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    full_table[export_columns].to_csv(
        output_dir / "axial_peak_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    render_batch_summary(full_table, output_dir / "axial_peak_qc_summary.png")
    page_count = render_contact_pages(sample_figures, output_dir / "pages")
    summary = {
        "source_batch": str(batch_root),
        "sample_count": len(sample_figures),
        "group_direction_count": len(full_table),
        "pass_count": int((full_table["qc_status"] == "PASS").sum()),
        "review_count": int((full_table["qc_status"] == "REVIEW").sum()),
        "curve_support_min": CURVE_SUPPORT_MIN,
        "spectral_snr_min_db": SPECTRAL_SNR_MIN_DB,
        "spectral_power_ratio_min": 10 ** (SPECTRAL_SNR_MIN_DB / 10),
        "resolution_color_min_nm": resolution_min,
        "resolution_color_max_nm": resolution_max,
        "spectral_color_min_db": 0.0,
        "spectral_color_max_db": 15.0,
        "curve_support_color_min": 0,
        "curve_support_color_max": 10,
        "baseline_results_changed": False,
        "contact_page_count": page_count,
    }
    (output_dir / "qc_run_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze_batch(args.batch_root, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
