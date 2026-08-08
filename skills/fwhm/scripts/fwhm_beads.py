from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import curve_fit

from fwhm_common import (
    apply_tukey_outer_fence,
    find_tiff_files as common_find_tiff_files,
    plot_resolution_boxplot,
    read_tiff,
    sample_status,
    write_summary_markdown,
)


FWHM_FACTOR = 2.0 * math.sqrt(2.0 * math.log(2.0))


def default_run_parameter() -> dict[str, Any]:
    return {
        "input_path": None,
        "output_path": Path.cwd() / "fwhm_results",
        "axes": "auto",
        "channel": None,
        "timepoint": None,
        "recursive": False,
        "detection_snr": 100.0,
        "min_fit_snr": 5.0,
        "min_fit_r2": 0.90,
        "max_center_offset_px": 1.0,
        "min_z_fit_half_range_sigma": 2.0,
        "detection_sigma_xy_px": 1.0,
        "detection_sigma_z_px": 0.7,
        "min_distance_xy_px": 6,
        "min_distance_z_px": 2,
        "roi_radius_xy_px": 6,
        "roi_radius_z_px": 0,
        "profile_half_width_px": 1,
        "edge_margin_fraction": 0.10,
        "shape_radius_xy_px": 8,
        "max_lateral_moment_ratio": 1.25,
        "max_lateral_centroid_offset_px": 0.75,
        "min_isolation_xy_px": 13.0,
        "max_isolation_z_px": 2,
        "exclude_z_start": 1,
        "exclude_z_end": 1,
        "min_fwhm_xy_nm": None,
        "max_fwhm_xy_nm": None,
        "min_fwhm_z_nm": None,
        "max_fwhm_z_nm": None,
        "max_candidates": 500,
        "diagnostic_count": 8,
        "tukey_minimum_count": 10,
        "median_nearby_count": 0,
    }


@dataclass(frozen=True)
class FitResult:
    amplitude: float
    center_nm: float
    sigma_nm: float
    offset: float
    fwhm_nm: float
    r2: float
    center_offset_px: float


def gaussian_with_offset(
    coordinate_nm: np.ndarray,
    amplitude: float,
    center_nm: float,
    sigma_nm: float,
    offset: float,
) -> np.ndarray:
    return offset + amplitude * np.exp(
        -0.5 * ((coordinate_nm - center_nm) / sigma_nm) ** 2
    )


def robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    median = float(np.median(finite))
    sigma = 1.4826 * float(np.median(np.abs(finite - median)))
    if sigma > 0:
        return sigma
    q16, q84 = np.percentile(finite, [16.0, 84.0])
    return max(float(q84 - q16) / 2.0, 0.0)


def fit_gaussian_profile(
    profile: np.ndarray,
    spacing_nm: float,
    expected_center_px: float,
) -> FitResult:
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or values.size < 5 or not np.all(np.isfinite(values)):
        raise ValueError("profile must contain at least five finite samples")

    coordinate_nm = np.arange(values.size, dtype=np.float64) * spacing_nm
    low = float(np.percentile(values, 10.0))
    high = float(np.max(values))
    amplitude = high - low
    if amplitude <= 0:
        raise ValueError("profile has no positive peak")

    peak_index = int(np.argmax(values))
    half_level = low + amplitude / 2.0
    above_half = np.flatnonzero(values >= half_level)
    if above_half.size >= 2:
        fwhm_guess_nm = max(
            float(above_half[-1] - above_half[0]) * spacing_nm,
            spacing_nm,
        )
    else:
        fwhm_guess_nm = 2.0 * spacing_nm

    sigma_guess_nm = fwhm_guess_nm / FWHM_FACTOR
    span_nm = float(coordinate_nm[-1] - coordinate_nm[0])
    value_range = max(float(np.ptp(values)), np.finfo(np.float64).eps)
    lower = [0.0, coordinate_nm[0], spacing_nm / 4.0, low - value_range]
    upper = [
        5.0 * value_range,
        coordinate_nm[-1],
        max(span_nm, spacing_nm),
        high + value_range,
    ]
    initial = [
        amplitude,
        coordinate_nm[peak_index],
        max(sigma_guess_nm, spacing_nm / 2.0),
        low,
    ]

    parameters, _ = curve_fit(
        gaussian_with_offset,
        coordinate_nm,
        values,
        p0=initial,
        bounds=(lower, upper),
        maxfev=20000,
    )
    prediction = gaussian_with_offset(coordinate_nm, *parameters)
    residual_sum = float(np.sum((values - prediction) ** 2))
    total_sum = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else 0.0
    center_offset_px = (
        float(parameters[1]) / spacing_nm - float(expected_center_px)
    )
    return FitResult(
        amplitude=float(parameters[0]),
        center_nm=float(parameters[1]),
        sigma_nm=abs(float(parameters[2])),
        offset=float(parameters[3]),
        fwhm_nm=FWHM_FACTOR * abs(float(parameters[2])),
        r2=r2,
        center_offset_px=center_offset_px,
    )


def _normalise_image_shape(image: np.ndarray) -> np.ndarray:
    volume = np.squeeze(np.asarray(image))
    if volume.ndim not in (2, 3):
        raise ValueError(
            f"expected a 2D image or ZYX stack after squeeze, got {volume.shape}"
        )
    if not np.issubdtype(volume.dtype, np.number):
        raise ValueError(f"unsupported TIFF dtype: {volume.dtype}")
    return volume


def _detection_image(
    image: np.ndarray,
    run_parameter: dict[str, Any],
) -> np.ndarray:
    if image.ndim == 3:
        small_sigma = (
            run_parameter["detection_sigma_z_px"],
            run_parameter["detection_sigma_xy_px"],
            run_parameter["detection_sigma_xy_px"],
        )
        large_sigma = (
            max(2.0 * run_parameter["detection_sigma_z_px"], 1.2),
            max(3.0 * run_parameter["detection_sigma_xy_px"], 2.0),
            max(3.0 * run_parameter["detection_sigma_xy_px"], 2.0),
        )
    else:
        small_sigma = (
            run_parameter["detection_sigma_xy_px"],
            run_parameter["detection_sigma_xy_px"],
        )
        large_sigma = (
            max(3.0 * run_parameter["detection_sigma_xy_px"], 2.0),
            max(3.0 * run_parameter["detection_sigma_xy_px"], 2.0),
        )
    finite = np.nan_to_num(
        image.astype(np.float64, copy=False),
        nan=float(np.nanmedian(image)),
        posinf=0.0,
        neginf=0.0,
    )
    return gaussian_filter(finite, small_sigma) - gaussian_filter(
        finite, large_sigma
    )


def detect_candidates(
    image: np.ndarray,
    run_parameter: dict[str, Any],
) -> tuple[list[tuple[int, ...]], np.ndarray, float | None, int]:
    detection = _detection_image(image, run_parameter)
    noise = robust_sigma(detection)
    center = float(np.median(detection))
    dynamic_range = float(np.ptp(detection))
    scale = max(float(np.max(np.abs(detection))), 1.0)
    if dynamic_range <= np.finfo(np.float64).eps * scale:
        return [], detection, None, 0
    if noise > 0:
        threshold = center + run_parameter["detection_snr"] * noise
    else:
        threshold = float(np.percentile(detection, 99.5))

    if image.ndim == 3:
        size = (
            2 * run_parameter["min_distance_z_px"] + 1,
            2 * run_parameter["min_distance_xy_px"] + 1,
            2 * run_parameter["min_distance_xy_px"] + 1,
        )
    else:
        size = (
            2 * run_parameter["min_distance_xy_px"] + 1,
            2 * run_parameter["min_distance_xy_px"] + 1,
        )
    local_maximum = detection == maximum_filter(detection, size=size, mode="nearest")
    coordinates = np.argwhere(local_maximum & (detection >= threshold))
    if coordinates.size == 0:
        return [], detection, threshold, 0

    scores = detection[tuple(coordinates.T)]
    candidate_count_before_limit = int(coordinates.shape[0])
    order = np.argsort(scores)[::-1][: run_parameter["max_candidates"]]
    return (
        [tuple(int(v) for v in coordinates[i]) for i in order],
        detection,
        threshold,
        candidate_count_before_limit,
    )


def _axis_profiles(
    image: np.ndarray,
    coordinate: tuple[int, ...],
    run_parameter: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    half_width = run_parameter["profile_half_width_px"]
    radius_xy = run_parameter["roi_radius_xy_px"]

    if image.ndim == 2:
        y, x = coordinate
        profiles = {
            "x": np.mean(
                image[
                    y - half_width : y + half_width + 1,
                    x - radius_xy : x + radius_xy + 1,
                ],
                axis=0,
            ),
            "y": np.mean(
                image[
                    y - radius_xy : y + radius_xy + 1,
                    x - half_width : x + half_width + 1,
                ],
                axis=1,
            ),
        }
        centers = {"x": float(radius_xy), "y": float(radius_xy)}
        return profiles, centers

    z, y, x = coordinate
    z_start = run_parameter["exclude_z_start"]
    z_stop = image.shape[0] - run_parameter["exclude_z_end"]
    if run_parameter["roi_radius_z_px"] > 0:
        z_start = z - run_parameter["roi_radius_z_px"]
        z_stop = z + run_parameter["roi_radius_z_px"] + 1
    profiles = {
        "x": np.mean(
            image[
                z,
                y - half_width : y + half_width + 1,
                x - radius_xy : x + radius_xy + 1,
            ],
            axis=0,
        ),
        "y": np.mean(
            image[
                z,
                y - radius_xy : y + radius_xy + 1,
                x - half_width : x + half_width + 1,
            ],
            axis=1,
        ),
        "z": np.mean(
            image[
                z_start:z_stop,
                y - half_width : y + half_width + 1,
                x - half_width : x + half_width + 1,
            ],
            axis=(1, 2),
        ),
    }
    centers = {
        "x": float(radius_xy),
        "y": float(radius_xy),
        "z": float(z - z_start),
    }
    return profiles, centers


def _local_signal_metrics(
    image: np.ndarray,
    coordinate: tuple[int, ...],
    run_parameter: dict[str, Any],
    global_noise: float,
) -> tuple[float, float, float | None]:
    radius_xy = run_parameter["roi_radius_xy_px"]
    if image.ndim == 2:
        y, x = coordinate
        region = image[
            y - radius_xy : y + radius_xy + 1,
            x - radius_xy : x + radius_xy + 1,
        ]
        shell_mask = np.zeros(region.shape, dtype=bool)
        shell_mask[[0, -1], :] = True
        shell_mask[:, [0, -1]] = True
    else:
        z, y, x = coordinate
        z_radius = run_parameter["roi_radius_z_px"]
        z_start = run_parameter["exclude_z_start"] if z_radius == 0 else z - z_radius
        z_stop = (
            image.shape[0] - run_parameter["exclude_z_end"]
            if z_radius == 0
            else z + z_radius + 1
        )
        region = image[
            z_start:z_stop,
            y - radius_xy : y + radius_xy + 1,
            x - radius_xy : x + radius_xy + 1,
        ]
        shell_mask = np.zeros(region.shape, dtype=bool)
        shell_mask[:, [0, -1], :] = True
        shell_mask[:, :, [0, -1]] = True
        shell_mask[[0, -1], :, :] = True

    shell = np.asarray(region[shell_mask], dtype=np.float64)
    background = float(np.median(shell))
    local_noise = robust_sigma(shell)
    noise = local_noise if local_noise > 0 else global_noise
    peak = float(np.max(region))
    snr = float((peak - background) / noise) if noise > 0 else None
    return peak, background, snr


def _lateral_shape_metrics(
    image: np.ndarray,
    coordinate: tuple[int, ...],
    run_parameter: dict[str, Any],
) -> tuple[float | None, float | None, float | None]:
    radius = run_parameter["shape_radius_xy_px"]
    if image.ndim == 3:
        z, y, x = coordinate
        plane = image[z]
    else:
        y, x = coordinate
        plane = image
    patch = np.asarray(
        plane[y - radius : y + radius + 1, x - radius : x + radius + 1],
        dtype=np.float64,
    )
    shell = np.concatenate(
        (patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1])
    )
    signal = np.clip(patch - np.median(shell), 0.0, None)
    total = float(np.sum(signal))
    if total <= 0:
        return None, None, None

    yy, xx = np.indices(signal.shape, dtype=np.float64)
    centroid_x = float(np.sum(signal * xx) / total)
    centroid_y = float(np.sum(signal * yy) / total)
    dx = xx - centroid_x
    dy = yy - centroid_y
    covariance = np.array(
        [
            [np.sum(signal * dx * dx), np.sum(signal * dx * dy)],
            [np.sum(signal * dx * dy), np.sum(signal * dy * dy)],
        ],
        dtype=np.float64,
    )
    covariance /= total
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0:
        moment_ratio = None
    else:
        moment_ratio = float(math.sqrt(eigenvalues[-1] / eigenvalues[0]))
    centroid_offset = float(
        math.hypot(centroid_x - radius, centroid_y - radius)
    )
    symmetry = float(
        1.0 - np.sum(np.abs(signal - np.rot90(signal, 2))) / (2.0 * total)
    )
    return moment_ratio, centroid_offset, symmetry


def _nearest_candidate_distance(
    candidates: Sequence[tuple[int, ...]],
    candidate_index: int,
    max_z_distance: int,
) -> tuple[float | None, int | None]:
    coordinate = candidates[candidate_index]
    nearest_xy: float | None = None
    nearest_id: int | None = None
    for other_index, other in enumerate(candidates):
        if other_index == candidate_index:
            continue
        if len(coordinate) == 3:
            if abs(coordinate[0] - other[0]) > max_z_distance:
                continue
            y, x = coordinate[1:]
            other_y, other_x = other[1:]
        else:
            y, x = coordinate
            other_y, other_x = other
        distance = float(math.hypot(y - other_y, x - other_x))
        if nearest_xy is None or distance < nearest_xy:
            nearest_xy = distance
            nearest_id = other_index + 1
    return nearest_xy, nearest_id


def _border_reasons(
    image: np.ndarray,
    coordinate: tuple[int, ...],
    run_parameter: dict[str, Any],
) -> list[str]:
    radius_xy = run_parameter["roi_radius_xy_px"]
    half_width = run_parameter["profile_half_width_px"]
    required_xy = max(
        radius_xy,
        half_width,
        run_parameter["shape_radius_xy_px"],
    )
    field_margin_y = max(
        required_xy,
        math.ceil(image.shape[-2] * run_parameter["edge_margin_fraction"]),
    )
    field_margin_x = max(
        required_xy,
        math.ceil(image.shape[-1] * run_parameter["edge_margin_fraction"]),
    )
    if image.ndim == 2:
        y, x = coordinate
        reasons = []
        if (
            y - required_xy < 0
            or y + required_xy >= image.shape[0]
            or x - required_xy < 0
            or x + required_xy >= image.shape[1]
        ):
            reasons.append("xy_border")
        if (
            y < field_margin_y
            or y >= image.shape[0] - field_margin_y
            or x < field_margin_x
            or x >= image.shape[1] - field_margin_x
        ):
            reasons.append("field_edge")
        return reasons

    z, y, x = coordinate
    reasons = []
    if (
        y - required_xy < 0
        or y + required_xy >= image.shape[1]
        or x - required_xy < 0
        or x + required_xy >= image.shape[2]
    ):
        reasons.append("xy_border")
    if (
        y < field_margin_y
        or y >= image.shape[1] - field_margin_y
        or x < field_margin_x
        or x >= image.shape[2] - field_margin_x
    ):
        reasons.append("field_edge")
    valid_start = run_parameter["exclude_z_start"]
    valid_stop = image.shape[0] - run_parameter["exclude_z_end"]
    if not (valid_start <= z < valid_stop):
        reasons.append("excluded_z")
    elif run_parameter["roi_radius_z_px"] > 0 and (
        z - run_parameter["roi_radius_z_px"] < valid_start
        or z + run_parameter["roi_radius_z_px"] >= valid_stop
    ):
        reasons.append("z_border")
    elif valid_stop - valid_start < 5:
        reasons.append("too_few_z_samples")
    return reasons


def _empty_fit_fields(axis: str) -> dict[str, Any]:
    return {
        f"fwhm_{axis}_nm": None,
        f"r2_{axis}": None,
        f"center_offset_{axis}_px": None,
        f"amplitude_{axis}": None,
        f"offset_{axis}": None,
    }


def analyze_image(
    image: np.ndarray,
    source_name: str,
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    volume = _normalise_image_shape(image)
    candidates, detection, threshold, candidate_count_before_limit = detect_candidates(
        volume, run_parameter
    )
    global_noise = robust_sigma(
        volume.astype(np.float64) - gaussian_filter(volume.astype(np.float64), 2.0)
    )
    diagnostics: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for candidate_index, coordinate in enumerate(candidates):
        candidate_id = candidate_index + 1
        if volume.ndim == 3:
            z, y, x = coordinate
        else:
            y, x = coordinate
            z = None
        row: dict[str, Any] = {
            "source": source_name,
            "candidate_id": candidate_id,
            "z_px": z,
            "y_px": y,
            "x_px": x,
            "accepted": False,
            "rejection_reasons": "",
            "detection_score": float(detection[coordinate]),
            "detection_threshold": threshold,
            "peak": None,
            "background": None,
            "snr": None,
            "fwhm_xy_nm": None,
            "lateral_moment_ratio": None,
            "lateral_centroid_offset_px": None,
            "lateral_symmetry": None,
            "nearest_candidate_xy_px": None,
            "nearest_candidate_id": None,
        }
        for axis in ("x", "y", "z"):
            row.update(_empty_fit_fields(axis))

        reasons = _border_reasons(volume, coordinate, run_parameter)
        nearest_distance, nearest_id = _nearest_candidate_distance(
            candidates,
            candidate_index,
            run_parameter["max_isolation_z_px"],
        )
        row["nearest_candidate_xy_px"] = nearest_distance
        row["nearest_candidate_id"] = nearest_id
        if (
            nearest_distance is not None
            and nearest_distance < run_parameter["min_isolation_xy_px"]
        ):
            reasons.append("nearby_candidate")
        if reasons:
            row["rejection_reasons"] = ";".join(reasons)
            rows.append(row)
            continue

        peak, background, snr = _local_signal_metrics(
            volume, coordinate, run_parameter, global_noise
        )
        row.update({"peak": peak, "background": background, "snr": snr})
        if snr is not None and snr < run_parameter["min_fit_snr"]:
            reasons.append("low_snr")
        if np.issubdtype(volume.dtype, np.integer) and peak >= np.iinfo(volume.dtype).max:
            reasons.append("saturated")

        moment_ratio, centroid_offset, symmetry = _lateral_shape_metrics(
            volume,
            coordinate,
            run_parameter,
        )
        row.update(
            {
                "lateral_moment_ratio": moment_ratio,
                "lateral_centroid_offset_px": centroid_offset,
                "lateral_symmetry": symmetry,
            }
        )
        if (
            moment_ratio is None
            or centroid_offset is None
            or moment_ratio > run_parameter["max_lateral_moment_ratio"]
            or centroid_offset
            > run_parameter["max_lateral_centroid_offset_px"]
        ):
            reasons.append("non_single_bead_shape")

        profiles, expected_centers = _axis_profiles(
            volume, coordinate, run_parameter
        )
        fits: dict[str, FitResult] = {}
        for axis, profile in profiles.items():
            spacing = (
                opt.pixel_size_z_nm if axis == "z" else opt.pixel_size_xy_nm
            )
            try:
                fit = fit_gaussian_profile(
                    profile,
                    spacing_nm=spacing,
                    expected_center_px=expected_centers[axis],
                )
                fits[axis] = fit
                row.update(
                    {
                        f"fwhm_{axis}_nm": fit.fwhm_nm,
                        f"r2_{axis}": fit.r2,
                        f"center_offset_{axis}_px": fit.center_offset_px,
                        f"amplitude_{axis}": fit.amplitude,
                        f"offset_{axis}": fit.offset,
                    }
                )
                if fit.r2 < run_parameter["min_fit_r2"]:
                    reasons.append(f"low_r2_{axis}")
                if (
                    abs(fit.center_offset_px)
                    > run_parameter["max_center_offset_px"]
                ):
                    reasons.append(f"off_center_{axis}")
                if axis == "z":
                    profile_span_nm = (profile.size - 1) * spacing
                    required_half_range_nm = (
                        run_parameter["min_z_fit_half_range_sigma"]
                        * fit.sigma_nm
                    )
                    if (
                        fit.center_nm < required_half_range_nm
                        or profile_span_nm - fit.center_nm
                        < required_half_range_nm
                    ):
                        reasons.append("truncated_z_profile")
            except (RuntimeError, ValueError, FloatingPointError):
                reasons.append(f"fit_failed_{axis}")

        if "x" in fits and "y" in fits:
            fwhm_xy = (fits["x"].fwhm_nm + fits["y"].fwhm_nm) / 2.0
            row["fwhm_xy_nm"] = fwhm_xy
            minimum_xy = run_parameter["min_fwhm_xy_nm"]
            maximum_xy = run_parameter["max_fwhm_xy_nm"]
            if (
                (minimum_xy is not None and fwhm_xy < minimum_xy)
                or (maximum_xy is not None and fwhm_xy > maximum_xy)
            ):
                reasons.append("xy_out_of_range")
        if "z" in fits:
            minimum_z = run_parameter["min_fwhm_z_nm"]
            maximum_z = run_parameter["max_fwhm_z_nm"]
            if (
                (minimum_z is not None and fits["z"].fwhm_nm < minimum_z)
                or (maximum_z is not None and fits["z"].fwhm_nm > maximum_z)
            ):
                reasons.append("z_out_of_range")

        row["accepted"] = not reasons
        row["rejection_reasons"] = ";".join(dict.fromkeys(reasons))
        rows.append(row)
        if len(diagnostics) < run_parameter["diagnostic_count"]:
            diagnostics.append(
                {
                    "candidate_id": candidate_id,
                    "accepted": row["accepted"],
                    "profiles": profiles,
                    "fits": fits,
                    "expected_centers": expected_centers,
                }
            )

    detection_metadata = {
        "threshold": threshold,
        "candidate_count_before_limit": candidate_count_before_limit,
        "candidate_count_analyzed": len(candidates),
        "candidate_limit": run_parameter["max_candidates"],
        "candidate_limit_reached": candidate_count_before_limit > len(candidates),
    }
    return rows, diagnostics, detection, detection_metadata


def _clean_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in path.stem
    )
    return f"{safe[:80]}-{digest}"


def _summary_statistics(values: Sequence[float], seed: int = 0) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    rng = np.random.default_rng(seed)
    medians = np.empty(2000, dtype=np.float64)
    for start in range(0, medians.size, 100):
        stop = min(start + 100, medians.size)
        bootstrap = rng.choice(
            array,
            size=(stop - start, array.size),
            replace=True,
        )
        medians[start:stop] = np.median(bootstrap, axis=1)
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    ci_low, ci_high = np.percentile(medians, [2.5, 97.5])
    return {
        "count": int(array.size),
        "mean_nm": float(np.mean(array)),
        "std_nm": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median_nm": float(median),
        "q1_nm": float(q1),
        "q3_nm": float(q3),
        "iqr_nm": float(q3 - q1),
        "median_bootstrap_95ci_nm": [float(ci_low), float(ci_high)],
        "min_nm": float(np.min(array)),
        "max_nm": float(np.max(array)),
    }


def _median_baseline(values: Sequence[float]) -> tuple[float, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot estimate a median from empty values")
    q1, median, q3 = np.percentile(array, [25.0, 50.0, 75.0])
    return float(median), {
        "q1_nm": float(q1),
        "q3_nm": float(q3),
        "iqr_nm": float(q3 - q1),
    }


def select_median_nearby_rows(
    rows: Sequence[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 0:
        raise ValueError("median-nearby count cannot be negative")
    if count == 0:
        return list(rows), {
            "enabled": False,
            "count": 0,
            "method": "disabled",
        }

    eligible_by_direction: dict[str, set[int]] = {}
    baseline_by_direction: dict[str, tuple[float | None, dict[str, Any]]] = {}
    direction_details: dict[str, Any] = {}
    for direction, value_key in (("xy", "fwhm_xy_nm"), ("z", "fwhm_z_nm")):
        tukey_key = f"tukey_excluded_{value_key}"
        eligible = [
            (index, row)
            for index, row in enumerate(rows)
            if row.get("accepted")
            and row.get(value_key) is not None
            and not row.get(tukey_key, False)
            and np.isfinite(float(row[value_key]))
        ]
        baseline_nm = None
        baseline_details: dict[str, Any] = {}
        if eligible:
            baseline_nm, baseline_details = _median_baseline(
                [float(row[value_key]) for _, row in eligible]
            )
        eligible_by_direction[direction] = {index for index, _ in eligible}
        baseline_by_direction[direction] = (baseline_nm, baseline_details)
        direction_details[direction] = {
            "eligible_count": len(eligible),
            "selected_count": 0,
            "baseline_nm": baseline_nm,
            **baseline_details,
        }

    jointly_eligible = eligible_by_direction["xy"] & eligible_by_direction["z"]
    selected_indices: set[int] = set()
    if jointly_eligible:
        xy_baseline, xy_details = baseline_by_direction["xy"]
        z_baseline, z_details = baseline_by_direction["z"]
        assert xy_baseline is not None and z_baseline is not None
        xy_scale = max(float(xy_details["iqr_nm"]), np.finfo(float).eps)
        z_scale = max(float(z_details["iqr_nm"]), np.finfo(float).eps)
        ranked = sorted(
            jointly_eligible,
            key=lambda index: (
                max(
                    abs(float(rows[index]["fwhm_xy_nm"]) - xy_baseline) / xy_scale,
                    abs(float(rows[index]["fwhm_z_nm"]) - z_baseline) / z_scale,
                ),
                abs(float(rows[index]["fwhm_xy_nm"]) - xy_baseline) / xy_scale
                + abs(float(rows[index]["fwhm_z_nm"]) - z_baseline) / z_scale,
                index,
            ),
        )
        selected_indices = set(ranked[:count])

    for direction in ("xy", "z"):
        direction_details[direction]["selected_count"] = len(selected_indices)
    for index, row in enumerate(rows):
        selected = index in selected_indices
        row["median_selected_xy"] = selected
        row["median_selected_z"] = selected

    return [
        row for index, row in enumerate(rows) if index in selected_indices
    ], {
        "enabled": True,
        "count": count,
        "required_directions": ["xy", "z"],
        "jointly_eligible_count": len(jointly_eligible),
        "selected_count": len(selected_indices),
        "method": "joint normalized worst-distance to XY/Z median baselines; shared ROI retained",
        **direction_details,
    }


def build_summary(
    rows: Sequence[dict[str, Any]],
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
    detection_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tukey_xy = apply_tukey_outer_fence(
        rows,
        "fwhm_xy_nm",
        minimum_count=run_parameter["tukey_minimum_count"],
    )
    tukey_z = apply_tukey_outer_fence(
        rows,
        "fwhm_z_nm",
        minimum_count=run_parameter["tukey_minimum_count"],
    )
    output_rows, median_selection = select_median_nearby_rows(
        rows,
        int(run_parameter.get("median_nearby_count", 0)),
    )
    accepted = [row for row in rows if row["accepted"]]
    output_accepted = [row for row in output_rows if row["accepted"]]
    dimensions = {}
    for key in ("fwhm_x_nm", "fwhm_y_nm", "fwhm_xy_nm", "fwhm_z_nm"):
        tukey_key = (
            "tukey_excluded_fwhm_z_nm"
            if key == "fwhm_z_nm"
            else "tukey_excluded_fwhm_xy_nm"
        )
        values = [
            float(row[key])
            for row in output_accepted
            if row.get(key) is not None and not row.get(tukey_key, False)
        ]
        dimensions[key] = _summary_statistics(values)

    per_file = {}
    for source in sorted({str(row["source"]) for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        source_accepted = [row for row in source_rows if row["accepted"]]
        source_output_rows = [row for row in output_rows if row["source"] == source]
        source_output_accepted = [
            row for row in source_output_rows if row["accepted"]
        ]
        per_file[source] = {
            "candidate_count": len(source_rows),
            "accepted_count": len(source_accepted),
            "saved_count": len(source_output_rows),
            "rejected_count": len(source_rows) - len(source_accepted),
            "rejection_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in source_rows
                        for reason in str(row["rejection_reasons"]).split(";")
                        if reason
                    ).items()
                )
            ),
            "fwhm_xy_nm": _summary_statistics(
                [
                    float(row["fwhm_xy_nm"])
                    for row in source_output_accepted
                    if row.get("fwhm_xy_nm") is not None
                    and not row.get("tukey_excluded_fwhm_xy_nm", False)
                ]
            ),
            "fwhm_z_nm": _summary_statistics(
                [
                    float(row["fwhm_z_nm"])
                    for row in source_output_accepted
                    if row.get("fwhm_z_nm") is not None
                    and not row.get("tukey_excluded_fwhm_z_nm", False)
                ]
            ),
        }
    detection_metadata = detection_metadata or {}
    candidate_limit_reached = any(
        metadata.get("candidate_limit_reached", False)
        for metadata in detection_metadata.values()
    )
    final_xy_count = dimensions["fwhm_xy_nm"]["count"]
    result_status = (
        "too_many_candidates"
        if candidate_limit_reached
        else sample_status(final_xy_count)
    )
    return {
        "method": "automatic bead detection and offset-Gaussian profile fitting",
        "fwhm_definition": "2*sqrt(2*ln(2))*sigma",
        "reported_values": "observed FWHM; bead-diameter deconvolution not applied",
        "opt": vars(opt),
        "run_parameter": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in run_parameter.items()
        },
        "candidate_count": len(rows),
        "accepted_count": len(accepted),
        "saved_row_count": len(output_rows),
        "final_xy_roi_count": final_xy_count,
        "final_z_roi_count": dimensions["fwhm_z_nm"]["count"],
        "rejected_count": len(rows) - len(accepted),
        "status": result_status,
        "status_label_zh": (
            "候选太多"
            if result_status == "too_many_candidates"
            else {
                "normal": "正常",
                "sample_warning": "合格ROI少于30个",
                "insufficient_sample": "样本量不足",
            }[result_status]
        ),
        "formal_result": not candidate_limit_reached,
        "tukey": {"xy": tukey_xy, "z": tukey_z},
        "median_selection": median_selection,
        "rejection_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in str(row["rejection_reasons"]).split(";")
                    if reason
                ).items()
            )
        ),
        "dimensions": dimensions,
        "files": per_file,
        "detection": detection_metadata,
    }


def _write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys()) if rows else [
        "source",
        "candidate_id",
        "accepted",
        "rejection_reasons",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_overlay(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    projection = np.max(image, axis=0) if image.ndim == 3 else image
    finite = projection[np.isfinite(projection)]
    low, high = (
        np.percentile(finite, [1.0, 99.8]) if finite.size else (0.0, 1.0)
    )
    figure, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(projection, cmap="gray", vmin=low, vmax=max(high, low + 1e-12))
    for accepted, color, label in (
        (True, "#2ca02c", "accepted"),
        (False, "#d62728", "rejected"),
    ):
        selected = [row for row in rows if bool(row["accepted"]) is accepted]
        if selected:
            axis.scatter(
                [row["x_px"] for row in selected],
                [row["y_px"] for row in selected],
                s=28,
                facecolors="none",
                edgecolors=color,
                linewidths=0.9,
                label=label,
            )
            for row in selected:
                axis.annotate(
                    str(row["candidate_id"]),
                    (row["x_px"], row["y_px"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    color=color,
                    fontsize=6,
                    fontweight="bold",
                    path_effects=[
                        path_effects.withStroke(linewidth=1.4, foreground="black")
                    ],
                )
    axis.set_title("Automatic bead candidates")
    axis.set_xlabel("x (px)")
    axis.set_ylabel("y (px)")
    if rows:
        axis.legend(loc="upper right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, [1.0, 99.8])
    return float(low), float(max(high, low + 1e-12))


def _add_sampling_rectangle(
    axis: plt.Axes,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
    color: str,
) -> None:
    axis.add_patch(
        Rectangle(
            (x_start - 0.5, y_start - 0.5),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=1.2,
        )
    )


def _plot_roi_overview(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    run_parameter: dict[str, Any],
    path: Path,
) -> None:
    projection = np.max(image, axis=0) if image.ndim == 3 else image
    low, high = _display_limits(projection)
    radius = run_parameter["roi_radius_xy_px"]
    half_width = run_parameter["profile_half_width_px"]
    accepted = [row for row in rows if row["accepted"]]

    figure, axis = plt.subplots(figsize=(12, 12), constrained_layout=True)
    axis.imshow(projection, cmap="gray", vmin=low, vmax=high)
    margin_y = math.ceil(
        projection.shape[0] * run_parameter["edge_margin_fraction"]
    )
    margin_x = math.ceil(
        projection.shape[1] * run_parameter["edge_margin_fraction"]
    )
    axis.add_patch(
        Rectangle(
            (margin_x - 0.5, margin_y - 0.5),
            projection.shape[1] - 2 * margin_x,
            projection.shape[0] - 2 * margin_y,
            fill=False,
            edgecolor="#ff4fd8",
            linestyle="--",
            linewidth=1.0,
            label="allowed central field",
        )
    )
    for row in accepted:
        x = int(row["x_px"])
        y = int(row["y_px"])
        axis.plot(
            [x - radius, x + radius],
            [y, y],
            color="#00b7ff",
            linewidth=0.7,
        )
        axis.plot(
            [x, x],
            [y - radius, y + radius],
            color="#ff9f1c",
            linewidth=0.7,
        )
        axis.text(
            x + half_width + 1,
            y - half_width - 1,
            str(row["candidate_id"]),
            color="#fff176",
            fontsize=4.5,
        )

    rejected = [row for row in rows if not row["accepted"]]
    if rejected:
        axis.scatter(
            [row["x_px"] for row in rejected],
            [row["y_px"] for row in rejected],
            marker="x",
            s=24,
            color="#ff1744",
            linewidths=0.9,
            label="rejected",
        )
    axis.legend(loc="upper right")
    axis.set_title(
        "Accepted profile ROI overview on original-image XY maximum projection\n"
        "cyan: X profile; orange: Y profile; yellow: candidate ID; "
        "magenta: allowed field boundary"
    )
    axis.set_xlabel("x (px)")
    axis.set_ylabel("y (px)")
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _plot_roi_verification_pdf(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    run_parameter: dict[str, Any],
    path: Path,
    rows_per_page: int = 6,
) -> None:
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        return

    radius = run_parameter["roi_radius_xy_px"]
    half_width = run_parameter["profile_half_width_px"]
    crop_radius = radius + 3
    z_start = run_parameter["exclude_z_start"]
    z_stop = (
        image.shape[0] - run_parameter["exclude_z_end"]
        if image.ndim == 3
        else 0
    )
    panels_per_row = 3 if image.ndim == 3 else 1

    with PdfPages(path) as pdf:
        for page_start in range(0, len(accepted), rows_per_page):
            page_rows = accepted[page_start : page_start + rows_per_page]
            figure, axes = plt.subplots(
                len(page_rows),
                panels_per_row,
                figsize=(16.5, 11.7),
                squeeze=False,
                constrained_layout=True,
            )
            for row_index, row in enumerate(page_rows):
                x = int(row["x_px"])
                y = int(row["y_px"])
                z = int(row["z_px"]) if image.ndim == 3 else None
                x0 = max(0, x - crop_radius)
                x1 = min(image.shape[-1], x + crop_radius + 1)
                y0 = max(0, y - crop_radius)
                y1 = min(image.shape[-2], y + crop_radius + 1)
                xy_image = image[z, y0:y1, x0:x1] if z is not None else image[y0:y1, x0:x1]
                low, high = _display_limits(xy_image)
                xy_axis = axes[row_index, 0]
                xy_axis.imshow(
                    xy_image,
                    cmap="gray",
                    vmin=low,
                    vmax=high,
                    origin="upper",
                )
                _add_sampling_rectangle(
                    xy_axis,
                    x - radius - x0,
                    y - half_width - y0,
                    2 * radius + 1,
                    2 * half_width + 1,
                    "#00b7ff",
                )
                _add_sampling_rectangle(
                    xy_axis,
                    x - half_width - x0,
                    y - radius - y0,
                    2 * half_width + 1,
                    2 * radius + 1,
                    "#ff9f1c",
                )
                xy_axis.set_title(
                    f"#{row['candidate_id']} XY at z={z if z is not None else 0} | "
                    f"FWHM XY={row['fwhm_xy_nm']:.1f} nm",
                    fontsize=8,
                )
                xy_axis.set_xlabel("cyan: X band; orange: Y band", fontsize=7)

                if image.ndim == 3:
                    xz_image = np.mean(
                        image[z_start:z_stop, y - half_width : y + half_width + 1, x0:x1],
                        axis=1,
                    )
                    low, high = _display_limits(xz_image)
                    xz_axis = axes[row_index, 1]
                    xz_axis.imshow(
                        xz_image,
                        cmap="gray",
                        vmin=low,
                        vmax=high,
                        aspect="auto",
                        origin="upper",
                    )
                    _add_sampling_rectangle(
                        xz_axis,
                        x - half_width - x0,
                        0,
                        2 * half_width + 1,
                        z_stop - z_start,
                        "#00d084",
                    )
                    xz_axis.axhline(z - z_start, color="#fff176", linewidth=0.8)
                    xz_axis.set_title(
                        f"#{row['candidate_id']} XZ | Z band x={x - half_width}:{x + half_width}",
                        fontsize=8,
                    )
                    xz_axis.set_xlabel("green: 3-pixel X width; yellow: center Z", fontsize=7)

                    yz_image = np.mean(
                        image[z_start:z_stop, y0:y1, x - half_width : x + half_width + 1],
                        axis=2,
                    )
                    low, high = _display_limits(yz_image)
                    yz_axis = axes[row_index, 2]
                    yz_axis.imshow(
                        yz_image,
                        cmap="gray",
                        vmin=low,
                        vmax=high,
                        aspect="auto",
                        origin="upper",
                    )
                    _add_sampling_rectangle(
                        yz_axis,
                        y - half_width - y0,
                        0,
                        2 * half_width + 1,
                        z_stop - z_start,
                        "#00d084",
                    )
                    yz_axis.axhline(z - z_start, color="#fff176", linewidth=0.8)
                    yz_axis.set_title(
                        f"#{row['candidate_id']} YZ | Z FWHM={row['fwhm_z_nm']:.1f} nm",
                        fontsize=8,
                    )
                    yz_axis.set_xlabel("green: 3-pixel Y width; yellow: center Z", fontsize=7)

                for axis in axes[row_index]:
                    axis.set_xticks([])
                    axis.set_yticks([])

            page_number = page_start // rows_per_page + 1
            page_count = math.ceil(len(accepted) / rows_per_page)
            figure.suptitle(
                "Original-image profile ROI verification "
                f"(accepted beads {page_start + 1}-{page_start + len(page_rows)}"
                f" of {len(accepted)}; page {page_number}/{page_count})",
                fontsize=12,
            )
            pdf.savefig(figure, dpi=180)
            plt.close(figure)


def _plot_profile_image(
    image: np.ndarray,
    row: dict[str, Any],
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
    path: Path,
) -> None:
    coordinate = (
        (int(row["z_px"]), int(row["y_px"]), int(row["x_px"]))
        if image.ndim == 3
        else (int(row["y_px"]), int(row["x_px"]))
    )
    profiles, expected_centers = _axis_profiles(
        image,
        coordinate,
        run_parameter,
    )
    axes_names = ("x", "y", "z")
    figure, axes = plt.subplots(
        len(axes_names),
        figsize=(12, 3.6),
        constrained_layout=True,
    )
    for axis, axis_name in zip(axes, axes_names):
        if axis_name not in profiles:
            axis.axis("off")
            continue
        profile = profiles[axis_name]
        spacing = (
            opt.pixel_size_z_nm
            if axis_name == "z"
            else opt.pixel_size_xy_nm
        )
        coordinate_nm = np.arange(profile.size) * spacing
        axis.plot(coordinate_nm, profile, "o", ms=3, label="data")
        fit = fit_gaussian_profile(
            profile,
            spacing_nm=spacing,
            expected_center_px=expected_centers[axis_name],
        )
        dense = np.linspace(coordinate_nm[0], coordinate_nm[-1], 300)
        axis.plot(
            dense,
            gaussian_with_offset(
                dense,
                fit.amplitude,
                fit.center_nm,
                fit.sigma_nm,
                fit.offset,
            ),
            "-",
            lw=1.2,
            label=f"fit, FWHM={fit.fwhm_nm:.1f} nm",
        )
        axis.set_title(axis_name.upper())
        axis.set_xlabel("position (nm)")
        axis.set_ylabel("intensity")
        axis.legend(fontsize=8)
    figure.suptitle(f"Accepted ROI {int(float(row['candidate_id']))}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_accepted_profile_images(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    accepted = [row for row in rows if bool(row["accepted"])]
    for index, row in enumerate(accepted, start=1):
        _plot_profile_image(
            image,
            row,
            opt,
            run_parameter,
            output_dir / f"roi-{index:04d}.png",
        )


def _plot_boxplot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    accepted = [row for row in rows if row["accepted"]]
    lateral_values = [
        float(row["fwhm_xy_nm"])
        for row in accepted
        if row.get("fwhm_xy_nm") is not None
        and not row.get("tukey_excluded_fwhm_xy_nm", False)
    ]
    axial_values = [
        float(row["fwhm_z_nm"])
        for row in accepted
        if row.get("fwhm_z_nm") is not None
        and not row.get("tukey_excluded_fwhm_z_nm", False)
    ]
    plot_resolution_boxplot(lateral_values, axial_values, path)


def find_tiff_files(input_path: Path, recursive: bool = False) -> list[Path]:
    return common_find_tiff_files(input_path, recursive=recursive)


def _precompute_review_assets(output_path: Path) -> int:
    from fwhm_reviewer import ReviewProject

    project = ReviewProject(output_path)
    items = project.manifest()["items"]
    for item in items:
        project.roi_asset(item["key"])
    return len(items)


def read_tiff_image(
    path: Path,
    axes_override: str = "auto",
    channel: int | None = None,
    timepoint: int | None = None,
) -> tuple[np.ndarray, str]:
    image, effective_axes = read_tiff(
        path,
        axes_override=axes_override,
        channel=channel,
        timepoint=timepoint,
    )
    return _normalise_image_shape(image), effective_axes


def run_analysis(
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
) -> dict[str, Any]:
    input_path = Path(run_parameter["input_path"])
    output_path = Path(run_parameter["output_path"])
    files = find_tiff_files(input_path, recursive=run_parameter["recursive"])
    output_path.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    all_detection_metadata: dict[str, dict[str, Any]] = {}
    per_file_outputs: dict[str, Path] = {}
    is_batch = len(files) > 1

    for tiff_path in files:
        image, image_axes = read_tiff_image(
            tiff_path,
            run_parameter["axes"],
            run_parameter["channel"],
            run_parameter["timepoint"],
        )
        source_name = str(tiff_path.resolve())
        rows, _, _, detection_metadata = analyze_image(
            image, source_name, opt, run_parameter
        )
        all_rows.extend(rows)
        detection_metadata.update(
            {
                "image_shape": list(image.shape),
                "image_dtype": str(image.dtype),
                "image_axes": image_axes,
            }
        )
        all_detection_metadata[source_name] = detection_metadata
        file_output = output_path
        if is_batch:
            file_output = output_path / "per-file" / _clean_stem(tiff_path)
            file_output.mkdir(parents=True, exist_ok=False)
            per_file_outputs[source_name] = file_output

    summary = build_summary(
        all_rows,
        opt,
        run_parameter,
        all_detection_metadata,
    )
    output_rows = (
        [
            row
            for row in all_rows
            if row.get("median_selected_xy") and row.get("median_selected_z")
        ]
        if summary["median_selection"]["enabled"]
        else all_rows
    )
    for tiff_path in files:
        source_name = str(tiff_path.resolve())
        image, _ = read_tiff_image(
            tiff_path,
            run_parameter["axes"],
            run_parameter["channel"],
            run_parameter["timepoint"],
        )
        file_output = output_path
        if is_batch:
            file_output = per_file_outputs[source_name]
        _plot_overlay(
            image,
            [row for row in output_rows if row["source"] == source_name],
            file_output / "candidates.png",
        )
    summary["mode"] = "beads"
    _write_csv(output_rows, output_path / "fwhm_measurements.csv")
    (output_path / "fwhm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    summary["review_asset_count"] = _precompute_review_assets(output_path)
    (output_path / "fwhm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _plot_boxplot(output_rows, output_path / "fwhm_boxplot.png")
    if is_batch:
        for source, source_summary in summary["files"].items():
            source_rows = [row for row in output_rows if str(row["source"]) == source]
            file_output = per_file_outputs[source]
            _write_csv(source_rows, file_output / "fwhm_measurements.csv")
            (file_output / "fwhm_summary.json").write_text(
                json.dumps(
                    {
                        "mode": "beads",
                        "source_file": source,
                        **source_summary,
                        "batch_summary": str(output_path / "fwhm_summary.json"),
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
    index_entries = [
        ("SUMMARY.md", "汇总与文件索引。"),
        ("fwhm_measurements.csv", "输出 ROI 数值与判定；可只含中位数附近的保留 ROI。"),
        ("fwhm_summary.json", "完整参数与统计。"),
        ("fwhm_boxplot.png", "输出 ROI 的横向、轴向箱线图。"),
        ("review-assets/", "审核页面预生成的自动合格 ROI 图。"),
        ("reviewed/", "人工审核后生成。"),
    ]
    if is_batch:
        index_entries.append(
            ("per-file/", "每个 TIFF 的 CSV、JSON 和带编号候选图。")
        )
    else:
        index_entries.extend(
            [
                ("candidates.png", "输出候选的带 candidate ID 图。"),
            ]
        )
    write_summary_markdown(summary, output_path / "SUMMARY.md", index_entries)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatically detect fluorescent beads and measure XY/Z FWHM."
    )
    parser.add_argument("input", type=Path, help="TIFF file or directory of TIFF files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd() / "fwhm_results",
        help="output directory (default: ./fwhm_results)",
    )
    parser.add_argument("--pixel-size-xy-nm", type=float, default=30.1)
    parser.add_argument("--pixel-size-z-nm", type=float, default=96.9)
    parser.add_argument(
        "--axes",
        choices=("auto", "YX", "ZYX"),
        default="auto",
        help="TIFF axes; auto accepts only metadata-resolved YX or ZYX",
    )
    parser.add_argument("--detection-snr", type=float, default=100.0)
    parser.add_argument("--min-fit-snr", type=float, default=5.0)
    parser.add_argument("--min-fit-r2", type=float, default=0.90)
    parser.add_argument("--max-center-offset-px", type=float, default=2.0)
    parser.add_argument("--min-z-fit-half-range-sigma", type=float, default=2.0)
    parser.add_argument("--min-distance-xy-px", type=int, default=6)
    parser.add_argument("--min-distance-z-px", type=int, default=2)
    parser.add_argument("--roi-radius-xy-px", type=int, default=6)
    parser.add_argument(
        "--roi-radius-z-px",
        type=int,
        default=0,
        help="0 uses all non-excluded Z slices",
    )
    parser.add_argument("--profile-half-width-px", type=int, default=1)
    parser.add_argument("--edge-margin-fraction", type=float, default=0.10)
    parser.add_argument("--shape-radius-xy-px", type=int, default=8)
    parser.add_argument("--max-lateral-moment-ratio", type=float, default=1.25)
    parser.add_argument(
        "--max-lateral-centroid-offset-px",
        type=float,
        default=0.75,
    )
    parser.add_argument("--min-isolation-xy-px", type=float, default=13.0)
    parser.add_argument("--max-isolation-z-px", type=int, default=2)
    parser.add_argument("--exclude-z-start", type=int, default=0)
    parser.add_argument("--exclude-z-end", type=int, default=0)
    parser.add_argument("--min-fwhm-xy-nm", type=float, default=40.0)
    parser.add_argument("--max-fwhm-xy-nm", type=float, default=1000.0)
    parser.add_argument("--min-fwhm-z-nm", type=float, default=100.0)
    parser.add_argument("--max-fwhm-z-nm", type=float, default=3000.0)
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--diagnostic-count", type=int, default=8)
    parser.add_argument(
        "--median-nearby-count",
        type=int,
        default=0,
        metavar="N",
        help="save N shared accepted ROI nearest to the XY/Z medians; 0 saves all",
    )
    return parser


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def parameters_from_args(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    _positive(args.pixel_size_xy_nm, "pixel size XY")
    _positive(args.pixel_size_z_nm, "pixel size Z")
    if args.exclude_z_start < 0 or args.exclude_z_end < 0:
        raise ValueError("excluded Z slice counts cannot be negative")
    if args.roi_radius_xy_px < 2 or args.roi_radius_z_px < 0:
        raise ValueError("ROI radii are invalid")
    if (
        args.min_distance_xy_px < 1
        or args.min_distance_z_px < 0
        or args.profile_half_width_px < 0
        or args.shape_radius_xy_px < args.roi_radius_xy_px
        or args.max_isolation_z_px < 0
        or args.max_candidates < 1
        or args.diagnostic_count < 0
        or args.median_nearby_count < 0
    ):
        raise ValueError("distance, profile, candidate, or diagnostic limits are invalid")
    _positive(args.max_center_offset_px, "maximum center offset")
    _positive(args.min_z_fit_half_range_sigma, "minimum Z fit half range")
    if not 0 <= args.edge_margin_fraction < 0.5:
        raise ValueError("edge margin fraction must be in [0, 0.5)")
    _positive(args.max_lateral_moment_ratio, "maximum lateral moment ratio")
    _positive(
        args.max_lateral_centroid_offset_px,
        "maximum lateral centroid offset",
    )
    _positive(args.min_isolation_xy_px, "minimum isolation distance")

    opt = argparse.Namespace()
    opt.pixel_size_xy_nm = args.pixel_size_xy_nm
    opt.pixel_size_z_nm = args.pixel_size_z_nm

    run_parameter = default_run_parameter()
    run_parameter.update(
        {
            "input_path": args.input,
            "output_path": args.output,
            "axes": args.axes,
            "detection_snr": args.detection_snr,
            "min_fit_snr": args.min_fit_snr,
            "min_fit_r2": args.min_fit_r2,
            "max_center_offset_px": args.max_center_offset_px,
            "min_z_fit_half_range_sigma": args.min_z_fit_half_range_sigma,
            "min_distance_xy_px": args.min_distance_xy_px,
            "min_distance_z_px": args.min_distance_z_px,
            "roi_radius_xy_px": args.roi_radius_xy_px,
            "roi_radius_z_px": args.roi_radius_z_px,
            "profile_half_width_px": args.profile_half_width_px,
            "edge_margin_fraction": args.edge_margin_fraction,
            "shape_radius_xy_px": args.shape_radius_xy_px,
            "max_lateral_moment_ratio": args.max_lateral_moment_ratio,
            "max_lateral_centroid_offset_px": (
                args.max_lateral_centroid_offset_px
            ),
            "min_isolation_xy_px": args.min_isolation_xy_px,
            "max_isolation_z_px": args.max_isolation_z_px,
            "exclude_z_start": args.exclude_z_start,
            "exclude_z_end": args.exclude_z_end,
            "min_fwhm_xy_nm": args.min_fwhm_xy_nm,
            "max_fwhm_xy_nm": args.max_fwhm_xy_nm,
            "min_fwhm_z_nm": args.min_fwhm_z_nm,
            "max_fwhm_z_nm": args.max_fwhm_z_nm,
            "max_candidates": args.max_candidates,
            "diagnostic_count": args.diagnostic_count,
            "median_nearby_count": args.median_nearby_count,
        }
    )
    return opt, run_parameter


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    opt, run_parameter = parameters_from_args(args)
    summary = run_analysis(opt, run_parameter)
    print(
        "FWHM analysis complete: "
        f"{summary['accepted_count']}/{summary['candidate_count']} candidates accepted"
    )
    limited_files = [
        source
        for source, metadata in summary["detection"].items()
        if metadata["candidate_limit_reached"]
    ]
    if limited_files:
        print(
            "WARNING: candidate limit reached for "
            f"{len(limited_files)} file(s); inspect fwhm_summary.json"
        )
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
