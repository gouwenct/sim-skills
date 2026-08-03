from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from skimage.draw import circle_perimeter, line as raster_line
from skimage.filters import sato, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, skeletonize
import tifffile

from fwhm_beads import fit_gaussian_profile, gaussian_with_offset, robust_sigma
from fwhm_common import (
    apply_tukey_outer_fence,
    compact_stem,
    find_tiff_files,
    plot_resolution_boxplot,
    read_tiff,
    sample_status,
    summary_statistics,
    write_json,
    write_rows_csv,
    write_summary_markdown,
)


def default_line_run_parameter() -> dict[str, Any]:
    return {
        "input_path": None,
        "output_path": None,
        "axes": "auto",
        "channel": None,
        "timepoint": None,
        "recursive": False,
        "exclude_z_start": 1,
        "exclude_z_end": 1,
        "edge_margin_fraction": 0.0,
        "minimum_line_length_factor": 2.0,
        "maximum_curvature_degrees": 15.0,
        "minimum_snr": 5.0,
        "nearby_structure_factor": 1.5,
        "maximum_tilt_degrees": 10.0,
        "maximum_rois_per_line": 3,
        "minimum_roi_spacing_factor": 2.0,
        "profile_half_length_factor": 2.0,
        "minimum_axial_half_length_factor": 1.5,
        "profile_average_half_width_px": 1,
        "lateral_plane_mode": "single_peak_layer",
        "minimum_fit_r2": 0.90,
        "maximum_center_offset_samples": 1.0,
        "maximum_gaussian_direct_difference_fraction": 0.20,
        "tukey_minimum_count": 10,
    }


def direct_fwhm(profile: np.ndarray, spacing_nm: float) -> float:
    values = np.asarray(profile, dtype=np.float64)
    if values.size < 5 or not np.all(np.isfinite(values)):
        raise ValueError("profile must contain at least five finite samples")
    edge_count = max(2, values.size // 5)
    baseline = float(np.median(np.concatenate((values[:edge_count], values[-edge_count:]))))
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    if peak <= baseline:
        raise ValueError("profile has no positive peak")
    half = baseline + 0.5 * (peak - baseline)

    left = None
    for index in range(peak_index - 1, -1, -1):
        if values[index] <= half <= values[index + 1]:
            denominator = values[index + 1] - values[index]
            fraction = 0.0 if denominator == 0 else (half - values[index]) / denominator
            left = index + float(fraction)
            break
    right = None
    for index in range(peak_index, values.size - 1):
        if values[index] >= half >= values[index + 1]:
            denominator = values[index] - values[index + 1]
            fraction = 0.0 if denominator == 0 else (values[index] - half) / denominator
            right = index + float(fraction)
            break
    if left is None or right is None or right <= left:
        raise ValueError("profile does not cross half maximum on both sides")
    return float((right - left) * spacing_nm)


def robust_profile_snr(profile: np.ndarray) -> float | None:
    values = np.asarray(profile, dtype=np.float64)
    edge_count = max(2, values.size // 5)
    background_values = np.concatenate((values[:edge_count], values[-edge_count:]))
    background = float(np.median(background_values))
    noise = robust_sigma(background_values)
    if noise <= 0:
        return None
    return float((np.max(values) - background) / noise)


def _angle_between_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = first / max(float(np.linalg.norm(first)), np.finfo(float).eps)
    second = second / max(float(np.linalg.norm(second)), np.finfo(float).eps)
    cosine = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return math.degrees(math.acos(cosine))


def _principal_direction(points_yx: np.ndarray) -> np.ndarray:
    centered = points_yx - np.mean(points_yx, axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction_yx = eigenvectors[:, int(np.argmax(eigenvalues))]
    if direction_yx[1] < 0:
        direction_yx = -direction_yx
    return direction_yx


def _local_geometry(
    component_points: np.ndarray,
    center_yx: np.ndarray,
    half_length_px: float,
    component_lookup: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray | None, float | None, int]:
    component = (
        component_lookup
        if component_lookup is not None
        else {tuple(map(int, point)) for point in component_points}
    )
    center = tuple(np.rint(center_yx).astype(int))
    if center not in component:
        return None, None, 0

    def neighbors(point: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = point
        return [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and (y + dy, x + dx) in component
        ]

    center_neighbors = neighbors(center)
    if len(center_neighbors) != 2:
        return None, None, 1

    # Follow both skeleton arms so diagonal steps count by physical pixel length.
    def trace(first: tuple[int, int]) -> tuple[list[tuple[int, int]], float]:
        points = [center, first]
        length = math.dist(center, first)
        previous, current = center, first
        while length < half_length_px:
            following = [
                point
                for point in neighbors(current)
                if point != previous and point not in points
            ]
            if len(following) != 1:
                break
            following_point = following[0]
            length += math.dist(current, following_point)
            points.append(following_point)
            previous, current = current, following_point
        return points, length

    first_arm, first_length = trace(center_neighbors[0])
    second_arm, second_length = trace(center_neighbors[1])
    local_count = len(first_arm) + len(second_arm) - 1
    if first_length < half_length_px or second_length < half_length_px:
        return None, None, local_count

    local = np.asarray(first_arm[:0:-1] + second_arm, dtype=np.float64)
    direction = _principal_direction(local)
    curvature = _angle_between_degrees(
        _principal_direction(np.asarray(first_arm, dtype=np.float64)),
        _principal_direction(np.asarray(second_arm, dtype=np.float64)),
    )
    return direction, curvature, local_count


def _ridge_skeleton(image_yx: np.ndarray, minimum_object_size: int) -> np.ndarray:
    plane = np.asarray(image_yx, dtype=np.float64)
    finite = np.nan_to_num(
        plane,
        nan=float(np.nanmedian(plane)),
        posinf=0.0,
        neginf=0.0,
    )
    response = sato(finite, sigmas=(1.0, 1.5, 2.0), black_ridges=False)
    positive = response[np.isfinite(response) & (response > 0)]
    if positive.size == 0 or float(np.ptp(positive)) <= np.finfo(float).eps:
        return np.zeros(plane.shape, dtype=bool)
    threshold = float(threshold_otsu(positive))
    mask = response >= threshold
    mask = remove_small_objects(mask, min_size=max(3, minimum_object_size))
    return skeletonize(mask)


def _sample_component_points(
    component_points: np.ndarray,
    image_yx: np.ndarray,
    minimum_spacing_px: float,
    maximum_count: int,
    geometry_half_length_px: float,
    maximum_curvature_degrees: float,
    edge_margin_y: int,
    edge_margin_x: int,
    skeleton: np.ndarray,
    labels: np.ndarray,
    line_id: int,
    nearby_radius_px: float,
    rejection_counts: dict[str, int],
) -> list[tuple[np.ndarray, np.ndarray, float, int]]:
    intensities = image_yx[component_points[:, 0], component_points[:, 1]]
    order = np.argsort(intensities)[::-1]
    selected: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    component_lookup = {tuple(map(int, point)) for point in component_points}
    for index in order:
        point = component_points[index].astype(np.float64)
        direction, curvature, local_count = _local_geometry(
            component_points,
            point,
            geometry_half_length_px,
            component_lookup,
        )
        if direction is None or curvature is None:
            rejection_counts["line_too_short"] += 1
            continue
        if curvature > maximum_curvature_degrees:
            rejection_counts["curved_line"] += 1
            continue
        y, x = point
        if (
            y < edge_margin_y
            or y >= image_yx.shape[0] - edge_margin_y
            or x < edge_margin_x
            or x >= image_yx.shape[1] - edge_margin_x
        ):
            rejection_counts["field_edge"] += 1
            continue
        if _nearby_or_branching(
            skeleton,
            labels,
            line_id,
            point,
            nearby_radius_px,
        ):
            rejection_counts["nearby_or_branching_structure"] += 1
            continue
        if all(
            np.linalg.norm(point - other[0]) >= minimum_spacing_px
            for other in selected
        ):
            selected.append((point, direction, curvature, local_count))
            if len(selected) >= maximum_count:
                break
    return selected


def _extract_perpendicular_profile(
    image_yx: np.ndarray,
    center_yx: np.ndarray,
    tangent_yx: np.ndarray,
    half_length_px: int,
    average_half_width_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal_yx = np.array([-tangent_yx[1], tangent_yx[0]], dtype=np.float64)
    offsets = np.arange(-half_length_px, half_length_px + 1, dtype=np.float64)
    parallel = np.arange(
        -average_half_width_px,
        average_half_width_px + 1,
        dtype=np.float64,
    )
    samples = []
    for tangent_offset in parallel:
        coordinates = (
            center_yx[:, None]
            + normal_yx[:, None] * offsets[None, :]
            + tangent_yx[:, None] * tangent_offset
        )
        samples.append(map_coordinates(image_yx, coordinates, order=1, mode="nearest"))
    return np.mean(samples, axis=0), offsets, normal_yx


def _nearby_or_branching(
    skeleton: np.ndarray,
    labels: np.ndarray,
    line_id: int,
    center_yx: np.ndarray,
    radius_px: float,
) -> bool:
    y0 = max(0, int(math.floor(center_yx[0] - radius_px)))
    y1 = min(skeleton.shape[0], int(math.ceil(center_yx[0] + radius_px + 1)))
    x0 = max(0, int(math.floor(center_yx[1] - radius_px)))
    x1 = min(skeleton.shape[1], int(math.ceil(center_yx[1] + radius_px + 1)))
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=np.float64)
    inside = (yy + y0 - center_yx[0]) ** 2 + (xx + x0 - center_yx[1]) ** 2 <= radius_px**2
    local_skeleton = skeleton[y0:y1, x0:x1] & inside
    local_labels = labels[y0:y1, x0:x1][local_skeleton]
    if np.any((local_labels != 0) & (local_labels != line_id)):
        return True

    padded = np.pad(skeleton, 1, mode="constant")
    points = np.argwhere(local_skeleton)
    for local_y, local_x in points:
        global_y, global_x = y0 + local_y, x0 + local_x
        neighbors = padded[global_y : global_y + 3, global_x : global_x + 3]
        if int(np.count_nonzero(neighbors)) - 1 > 2:
            return True
    return False


def _fit_profile(
    profile: np.ndarray,
    spacing_nm: float,
    expected_center: float,
    run_parameter: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    values: dict[str, Any] = {
        "fwhm_gaussian_nm": None,
        "fwhm_direct_nm": None,
        "fit_r2": None,
        "fit_center_offset_samples": None,
        "snr": robust_profile_snr(profile),
    }
    try:
        fit = fit_gaussian_profile(profile, spacing_nm, expected_center)
        direct = direct_fwhm(profile, spacing_nm)
    except (RuntimeError, ValueError):
        reasons.append("fit_failed")
        return values, reasons

    values.update(
        {
            "fwhm_gaussian_nm": fit.fwhm_nm,
            "fwhm_direct_nm": direct,
            "fit_r2": fit.r2,
            "fit_center_offset_samples": fit.center_offset_px,
            "fit_amplitude": fit.amplitude,
            "fit_offset": fit.offset,
            "fit_sigma_nm": fit.sigma_nm,
        }
    )
    if values["snr"] is None or values["snr"] < run_parameter["minimum_snr"]:
        reasons.append("low_snr")
    if fit.r2 < run_parameter["minimum_fit_r2"]:
        reasons.append("low_fit_r2")
    if abs(fit.center_offset_px) > run_parameter["maximum_center_offset_samples"]:
        reasons.append("profile_not_centered")
    difference = abs(fit.fwhm_nm - direct) / max(fit.fwhm_nm, direct)
    values["gaussian_direct_difference_fraction"] = difference
    if difference > run_parameter["maximum_gaussian_direct_difference_fraction"]:
        reasons.append("non_gaussian_profile")
    return values, reasons


def analyze_line_image(
    image: np.ndarray,
    source_name: str,
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    volume = np.asarray(image)
    if volume.ndim not in (2, 3):
        raise ValueError(f"line mode expects YX or ZYX data, got {volume.shape}")
    plane = np.max(volume, axis=0) if volume.ndim == 3 else volume
    lateral_resolution_px = (
        opt.theoretical_lateral_resolution_nm / opt.pixel_size_xy_nm
    )
    minimum_length_px = run_parameter["minimum_line_length_factor"] * lateral_resolution_px
    skeleton = _ridge_skeleton(plane, int(math.ceil(minimum_length_px)))
    labels = label(skeleton, connectivity=2)
    half_profile_px = int(
        math.ceil(
            run_parameter["profile_half_length_factor"] * lateral_resolution_px
        )
    )
    profile_padding = (
        half_profile_px + run_parameter["profile_average_half_width_px"]
    )
    edge_margin_y = max(
        profile_padding,
        int(math.ceil(plane.shape[0] * run_parameter["edge_margin_fraction"])),
    )
    edge_margin_x = max(
        profile_padding,
        int(math.ceil(plane.shape[1] * run_parameter["edge_margin_fraction"])),
    )
    prefilter_rejections = {
        "line_too_short": 0,
        "curved_line": 0,
        "field_edge": 0,
        "nearby_or_branching_structure": 0,
    }

    rows: list[dict[str, Any]] = []
    candidate_id = 0
    for region in regionprops(labels):
        component = np.argwhere(labels == region.label)
        if component.shape[0] < minimum_length_px:
            continue
        sampled = _sample_component_points(
            component,
            plane,
            run_parameter["minimum_roi_spacing_factor"] * lateral_resolution_px,
            run_parameter["maximum_rois_per_line"],
            minimum_length_px / 2.0,
            run_parameter["maximum_curvature_degrees"],
            edge_margin_y,
            edge_margin_x,
            skeleton,
            labels,
            int(region.label),
            run_parameter["nearby_structure_factor"] * lateral_resolution_px,
            prefilter_rejections,
        )
        for center_yx, direction, curvature, local_count in sampled:
            candidate_id += 1
            y, x = center_yx
            reasons_xy: list[str] = []
            row: dict[str, Any] = {
                "source_file": source_name,
                "roi_id": candidate_id,
                "line_id": int(region.label),
                "y_px": float(y),
                "x_px": float(x),
                "z_px": None,
                "curvature_degrees": curvature,
                "local_skeleton_pixels": local_count,
            }

            z_profile_full = None
            valid_start = None
            valid_stop = None
            z_center = None
            lateral_plane = plane
            if volume.ndim == 3:
                z_profile_full = np.mean(
                    volume[
                        :,
                        max(0, int(round(y)) - 1) : min(
                            volume.shape[1], int(round(y)) + 2
                        ),
                        max(0, int(round(x)) - 1) : min(
                            volume.shape[2], int(round(x)) + 2
                        ),
                    ],
                    axis=(1, 2),
                )
                valid_start = run_parameter["exclude_z_start"]
                valid_stop = volume.shape[0] - run_parameter["exclude_z_end"]
                valid_profile = z_profile_full[valid_start:valid_stop]
                z_center = valid_start + int(np.argmax(valid_profile))
                row["z_px"] = z_center
                lateral_plane = volume[z_center]

            profile_xy, offsets, normal = _extract_perpendicular_profile(
                lateral_plane,
                center_yx,
                direction,
                half_profile_px,
                run_parameter["profile_average_half_width_px"],
            )
            fit_xy, fit_reasons_xy = _fit_profile(
                profile_xy,
                opt.pixel_size_xy_nm,
                half_profile_px,
                run_parameter,
            )
            reasons_xy.extend(fit_reasons_xy)
            row.update(
                {
                    "tangent_y": float(direction[0]),
                    "tangent_x": float(direction[1]),
                    "normal_y": float(normal[0]),
                    "normal_x": float(normal[1]),
                    "profile_half_length_xy_px": half_profile_px,
                    "fwhm_xy_nm": fit_xy["fwhm_gaussian_nm"],
                    "fwhm_xy_direct_nm": fit_xy["fwhm_direct_nm"],
                    "r2_xy": fit_xy["fit_r2"],
                    "snr_xy": fit_xy["snr"],
                    "center_offset_xy_px": fit_xy["fit_center_offset_samples"],
                    "rejection_reasons_xy": ";".join(dict.fromkeys(reasons_xy)),
                    "accepted_xy": not reasons_xy,
                    "_profile_xy": profile_xy,
                    "_profile_xy_offsets": offsets,
                }
            )

            reasons_z: list[str] = []
            if volume.ndim == 3:
                assert z_profile_full is not None
                assert valid_start is not None and valid_stop is not None
                assert z_center is not None
                valid_profile = z_profile_full[valid_start:valid_stop]
                target_half = int(
                    math.ceil(
                        run_parameter["profile_half_length_factor"]
                        * opt.theoretical_axial_resolution_nm
                        / opt.pixel_size_z_nm
                    )
                )
                minimum_half = int(
                    math.ceil(
                        run_parameter["minimum_axial_half_length_factor"]
                        * opt.theoretical_axial_resolution_nm
                        / opt.pixel_size_z_nm
                    )
                )
                available_half = min(z_center - valid_start, valid_stop - 1 - z_center)
                half_z = min(target_half, available_half)
                if half_z < minimum_half:
                    reasons_z.append("insufficient_z_range")
                if half_z >= 2:
                    profile_z = z_profile_full[z_center - half_z : z_center + half_z + 1]
                    fit_z, fit_reasons_z = _fit_profile(
                        profile_z,
                        opt.pixel_size_z_nm,
                        half_z,
                        run_parameter,
                    )
                    reasons_z.extend(fit_reasons_z)
                else:
                    profile_z = np.asarray([], dtype=np.float64)
                    fit_z = {
                        "fwhm_gaussian_nm": None,
                        "fwhm_direct_nm": None,
                        "fit_r2": None,
                        "snr": None,
                        "fit_center_offset_samples": None,
                    }

                prominence = max(
                    5.0 * robust_sigma(valid_profile),
                    0.2 * float(np.ptp(valid_profile)),
                )
                peaks, _ = find_peaks(valid_profile, prominence=prominence)
                clearance_px = (
                    run_parameter["nearby_structure_factor"]
                    * opt.theoretical_axial_resolution_nm
                    / opt.pixel_size_z_nm
                )
                relative_center = z_center - valid_start
                primary_peak = (
                    int(peaks[np.argmin(np.abs(peaks - relative_center))])
                    if peaks.size
                    else None
                )
                interfering = [
                    peak
                    for peak in peaks
                    if peak != primary_peak
                    and abs(peak - relative_center) <= clearance_px
                ]
                if interfering:
                    reasons_z.append("axial_interference")

                tangent_radius = max(2, int(math.ceil(lateral_resolution_px)))
                tangent_samples = np.linspace(
                    -tangent_radius,
                    tangent_radius,
                    2 * tangent_radius + 1,
                )
                sample_y = np.clip(
                    np.rint(y + direction[0] * tangent_samples).astype(int),
                    0,
                    volume.shape[1] - 1,
                )
                sample_x = np.clip(
                    np.rint(x + direction[1] * tangent_samples).astype(int),
                    0,
                    volume.shape[2] - 1,
                )
                sample_z = np.asarray(
                    [int(np.argmax(volume[:, sy, sx])) for sy, sx in zip(sample_y, sample_x)],
                    dtype=np.float64,
                )
                slope = float(np.polyfit(tangent_samples, sample_z, 1)[0])
                tilt = math.degrees(
                    math.atan(abs(slope) * opt.pixel_size_z_nm / opt.pixel_size_xy_nm)
                )
                row["tilt_degrees"] = tilt
                if tilt > run_parameter["maximum_tilt_degrees"]:
                    reasons_z.append("line_tilt")
                row.update(
                    {
                        "profile_half_length_z_px": half_z,
                        "fwhm_z_nm": fit_z["fwhm_gaussian_nm"],
                        "fwhm_z_direct_nm": fit_z["fwhm_direct_nm"],
                        "r2_z": fit_z["fit_r2"],
                        "snr_z": fit_z["snr"],
                        "center_offset_z_px": fit_z["fit_center_offset_samples"],
                        "rejection_reasons_z": ";".join(dict.fromkeys(reasons_z)),
                        "accepted_z": not reasons_z,
                        "_profile_z": profile_z,
                    }
                )
            else:
                row.update(
                    {
                        "tilt_degrees": None,
                        "profile_half_length_z_px": None,
                        "fwhm_z_nm": None,
                        "fwhm_z_direct_nm": None,
                        "r2_z": None,
                        "snr_z": None,
                        "center_offset_z_px": None,
                        "rejection_reasons_z": "",
                        "accepted_z": False,
                        "_profile_z": np.asarray([], dtype=np.float64),
                    }
                )
            row["accepted"] = bool(row["accepted_xy"]) and (
                bool(row["accepted_z"]) if volume.ndim == 3 else True
            )
            rows.append(row)

    metadata = {
        "skeleton_pixel_count": int(np.count_nonzero(skeleton)),
        "connected_line_count": int(labels.max()),
        "candidate_count": len(rows),
        "prefilter_rejections": prefilter_rejections,
        "image_shape": list(volume.shape),
    }
    return rows, metadata


def _serializable_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _plot_line_overview(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    plane = np.max(image, axis=0) if image.ndim == 3 else image
    low, high = np.percentile(plane[np.isfinite(plane)], [1.0, 99.8])
    figure, axis = plt.subplots(figsize=(10, 10), constrained_layout=True)
    axis.imshow(plane, cmap="gray", vmin=low, vmax=high)
    for row in rows:
        color = "#00d084" if row["accepted_xy"] else "#ff4057"
        half = row["profile_half_length_xy_px"]
        x = row["x_px"]
        y = row["y_px"]
        nx = row["normal_x"]
        ny = row["normal_y"]
        axis.plot(
            [x - nx * half, x + nx * half],
            [y - ny * half, y + ny * half],
            color=color,
            linewidth=1.0,
        )
        axis.text(x + 2, y - 2, str(row["roi_id"]), color="#fff176", fontsize=6)
    axis.set_title("Line FWHM ROI overview (green accepted, red rejected)")
    axis.set_xlabel("x (px)")
    axis.set_ylabel("y (px)")
    figure.savefig(path, dpi=250)
    plt.close(figure)


def _candidate_rgb_stack(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float64)
    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    gray = np.clip((values - low) / (high - low), 0.0, 1.0)
    gray = np.rint(gray * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _paint_pixels(
    image: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    color: tuple[int, int, int],
    width: int = 1,
) -> None:
    for offset_y in range(-width + 1, width):
        for offset_x in range(-width + 1, width):
            yy = np.clip(rows + offset_y, 0, image.shape[0] - 1)
            xx = np.clip(columns + offset_x, 0, image.shape[1] - 1)
            image[yy, xx] = color


def _draw_roi_labels(
    image: np.ndarray,
    labels: Sequence[tuple[int, int, str]],
) -> None:
    canvas = Image.fromarray(image)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=10)
    for x, y, text in labels:
        draw.text((x + 2, y - 2), text, fill=(255, 241, 118), font=font, anchor="ls")
    image[:] = np.asarray(canvas)


def _write_candidate_stacks(
    image: np.ndarray,
    rows: Sequence[dict[str, Any]],
    xy_path: Path,
    z_path: Path,
) -> None:
    if image.ndim != 3:
        return
    xy_stack = _candidate_rgb_stack(image)
    z_stack = xy_stack.copy()
    accepted_color = (0, 255, 132)
    rejected_color = (255, 64, 87)
    labels_by_z: dict[int, list[tuple[int, int, str]]] = {}

    for row in rows:
        z_center = int(row["z_px"])
        y = int(round(row["y_px"]))
        x = int(round(row["x_px"]))
        half_xy = int(row["profile_half_length_xy_px"])
        nx = float(row["normal_x"])
        ny = float(row["normal_y"])
        line_y, line_x = raster_line(
            int(round(y - ny * half_xy)),
            int(round(x - nx * half_xy)),
            int(round(y + ny * half_xy)),
            int(round(x + nx * half_xy)),
        )
        _paint_pixels(
            xy_stack[z_center],
            line_y,
            line_x,
            accepted_color if row["accepted_xy"] else rejected_color,
            width=1,
        )
        labels_by_z.setdefault(z_center, []).append((x, y, str(row["roi_id"])))

        half_z = int(row.get("profile_half_length_z_px") or 0)
        z_start = max(0, z_center - half_z)
        z_stop = min(image.shape[0], z_center + half_z + 1)
        color = accepted_color if row["accepted_z"] else rejected_color
        ring_y, ring_x = circle_perimeter(
            y,
            x,
            5,
            shape=image.shape[1:],
        )
        for z_index in range(z_start, z_stop):
            _paint_pixels(z_stack[z_index], ring_y, ring_x, color, width=1)
        center_y, center_x = raster_line(y, x - 5, y, x + 5)
        _paint_pixels(
            z_stack[z_center],
            center_y,
            center_x,
            color,
            width=1,
        )

    for z_index, labels in labels_by_z.items():
        _draw_roi_labels(xy_stack[z_index], labels)
        _draw_roi_labels(z_stack[z_index], labels)

    tifffile.imwrite(
        xy_path,
        xy_stack,
        photometric="rgb",
        metadata={"axes": "ZYXS"},
    )
    tifffile.imwrite(
        z_path,
        z_stack,
        photometric="rgb",
        metadata={"axes": "ZYXS"},
    )


def _plot_line_profiles(
    rows: Sequence[dict[str, Any]],
    opt: argparse.Namespace,
    path: Path,
) -> None:
    selected = [row for row in rows if row["accepted_xy"]]
    if not selected:
        return
    with PdfPages(path) as pdf:
        for row in selected:
            figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
            profile_xy = row["_profile_xy"]
            coordinate_xy = np.arange(profile_xy.size) * opt.pixel_size_xy_nm
            axes[0].plot(coordinate_xy, profile_xy, "o-", markersize=3, label="data")
            if row["fwhm_xy_nm"] is not None:
                fit = fit_gaussian_profile(
                    profile_xy,
                    opt.pixel_size_xy_nm,
                    row["profile_half_length_xy_px"],
                )
                dense = np.linspace(0, coordinate_xy[-1], 400)
                axes[0].plot(
                    dense,
                    gaussian_with_offset(
                        dense, fit.amplitude, fit.center_nm, fit.sigma_nm, fit.offset
                    ),
                    label=f"Gaussian FWHM {fit.fwhm_nm:.1f} nm",
                )
            axes[0].legend(fontsize=8)
            axes[0].set_title(f"ROI {row['roi_id']} lateral")
            axes[0].set_xlabel("distance (nm)")

            profile_z = row["_profile_z"]
            if profile_z.size:
                axes[1].plot(
                    np.arange(profile_z.size) * opt.pixel_size_z_nm,
                    profile_z,
                    "o-",
                    markersize=3,
                )
                axes[1].set_title(
                    f"axial FWHM {row['fwhm_z_nm']:.1f} nm"
                    if row["fwhm_z_nm"] is not None
                    else "axial rejected"
                )
                axes[1].set_xlabel("z distance (nm)")
            else:
                axes[1].axis("off")
            pdf.savefig(figure, dpi=160)
            plt.close(figure)


def _plot_line_boxplot(rows: Sequence[dict[str, Any]], path: Path) -> None:
    xy = [
        float(row["fwhm_xy_nm"])
        for row in rows
        if row["accepted_xy"]
        and not row.get("tukey_excluded_fwhm_xy_nm", False)
        and row["fwhm_xy_nm"] is not None
    ]
    z = [
        float(row["fwhm_z_nm"])
        for row in rows
        if row["accepted_z"]
        and not row.get("tukey_excluded_fwhm_z_nm", False)
        and row["fwhm_z_nm"] is not None
    ]
    plot_resolution_boxplot(xy, z, path)


def _precompute_review_assets(output_path: Path) -> int:
    from fwhm_reviewer import ReviewProject

    project = ReviewProject(output_path)
    items = project.manifest()["items"]
    for item in items:
        project.roi_asset(item["key"])
    return len(items)


def run_line_analysis(
    opt: argparse.Namespace,
    run_parameter: dict[str, Any],
) -> dict[str, Any]:
    files = find_tiff_files(
        Path(run_parameter["input_path"]),
        recursive=run_parameter["recursive"],
    )
    output_path = Path(run_parameter["output_path"])
    output_path.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    per_file_outputs: dict[str, Path] = {}
    detection: dict[str, Any] = {}
    is_batch = len(files) > 1
    for path in files:
        image, axes = read_tiff(
            path,
            run_parameter["axes"],
            run_parameter["channel"],
            run_parameter["timepoint"],
        )
        rows, metadata = analyze_line_image(image, str(path), opt, run_parameter)
        all_rows.extend(rows)
        rows_by_source[str(path)] = rows
        metadata["axes"] = axes
        detection[str(path)] = metadata
        file_output = output_path
        if is_batch:
            file_output = output_path / "per-file" / compact_stem(path)
            file_output.mkdir(parents=True, exist_ok=False)
            per_file_outputs[str(path)] = file_output
        _plot_line_overview(
            image,
            rows,
            file_output / "candidates.png",
        )
        _write_candidate_stacks(
            image,
            rows,
            file_output / "candidates-XY.tif",
            file_output / "candidates-Z.tif",
        )
        _plot_line_profiles(
            rows,
            opt,
            file_output / "profiles.pdf",
        )

    xy_tukey = apply_tukey_outer_fence(
        all_rows,
        "fwhm_xy_nm",
        accepted_key="accepted_xy",
        minimum_count=run_parameter["tukey_minimum_count"],
    )
    z_tukey = apply_tukey_outer_fence(
        all_rows,
        "fwhm_z_nm",
        accepted_key="accepted_z",
        minimum_count=run_parameter["tukey_minimum_count"],
    )
    serializable = _serializable_rows(all_rows)
    xy_eligible = [
        row
        for row in all_rows
        if row["accepted_xy"]
        and not row["tukey_excluded_fwhm_xy_nm"]
        and row["fwhm_xy_nm"] is not None
    ]
    z_eligible = [
        row
        for row in all_rows
        if row["accepted_z"]
        and not row["tukey_excluded_fwhm_z_nm"]
        and row["fwhm_z_nm"] is not None
    ]
    xy_count = sum(
        bool(row["accepted_xy"]) and not row["tukey_excluded_fwhm_xy_nm"]
        for row in serializable
    )
    z_count = sum(
        bool(row["accepted_z"]) and not row["tukey_excluded_fwhm_z_nm"]
        for row in serializable
    )
    summary = {
        "mode": "lines",
        "status_xy": sample_status(xy_count),
        "status_z": sample_status(z_count),
        "accepted_roi_count_xy": xy_count,
        "accepted_roi_count_z": z_count,
        "candidate_count": len(serializable),
        "tukey": {"xy": xy_tukey, "z": z_tukey},
        "dimensions": {
            "fwhm_xy_nm": summary_statistics(
                [float(row["fwhm_xy_nm"]) for row in xy_eligible],
                [
                    f"{row['source_file']}::{row['line_id']}"
                    for row in xy_eligible
                ],
            ),
            "fwhm_z_nm": summary_statistics(
                [float(row["fwhm_z_nm"]) for row in z_eligible],
                [
                    f"{row['source_file']}::{row['line_id']}"
                    for row in z_eligible
                ],
            ),
        },
        "opt": vars(opt),
        "run_parameter": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in run_parameter.items()
        },
        "detection": detection,
        "scientific_note": (
            "Line FWHM is observed structure width convolved with the PSF; it is "
            "system resolution only when the line diameter is negligible."
        ),
    }
    write_rows_csv(serializable, output_path / "fwhm_measurements.csv")
    write_json(summary, output_path / "fwhm_summary.json")
    summary["review_asset_count"] = _precompute_review_assets(output_path)
    write_json(summary, output_path / "fwhm_summary.json")
    _plot_line_boxplot(serializable, output_path / "fwhm_boxplot.png")
    per_file_sources = rows_by_source if len(files) > 1 else {}
    for source, source_rows in per_file_sources.items():
        file_output = per_file_outputs[source]
        source_rows_for_summary = [dict(row) for row in source_rows]
        source_xy_tukey = apply_tukey_outer_fence(
            source_rows_for_summary,
            "fwhm_xy_nm",
            accepted_key="accepted_xy",
            minimum_count=run_parameter["tukey_minimum_count"],
        )
        source_z_tukey = apply_tukey_outer_fence(
            source_rows_for_summary,
            "fwhm_z_nm",
            accepted_key="accepted_z",
            minimum_count=run_parameter["tukey_minimum_count"],
        )
        source_serializable = _serializable_rows(source_rows_for_summary)
        source_xy_rows = [
            row
            for row in source_rows_for_summary
            if row["accepted_xy"]
            and not row["tukey_excluded_fwhm_xy_nm"]
            and row["fwhm_xy_nm"] is not None
        ]
        source_z_rows = [
            row
            for row in source_rows_for_summary
            if row["accepted_z"]
            and not row["tukey_excluded_fwhm_z_nm"]
            and row["fwhm_z_nm"] is not None
        ]
        source_xy_count = len(source_xy_rows)
        source_z_count = len(source_z_rows)
        source_summary = {
            "mode": "lines",
            "source_file": source,
            "candidate_count": len(source_rows),
            "status_xy": sample_status(source_xy_count),
            "status_z": sample_status(source_z_count),
            "accepted_roi_count_xy": source_xy_count,
            "accepted_roi_count_z": source_z_count,
            "tukey": {"xy": source_xy_tukey, "z": source_z_tukey},
            "dimensions": {
                "fwhm_xy_nm": summary_statistics(
                    [float(row["fwhm_xy_nm"]) for row in source_xy_rows],
                    [f"{source}::{row['line_id']}" for row in source_xy_rows],
                ),
                "fwhm_z_nm": summary_statistics(
                    [float(row["fwhm_z_nm"]) for row in source_z_rows],
                    [f"{source}::{row['line_id']}" for row in source_z_rows],
                ),
            },
            "batch_summary": str(output_path / "fwhm_summary.json"),
        }
        write_rows_csv(source_serializable, file_output / "fwhm_measurements.csv")
        write_json(source_summary, file_output / "fwhm_summary.json")
    index_entries = [
        ("SUMMARY.md", "汇总与文件索引。"),
        ("fwhm_measurements.csv", "全部 ROI 数值与判定。"),
        ("fwhm_summary.json", "完整参数与统计。"),
        ("fwhm_boxplot.png", "横向、轴向箱线图。"),
        ("review-assets/", "审核页面预生成的自动合格 ROI 图。"),
        ("reviewed/", "人工审核后生成。"),
    ]
    if len(files) > 1:
        index_entries.append(
            ("per-file/", "每个 TIFF 的候选图、profile PDF、CSV 和 JSON。")
        )
    else:
        index_entries.extend(
            [
                ("candidates.png", "带 ROI ID 的全部测量候选。"),
                (
                    "candidates-XY.tif",
                    "逐层横向候选标注栈；法线只画在峰值 Z 层。",
                ),
                (
                    "candidates-Z.tif",
                    "逐层轴向候选标注栈；标记覆盖实际 Z profile 范围。",
                ),
                ("profiles.pdf", "自动横向合格 ROI 的曲线与拟合。"),
            ]
        )
    write_summary_markdown(summary, output_path / "SUMMARY.md", index_entries)
    return summary
