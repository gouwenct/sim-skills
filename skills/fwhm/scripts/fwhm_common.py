from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tifffile


TIFF_SUFFIXES = {".tif", ".tiff"}


def compact_stem(path: Path, maximum_length: int = 48) -> str:
    if maximum_length < 10:
        raise ValueError("maximum_length must be at least 10")
    source = Path(path)
    raw = source.name if source.is_dir() else source.stem
    safe = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in raw
    )
    if len(safe) <= maximum_length:
        return safe
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{safe[: maximum_length - len(digest) - 1]}-{digest}"


def physical_parameters(
    pixel_size_xy_nm: float,
    pixel_size_z_nm: float,
    theoretical_lateral_resolution_nm: float | None = None,
    theoretical_axial_resolution_nm: float | None = None,
) -> argparse.Namespace:
    values = {
        "pixel_size_xy_nm": pixel_size_xy_nm,
        "pixel_size_z_nm": pixel_size_z_nm,
        "theoretical_lateral_resolution_nm": theoretical_lateral_resolution_nm,
        "theoretical_axial_resolution_nm": theoretical_axial_resolution_nm,
    }
    for name, value in values.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    return argparse.Namespace(**values)


def default_output_path(input_path: Path, now: datetime | None = None) -> Path:
    source = input_path.resolve()
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    if source.is_dir():
        result_parent = source.parent / "FWHM-results"
    else:
        result_parent = source.parent.parent / "FWHM-results"
    input_name = compact_stem(source)
    candidate = result_parent / f"{input_name}-{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = result_parent / f"{input_name}-{timestamp}-{suffix:02d}"
        suffix += 1
    return candidate


def find_tiff_files(input_path: Path, recursive: bool = False) -> list[Path]:
    source = input_path.resolve()
    if source.is_file():
        if source.suffix.lower() not in TIFF_SUFFIXES:
            raise ValueError(f"input is not a TIFF file: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    iterator: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    files = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )
    if not files:
        scope = "recursively under" if recursive else "in"
        raise ValueError(f"no TIFF files found {scope} {source}")
    return files


def read_tiff(
    path: Path,
    axes_override: str = "auto",
    channel: int | None = None,
    timepoint: int | None = None,
) -> tuple[np.ndarray, str]:
    with tifffile.TiffFile(path) as tiff:
        series = tiff.series[0]
        image = np.asarray(series.asarray())
        metadata_axes = str(series.axes or "").upper()

    if len(metadata_axes) != image.ndim:
        metadata_axes = ""

    selectors: list[int | slice] = [slice(None)] * image.ndim
    for axis_name, selected in (("T", timepoint), ("C", channel)):
        if axis_name not in metadata_axes:
            if selected is not None:
                raise ValueError(f"--{axis_name.lower()} index was supplied but TIFF has no {axis_name} axis")
            continue
        axis_index = metadata_axes.index(axis_name)
        size = image.shape[axis_index]
        if size > 1 and selected is None:
            option = "--timepoint" if axis_name == "T" else "--channel"
            raise ValueError(
                f"TIFF axes {metadata_axes!r} contain {size} values on {axis_name}; "
                f"select one with {option}"
            )
        chosen = 0 if selected is None else selected
        if not 0 <= chosen < size:
            raise ValueError(f"{axis_name} index {chosen} is outside 0..{size - 1}")
        selectors[axis_index] = chosen

    image = image[tuple(selectors)]
    selected_axes = "".join(
        axis
        for axis, selector in zip(metadata_axes, selectors)
        if isinstance(selector, slice)
    )
    retained = [
        (axis, size)
        for axis, size in zip(selected_axes, image.shape)
        if size != 1
    ]
    image = np.squeeze(image)
    retained_axes = "".join(axis for axis, _ in retained)

    requested = axes_override.upper()
    if requested not in {"AUTO", "YX", "ZYX"}:
        raise ValueError("axes must be auto, YX, or ZYX")
    if requested != "AUTO":
        effective_axes = requested
    elif retained_axes in {"YX", "ZYX"}:
        effective_axes = retained_axes
    elif any(axis in retained_axes for axis in "TCS"):
        raise ValueError(
            f"TIFF axes {metadata_axes!r} remain ambiguous after selection: {retained_axes!r}"
        )
    elif image.ndim == 2:
        effective_axes = "YX"
    elif image.ndim == 3:
        # Generic QYX/ImageJ stacks without a reliable spatial label use the
        # user-confirmed project default.
        effective_axes = "ZYX"
    else:
        raise ValueError(
            f"TIFF axes {metadata_axes!r} cannot be reduced to YX or ZYX; "
            "select channel/timepoint or provide --axes"
        )

    expected_ndim = len(effective_axes)
    if image.ndim != expected_ndim:
        raise ValueError(
            f"axes {effective_axes} require {expected_ndim} dimensions after selection, "
            f"got {image.shape}"
        )
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError(f"unsupported TIFF dtype: {image.dtype}")
    return image, effective_axes


def sample_status(count: int) -> str:
    if count < 10:
        return "insufficient_sample"
    if count < 30:
        return "sample_warning"
    return "normal"


def summary_statistics(
    values: Sequence[float],
    groups: Sequence[str] | None = None,
    seed: int = 0,
    bootstrap_count: int = 2000,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    if groups is not None and len(groups) != array.size:
        raise ValueError("groups must match values")

    rng = np.random.default_rng(seed)
    medians = np.empty(bootstrap_count, dtype=np.float64)
    if groups is None:
        for index in range(bootstrap_count):
            medians[index] = float(
                np.median(rng.choice(array, size=array.size, replace=True))
            )
        bootstrap_method = "ROI bootstrap"
    else:
        group_array = np.asarray(groups, dtype=object)
        unique_groups = np.unique(group_array)
        for index in range(bootstrap_count):
            sampled_groups = rng.choice(
                unique_groups,
                size=unique_groups.size,
                replace=True,
            )
            sampled_indices = np.concatenate(
                [np.flatnonzero(group_array == group) for group in sampled_groups]
            )
            medians[index] = float(np.median(array[sampled_indices]))
        bootstrap_method = "cluster bootstrap by internal line_id"

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
        "bootstrap_method": bootstrap_method,
        "min_nm": float(np.min(array)),
        "max_nm": float(np.max(array)),
    }


def apply_tukey_outer_fence(
    rows: Sequence[dict[str, Any]],
    value_key: str,
    accepted_key: str = "accepted",
    minimum_count: int = 10,
) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if bool(row.get(accepted_key))
        and row.get(value_key) is not None
        and np.isfinite(float(row[value_key]))
    ]
    flag_key = f"tukey_excluded_{value_key}"
    for row in rows:
        row[flag_key] = False
    if len(eligible) < minimum_count:
        return {
            "applied": False,
            "eligible_count": len(eligible),
            "excluded_count": 0,
            "upper_outer_fence_nm": None,
        }

    values = np.asarray([float(row[value_key]) for row in eligible], dtype=np.float64)
    q1, q3 = np.percentile(values, [25.0, 75.0])
    upper_fence = float(q3 + 3.0 * (q3 - q1))
    excluded = 0
    for row in eligible:
        if float(row[value_key]) > upper_fence:
            row[flag_key] = True
            excluded += 1
    return {
        "applied": True,
        "eligible_count": len(eligible),
        "excluded_count": excluded,
        "q1_nm": float(q1),
        "q3_nm": float(q3),
        "iqr_nm": float(q3 - q1),
        "upper_outer_fence_nm": upper_fence,
        "method": "Tukey upper outer fence: Q3 + 3*IQR",
    }


def write_rows_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_summary_markdown(
    summary: dict[str, Any],
    path: Path,
    index_entries: Sequence[tuple[str, str]],
) -> None:
    dimensions = summary.get("dimensions", {})
    result_rows = []
    for label, key in (("横向", "fwhm_xy_nm"), ("轴向", "fwhm_z_nm")):
        values = dimensions.get(key, {})
        if not values or not values.get("count"):
            continue
        ci = values.get("median_bootstrap_95ci_nm", [None, None])
        ci_text = (
            f"{float(ci[0]):.2f}-{float(ci[1]):.2f}"
            if len(ci) == 2 and ci[0] is not None and ci[1] is not None
            else "-"
        )
        result_rows.append(
            f"| {label} | {int(values['count'])} | "
            f"{float(values['median_nm']):.2f} | "
            f"{float(values['iqr_nm']):.2f} | {ci_text} |"
        )

    mode = str(summary.get("mode", "unknown"))
    status = summary.get("status", summary.get("status_xy", "unknown"))
    candidate_count = summary.get("candidate_count", "-")
    accepted_count = summary.get(
        "accepted_count",
        summary.get(
            "retained_roi_count",
            summary.get("accepted_roi_count_xy", "-"),
        ),
    )
    content = [
        "# FWHM 汇总",
        "",
        f"- 模式：{mode}",
        f"- 状态：{status}",
        f"- 候选：{candidate_count}",
        f"- 合格：{accepted_count}",
        "",
        "## 结果",
        "",
        "| 方向 | N | 中位数 (nm) | IQR (nm) | 95% CI (nm) |",
        "|---|---:|---:|---:|---:|",
        *result_rows,
        "",
        "## Index",
        "",
        "| 路径 | 功能 |",
        "|---|---|",
        *[f"| `{name}` | {purpose} |" for name, purpose in index_entries],
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def plot_resolution_boxplot(
    lateral_values: Sequence[float],
    axial_values: Sequence[float],
    path: Path,
    title: str = "FWHM resolution",
) -> None:
    import matplotlib.pyplot as plt

    groups = []
    for label, values, color in (
        ("Lateral", lateral_values, "#218C83"),
        ("Axial", axial_values, "#3D73B9"),
    ):
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if array.size:
            groups.append((label, array, color))
    if not groups:
        return

    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    rng = np.random.default_rng(0)
    maximum = max(float(np.max(values)) for _, values, _ in groups)
    for position, (label, values, color) in enumerate(groups, start=1):
        axis.boxplot(
            [values],
            positions=[position],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops={
                "facecolor": color,
                "edgecolor": color,
                "alpha": 0.38,
                "linewidth": 1.5,
                "zorder": 1,
            },
            whiskerprops={"color": color, "linewidth": 2.0, "zorder": 4},
            capprops={"color": color, "linewidth": 2.0, "zorder": 4},
            medianprops={"color": "black", "linewidth": 2.6, "zorder": 6},
        )
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        axis.scatter(
            position + jitter,
            values,
            s=54,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            alpha=0.92,
            zorder=3,
        )
        q1, median, q3 = np.percentile(values, [25.0, 50.0, 75.0])
        axis.hlines(
            [q1, q3],
            position - 0.25,
            position + 0.25,
            colors=color,
            linewidth=1.8,
            zorder=5,
        )
        axis.hlines(
            median,
            position - 0.25,
            position + 0.25,
            colors="black",
            linewidth=2.8,
            zorder=6,
        )
        axis.text(
            position,
            0.965,
            f"n={values.size}  |  Median={median:.2f} nm  |  IQR={q3 - q1:.2f} nm",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
        )

    axis.set_title(title, fontsize=22, fontweight="bold", pad=18)
    axis.set_ylabel("Observed FWHM (nm)", fontsize=15)
    axis.set_xticks(
        range(1, len(groups) + 1),
        [label for label, _, _ in groups],
        fontsize=14,
    )
    axis.set_xlim(0.45, len(groups) + 0.55)
    axis.set_ylim(0, maximum * 1.15)
    axis.grid(axis="y", alpha=0.28)
    axis.tick_params(axis="y", labelsize=12)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
    figure.savefig(path, dpi=220, facecolor="white")
    plt.close(figure)
