from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import fwhm_beads
from fwhm_common import default_output_path, physical_parameters, write_json
import fwhm_lines
from fwhm_reviewer import serve_review


SKILL_ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = SKILL_ROOT / "state" / "last-parameters.json"


def _data_parent(input_path: Path) -> Path:
    source = input_path.resolve()
    return source if source.is_dir() else source.parent


def _load_recent(input_path: Path, reuse_last: bool) -> dict[str, Any]:
    if not STATE_PATH.is_file():
        return {}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    same_parent = state.get("data_parent") == str(_data_parent(input_path))
    return state if reuse_last or same_parent else {}


def _prompt_positive(label: str) -> float:
    if not sys.stdin.isatty():
        raise ValueError(f"{label} is required in non-interactive mode")
    value = float(input(f"{label}: ").strip())
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _resolve_sampling(
    args: argparse.Namespace,
    preset: str,
) -> tuple[float, float]:
    if preset == "mp-sim-beads-v1":
        return (
            30.1 if args.pixel_size_xy_nm is None else args.pixel_size_xy_nm,
            96.9 if args.pixel_size_z_nm is None else args.pixel_size_z_nm,
        )
    recent = _load_recent(args.input, args.reuse_last)
    xy = args.pixel_size_xy_nm or recent.get("pixel_size_xy_nm")
    z = args.pixel_size_z_nm or recent.get("pixel_size_z_nm")
    if xy is None:
        xy = _prompt_positive("XY sampling (nm/pixel)")
    if z is None:
        z = _prompt_positive("Z sampling (nm/layer)")
    return float(xy), float(z)


def _resolve_theory(
    args: argparse.Namespace,
) -> tuple[float, float]:
    recent = _load_recent(args.input, args.reuse_last)
    lateral = (
        args.theoretical_lateral_resolution_nm
        or recent.get("theoretical_lateral_resolution_nm")
    )
    axial = (
        args.theoretical_axial_resolution_nm
        or recent.get("theoretical_axial_resolution_nm")
    )
    if lateral is None:
        lateral = _prompt_positive("Theoretical lateral resolution (nm)")
    if axial is None:
        axial = _prompt_positive("Theoretical axial resolution (nm)")
    return float(lateral), float(axial)


def _save_recent(input_path: Path, opt: argparse.Namespace) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            {
                "data_parent": str(_data_parent(input_path)),
                "pixel_size_xy_nm": opt.pixel_size_xy_nm,
                "pixel_size_z_nm": opt.pixel_size_z_nm,
                "theoretical_lateral_resolution_nm": getattr(
                    opt, "theoretical_lateral_resolution_nm", None
                ),
                "theoretical_axial_resolution_nm": getattr(
                    opt, "theoretical_axial_resolution_nm", None
                ),
            },
            STATE_PATH,
        )
    except OSError as error:
        print(
            f"WARNING: result is complete, but recent parameters were not saved: {error}",
            file=sys.stderr,
        )


def _common_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="TIFF file or directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pixel-size-xy-nm", type=float)
    parser.add_argument("--pixel-size-z-nm", type=float)
    parser.add_argument("--axes", choices=("auto", "YX", "ZYX"), default="auto")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--timepoint", type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--reuse-last", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fwhm",
        description="Offline FWHM measurement for fluorescent beads and thin lines.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    beads = subparsers.add_parser("beads", help="measure isolated fluorescent beads")
    _common_input_arguments(beads)
    beads.add_argument(
        "--preset",
        choices=("mp-sim-beads-v1", "strict-single-bead-v1"),
        default="strict-single-bead-v1",
    )
    beads.add_argument("--exclude-z-start", type=int)
    beads.add_argument("--exclude-z-end", type=int)
    beads.add_argument("--max-candidates", type=int, default=500)
    beads.add_argument("--detection-snr", type=float, default=100.0)
    beads.add_argument("--min-fit-snr", type=float, default=5.0)
    beads.add_argument("--min-fit-r2", type=float, default=0.90)
    beads.add_argument("--edge-margin-fraction", type=float, default=0.10)

    line = subparsers.add_parser("lines", help="measure thin biological lines")
    _common_input_arguments(line)
    line.add_argument("--theoretical-lateral-resolution-nm", type=float)
    line.add_argument("--theoretical-axial-resolution-nm", type=float)
    line.add_argument("--exclude-z-start", type=int, default=1)
    line.add_argument("--exclude-z-end", type=int, default=1)
    line.add_argument("--edge-margin-fraction", type=float, default=0.0)
    line.add_argument("--minimum-snr", type=float, default=5.0)
    line.add_argument("--maximum-curvature-degrees", type=float, default=15.0)
    line.add_argument("--maximum-tilt-degrees", type=float, default=10.0)
    line.add_argument("--maximum-rois-per-line", type=int, default=3)

    review = subparsers.add_parser("review", help="open an existing result for ROI review")
    review.add_argument("result", type=Path)
    review.add_argument("--no-open", action="store_true")
    review.add_argument("--port", type=int, default=0)
    review.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)
    return parser


def _validate_common(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    for name in ("pixel_size_xy_nm", "pixel_size_z_nm"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.channel is not None and args.channel < 0:
        raise ValueError("channel cannot be negative")
    if args.timepoint is not None and args.timepoint < 0:
        raise ValueError("timepoint cannot be negative")


def _validate_analysis_arguments(args: argparse.Namespace, mode: str) -> None:
    for name in ("exclude_z_start", "exclude_z_end"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name} cannot be negative")
    if not 0 <= args.edge_margin_fraction < 0.5:
        raise ValueError("edge_margin_fraction must be in [0, 0.5)")
    if mode == "beads":
        if args.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if args.detection_snr <= 0 or args.min_fit_snr <= 0:
            raise ValueError("detection and fitting SNR thresholds must be positive")
        if not 0 <= args.min_fit_r2 <= 1:
            raise ValueError("min_fit_r2 must be in [0, 1]")
    else:
        if args.minimum_snr <= 0:
            raise ValueError("minimum_snr must be positive")
        if args.maximum_rois_per_line < 1:
            raise ValueError("maximum_rois_per_line must be at least 1")
        if not 0 < args.maximum_curvature_degrees <= 90:
            raise ValueError("maximum_curvature_degrees must be in (0, 90]")
        if not 0 <= args.maximum_tilt_degrees <= 90:
            raise ValueError("maximum_tilt_degrees must be in [0, 90]")


def run_beads(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    _validate_common(args)
    _validate_analysis_arguments(args, "beads")
    pixel_xy, pixel_z = _resolve_sampling(args, args.preset)
    opt = physical_parameters(pixel_xy, pixel_z)
    run_parameter = fwhm_beads.default_run_parameter()
    if args.preset == "mp-sim-beads-v1":
        exclude_start = 0 if args.exclude_z_start is None else args.exclude_z_start
        exclude_end = 2 if args.exclude_z_end is None else args.exclude_z_end
    else:
        exclude_start = 1 if args.exclude_z_start is None else args.exclude_z_start
        exclude_end = 1 if args.exclude_z_end is None else args.exclude_z_end
    output = args.output or default_output_path(args.input)
    run_parameter.update(
        {
            "input_path": args.input,
            "output_path": output,
            "axes": args.axes,
            "channel": args.channel,
            "timepoint": args.timepoint,
            "recursive": args.recursive,
            "exclude_z_start": exclude_start,
            "exclude_z_end": exclude_end,
            "max_candidates": args.max_candidates,
            "detection_snr": args.detection_snr,
            "min_fit_snr": args.min_fit_snr,
            "min_fit_r2": args.min_fit_r2,
            "max_center_offset_px": 1.0,
            "edge_margin_fraction": args.edge_margin_fraction,
            "preset": args.preset,
        }
    )
    summary = fwhm_beads.run_analysis(opt, run_parameter)
    summary["mode"] = "beads"
    summary["preset"] = args.preset
    write_json(summary, output / "fwhm_summary.json")
    _save_recent(args.input, opt)
    return summary, output


def run_lines(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    _validate_common(args)
    _validate_analysis_arguments(args, "lines")
    pixel_xy, pixel_z = _resolve_sampling(args, "strict-single-bead-v1")
    lateral, axial = _resolve_theory(args)
    opt = physical_parameters(pixel_xy, pixel_z, lateral, axial)
    output = args.output or default_output_path(args.input)
    run_parameter = fwhm_lines.default_line_run_parameter()
    run_parameter.update(
        {
            "input_path": args.input,
            "output_path": output,
            "axes": args.axes,
            "channel": args.channel,
            "timepoint": args.timepoint,
            "recursive": args.recursive,
            "exclude_z_start": args.exclude_z_start,
            "exclude_z_end": args.exclude_z_end,
            "edge_margin_fraction": args.edge_margin_fraction,
            "minimum_snr": args.minimum_snr,
            "maximum_curvature_degrees": args.maximum_curvature_degrees,
            "maximum_tilt_degrees": args.maximum_tilt_degrees,
            "maximum_rois_per_line": args.maximum_rois_per_line,
        }
    )
    summary = fwhm_lines.run_line_analysis(opt, run_parameter)
    _save_recent(args.input, opt)
    return summary, output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "review":
        serve_review(args.result, open_browser=not args.no_open, port=args.port)
        return 0
    if args.command == "beads":
        summary, output = run_beads(args)
        status = summary["status"]
        count = summary["final_xy_roi_count"]
    else:
        summary, output = run_lines(args)
        status = summary["status_xy"]
        count = summary["accepted_roi_count_xy"]
    print(f"FWHM analysis complete: {count} lateral ROI; status={status}")
    print(f"Results: {output.resolve()}")
    if args.review:
        serve_review(output, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
