"""Run the formal 27-sample decorrelation batch with resumable Python outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from decorr_pipeline import OUTPUT_SCHEMA_VERSION, analyze_file


PATTERN = "*/n-3.0-01-488_padding/留负MP-新建文件夹 (10)-mear1去相关*.tif"
WORKBOOK_SHEETS = (
    "样本汇总",
    "横向局部块",
    "轴向分组",
    "轴向种子稳定性",
    "运行信息",
)
SHEET_DESCRIPTIONS = {
    "样本汇总": "每个样本的选层、横向/轴向中位数和 QC 状态",
    "横向局部块": "4x4 网格结果；无信号块的分辨率留空",
    "轴向分组": "主种子下 12 组 XZ/YZ 结果",
    "轴向种子稳定性": "审计种子的样本级结果及相对主种子偏差",
    "运行信息": "输入、采样、算法版本、阈值、种子和时间",
}


def default_output_root(input_root: Path) -> Path:
    return input_root.parent / "decorr-results" / input_root.name


def sample_outputs_complete(
    sample_root: Path,
    summary: dict[str, object],
    *,
    axial_device: str | None = None,
) -> bool:
    required = (
        "lateral/lateral_result.csv",
        "lateral/lateral_local_results.csv",
        "lateral/selected_plane.tif",
        "lateral/lateral_diagnostics.npz",
        "axial/axial_result.csv",
        "axial/xz_montage.tif",
        "axial/yz_montage.tif",
        "axial/axial_diagnostics.npz",
        "axial/axial_seed_stability.csv",
    )
    actual_device = summary.get("axial_device")
    device_matches = (
        axial_device is None
        or actual_device == axial_device
        or axial_device == "auto" and actual_device in {"cpu", "cuda"}
    )
    return (
        summary.get("status") == "ok"
        and summary.get("output_schema_version") == OUTPUT_SCHEMA_VERSION
        and device_matches
        and all((sample_root / relative).is_file() for relative in required)
    )


def _concat(tables: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)


def write_workbook(
    output_root: Path,
    *,
    summaries: list[dict[str, object]],
    local_tables: list[pd.DataFrame],
    axial_tables: list[pd.DataFrame],
    stability_tables: list[pd.DataFrame],
    run_info: dict[str, object],
) -> Path:
    workbook = output_root / "decorr_results.xlsx"
    summary_table = pd.DataFrame(summaries)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary_table.to_excel(writer, sheet_name=WORKBOOK_SHEETS[0], index=False)
        _concat(
            local_tables,
            ["sample_name", "grid_row", "grid_column", "resolution_nm"],
        ).to_excel(writer, sheet_name=WORKBOOK_SHEETS[1], index=False)
        _concat(axial_tables, ["sample_name"]).to_excel(
            writer, sheet_name=WORKBOOK_SHEETS[2], index=False
        )
        _concat(
            stability_tables,
            [
                "sample_name",
                "seed",
                "xz_median_resolution_nm",
                "yz_median_resolution_nm",
                "max_relative_deviation",
                "status",
            ],
        ).to_excel(writer, sheet_name=WORKBOOK_SHEETS[3], index=False)
        pd.DataFrame(
            {"parameter": list(run_info), "value": list(run_info.values())}
        ).to_excel(writer, sheet_name=WORKBOOK_SHEETS[4], index=False)

    book = load_workbook(workbook)
    sheet = book[WORKBOOK_SHEETS[0]]
    first = sheet.max_row + 3
    sheet.cell(first, 1, "工作表说明")
    for offset, name in enumerate(WORKBOOK_SHEETS, start=1):
        sheet.cell(first + offset, 1, name)
        sheet.cell(first + offset, 2, SHEET_DESCRIPTIONS[name])
    book.save(workbook)
    return workbook


def run_batch(
    input_root: Path,
    output_root: Path | None = None,
    *,
    axial_device: str = "cpu",
    audit_seeds: tuple[int, ...] = (),
) -> Path:
    output_root = output_root or default_output_root(input_root)
    if output_root.resolve() == input_root.resolve() or input_root.resolve() in output_root.resolve().parents:
        raise ValueError("Output must not overlap the input tree")
    files = sorted(input_root.glob(PATTERN), key=lambda path: str(path.parent))
    if len(files) != 27:
        raise RuntimeError(f"Expected 27 formal TIFFs, found {len(files)}")
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    local_tables = []
    axial_tables = []
    stability_tables = []
    for index, input_file in enumerate(files, start=1):
        sample = input_file.parents[1].name
        sample_root = output_root / "samples" / sample
        summary_path = sample_root / "summary.json"
        print(f"[{index}/27] {sample}", flush=True)
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if sample_outputs_complete(
                sample_root,
                summary,
                axial_device=axial_device,
            ):
                print("  reused", flush=True)
            else:
                summary = {}
        else:
            summary = {}
        if not summary:
            started = time.perf_counter()
            try:
                summary = analyze_file(
                    input_file,
                    sample_root,
                    xy_pixel_size_nm=30.0,
                    z_pixel_size_nm=96.9,
                    audit_seeds=audit_seeds,
                    axial_device=axial_device,
                )
            except Exception as exc:
                summary = {
                    "sample_name": sample,
                    "input_file": str(input_file),
                    "status": "error",
                    "message": str(exc),
                }
            else:
                summary["sample_name"] = sample
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  seconds={time.perf_counter() - started:.3f}", flush=True)
        summaries.append(summary)
        if summary.get("status") == "ok":
            local = pd.read_csv(
                sample_root / "lateral" / "lateral_local_results.csv"
            )
            local.insert(0, "sample_name", sample)
            local_tables.append(local)
            axial_path = sample_root / "axial" / "axial_result.csv"
            if axial_path.is_file():
                axial = pd.read_csv(axial_path)
                axial.insert(0, "sample_name", sample)
                axial_tables.append(axial)
            stability_path = sample_root / "axial" / "axial_seed_stability.csv"
            if stability_path.is_file():
                stability = pd.read_csv(stability_path)
                stability.insert(0, "sample_name", sample)
                stability_tables.append(stability)
        pd.DataFrame(summaries).to_csv(
            output_root / "formal_progress.csv", index=False, encoding="utf-8-sig"
        )

    summary_table = pd.DataFrame(summaries)
    workbook = write_workbook(
        output_root,
        summaries=summaries,
        local_tables=local_tables,
        axial_tables=axial_tables,
        stability_tables=stability_tables,
        run_info={
            "input_root": str(input_root),
            "input_pattern": PATTERN,
            "sample_count": len(files),
            "xy_pixel_size_nm": 30.0,
            "z_pixel_size_nm": 96.9,
            "selection_mode": "tenengrad",
            "signal_contrast_threshold": 0.05,
            "axial_seed": 20250124,
            "audit_seeds": ",".join(map(str, audit_seeds)),
            "axial_device_requested": axial_device,
            "python_version": sys.version,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
        },
    )
    if not (summary_table["status"] == "ok").all():
        raise RuntimeError(f"Workbook written with failures: {workbook}")
    return workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-seeds", type=int, nargs="*", default=())
    parser.add_argument(
        "--axial-device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )
    args = parser.parse_args()
    print(
        run_batch(
            args.input_root,
            args.output_root,
            axial_device=args.axial_device,
            audit_seeds=tuple(args.audit_seeds),
        )
    )


if __name__ == "__main__":
    main()
