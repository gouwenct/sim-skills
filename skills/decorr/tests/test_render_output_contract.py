import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import render_decorr_results as renderer  # noqa: E402


def test_axial_spectrum_marks_kc_on_the_z_frequency_axis():
    figure, axis = renderer.plt.subplots()
    renderer._render_axial_spectrum(
        axis,
        np.arange(64, dtype=np.float32).reshape(8, 8),
        0.7,
        "YZ",
    )

    assert len(axis.images) == 1
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [0.7, 0.7])
    np.testing.assert_allclose(axis.lines[1].get_ydata(), [-0.7, -0.7])
    renderer.plt.close(figure)


def test_batch_limits_ignore_nan_and_cover_all_samples():
    assert renderer._finite_limits([]) == (0.0, 1.0)
    assert renderer._finite_limits(
        [np.array([110.0, np.nan]), np.array([90.0, 130.0])]
    ) == (90.0, 130.0)


def test_render_batch_writes_index_pages_and_four_sample_figures(tmp_path, monkeypatch):
    sample = tmp_path / "samples" / "sample-01"
    lateral = sample / "lateral"
    axial = sample / "axial"
    lateral.mkdir(parents=True)
    axial.mkdir()
    image = np.indices((40, 40)).sum(axis=0).astype(np.float32)
    tifffile.imwrite(lateral / "selected_plane.tif", image)
    pd.DataFrame(
        [
            {
                "data_axes": "ZYX",
                "selected_plane": 1,
                "pixel_size_nm": 30.0,
                "kc": 0.5,
                "resolution_nm": 120.0,
            }
        ]
    ).to_csv(lateral / "lateral_result.csv", index=False)
    local_rows = []
    for row in range(4):
        for column in range(4):
            local_rows.append(
                {
                    "grid_row": row + 1,
                    "grid_column": column + 1,
                    "y_first": row * 9 + 1,
                    "y_last": row * 9 + 9,
                    "x_first": column * 9 + 1,
                    "x_last": column * 9 + 9,
                    "resolution_nm": 110.0 + row + column,
                }
            )
    pd.DataFrame(local_rows).to_csv(
        lateral / "lateral_local_results.csv", index=False
    )
    groups = np.arange(1, 13)
    pd.DataFrame(
        {
            "group_index": groups,
            "axial_pixel_size_nm": 96.9,
            "xz_kc": 0.70,
            "xz_resolution_nm": 276.86,
            "yz_kc": 0.72,
            "yz_resolution_nm": 269.17,
        }
    ).to_csv(axial / "axial_result.csv", index=False)
    montages = np.random.RandomState(1).normal(size=(12, 64, 64)).astype(np.float32)
    tifffile.imwrite(axial / "xz_montage.tif", montages)
    tifffile.imwrite(axial / "yz_montage.tif", montages)
    np.savez_compressed(
        axial / "axial_diagnostics.npz",
        xz_d0=np.zeros((12, 50, 20)),
        yz_d0=np.zeros((12, 50, 20)),
        xz_curves=np.zeros((12, 50, 20, 20)),
        yz_curves=np.zeros((12, 50, 20, 20)),
    )
    pd.DataFrame({"status": ["ok"]}).to_excel(
        tmp_path / "decorr_results.xlsx",
        sheet_name="样本汇总",
        index=False,
    )
    legacy_peak_qc = (
        tmp_path
        / "axial-peak-qc"
        / "samples"
        / "sample-01"
        / "axial_peak_qc.png"
    )
    legacy_peak_qc.parent.mkdir(parents=True)
    legacy_peak_qc.touch()
    monkeypatch.setattr(
        renderer,
        "imdecorr_global",
        lambda *_args, **_kwargs: {
            "kc": 0.5,
            "d0": np.zeros(50),
            "curves": np.zeros((50, 20)),
        },
    )

    renderer.render_batch(tmp_path)

    assert (tmp_path / "INDEX.md").is_file()
    assert (tmp_path / "batch_decorrelation_summary.png").is_file()
    assert (tmp_path / "lateral_qc_page-01.png").is_file()
    assert (tmp_path / "axial_qc_page-01.png").is_file()
    assert (tmp_path / "lateral_qc_pages" / "sample-01.png").is_file()
    assert (tmp_path / "axial_qc_pages" / "sample-01.png").is_file()
    assert (
        tmp_path / "lateral_decorrelation_paper" / "sample-01.png"
    ).is_file()
    assert (
        tmp_path / "axial_decorrelation_paper" / "sample-01.png"
    ).is_file()
    assert not (sample / "figures").exists()
    peak_qc = tmp_path / "axial-peak-qc"
    assert (peak_qc / "axial_peak_qc.csv").is_file()
    assert (peak_qc / "axial_peak_qc_summary.png").is_file()
    assert (peak_qc / "pages" / "axial_peak_qc_page-01.png").is_file()
    assert (peak_qc / "samples" / "sample-01.png").is_file()
    assert not legacy_peak_qc.parent.exists()
    assert "spectral_support_6db" in pd.read_csv(
        peak_qc / "axial_peak_qc.csv"
    ).columns
    assert renderer.PEAK_QC_SHEET in pd.ExcelFile(
        tmp_path / "decorr_results.xlsx"
    ).sheet_names
    figure_summary = pd.read_csv(tmp_path / "decorrelation_figure_summary.csv")
    assert "axial_6db_median_resolution_nm" in figure_summary.columns
    assert "axial_6db_valid_groups" in figure_summary.columns
    index_text = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert "| N/A |" in index_text
    assert "| nan |" not in index_text
    qc_info = pd.read_json(
        peak_qc / "qc_run_info.json", typ="series"
    )
    peak_qc_table = pd.read_csv(peak_qc / "axial_peak_qc.csv")
    assert qc_info["resolution_color_min_nm"] == peak_qc_table[
        "baseline_resolution_nm"
    ].min()
    assert qc_info["resolution_color_max_nm"] == peak_qc_table[
        "baseline_resolution_nm"
    ].max()
    assert qc_info["spectral_color_min_db"] == 0
    assert qc_info["spectral_color_max_db"] == 15
    assert qc_info["curve_support_color_min"] == 0
    assert qc_info["curve_support_color_max"] == 10

    renderer.render_batch(sample)
    assert (sample / "INDEX.md").is_file()
    assert (sample / "batch_decorrelation_summary.png").is_file()
