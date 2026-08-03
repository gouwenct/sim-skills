from datetime import datetime
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from decorr_pipeline import default_output_path, prepare_output_path  # noqa: E402
from run_formal_decorr_batch import (  # noqa: E402
    OUTPUT_SCHEMA_VERSION,
    WORKBOOK_SHEETS,
    default_output_root,
    sample_outputs_complete,
    write_workbook,
)


def test_measure_default_output_is_next_to_input_data_folder(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    input_file = input_dir / "image.tif"
    input_file.touch()
    now = datetime(2026, 7, 30, 21, 5, 9)

    assert default_output_path(input_file, now) == (
        tmp_path / "decorr-results" / "image-20260730-210509"
    )


def test_measure_default_output_adds_suffix_on_collision(tmp_path):
    input_dir = tmp_path / "data"
    input_dir.mkdir()
    input_file = input_dir / "image.tif"
    input_file.touch()
    now = datetime(2026, 7, 30, 21, 5, 9)
    existing = tmp_path / "decorr-results" / "image-20260730-210509"
    existing.mkdir(parents=True)

    assert default_output_path(input_file, now) == (
        tmp_path / "decorr-results" / "image-20260730-210509-01"
    )


def test_measure_rejects_existing_explicit_output(tmp_path):
    input_file = tmp_path / "image.tif"
    input_file.touch()
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        prepare_output_path(input_file, output)


def test_default_output_root_is_next_to_input_folder(tmp_path):
    input_root = tmp_path / "batch"
    assert default_output_root(input_root) == tmp_path / "decorr-results" / "batch"


def test_workbook_uses_confirmed_five_sheet_contract(tmp_path):
    workbook = write_workbook(
        tmp_path,
        summaries=[{"sample_name": "sample-01", "status": "ok"}],
        local_tables=[pd.DataFrame({"sample_name": ["sample-01"]})],
        axial_tables=[pd.DataFrame({"sample_name": ["sample-01"]})],
        stability_tables=[],
        run_info={"input_root": "input"},
    )

    assert workbook.name == "decorr_results.xlsx"
    assert pd.ExcelFile(workbook).sheet_names == list(WORKBOOK_SHEETS)


def test_resume_requires_all_schema_artifacts(tmp_path):
    summary = {"status": "ok", "output_schema_version": OUTPUT_SCHEMA_VERSION}
    assert not sample_outputs_complete(tmp_path, summary)


def test_resume_requires_compatible_axial_device(tmp_path):
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
    for relative in required:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    summary = {
        "status": "ok",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "axial_device": "cuda",
    }

    assert sample_outputs_complete(tmp_path, summary, axial_device="cuda")
    assert sample_outputs_complete(tmp_path, summary, axial_device="auto")
    assert not sample_outputs_complete(tmp_path, summary, axial_device="cpu")
