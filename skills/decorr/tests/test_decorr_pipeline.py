import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import decorr_pipeline as pipeline  # noqa: E402
from decorr_pipeline import (  # noqa: E402
    DecorrParameters,
    MontageParameters,
    analyze_file,
    build_axial_montages,
    imdecorr_sectorial,
    local_grid,
    read_tiff_stack_float32,
    resolution_from_kc,
    select_xy_plane,
    write_tiff_float32,
)


def test_float_tiff_round_trip_preserves_negative_values(tmp_path):
    expected = np.array(
        [[[-2.5, 0.25], [1.5, -0.75]], [[3.0, -4.0], [0.0, 2.0]]],
        dtype=np.float32,
    )
    path = tmp_path / "negative.tif"
    write_tiff_float32(path, expected)
    actual = read_tiff_stack_float32(path)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    assert tifffile.TiffFile(path).pages[0].sampleformat == 3


def test_tenengrad_and_manual_selection():
    stack = np.zeros((5, 32, 32), dtype=np.float32)
    stack[2, :, :16] = -5
    stack[2, :, 16:] = 5
    stack[3] = 100
    _, automatic, _ = select_xy_plane(stack)
    assert automatic["selected_index"] == 3
    assert automatic["mode"] == "tenengrad"
    manual_plane, manual, _ = select_xy_plane(stack, plane_index=2)
    assert manual["selected_index"] == 2
    assert manual["mode"] == "manual"
    np.testing.assert_array_equal(manual_plane, stack[1])


def test_deterministic_montage_has_one_page_per_group():
    z, y, x = np.indices((4, 16, 16), dtype=np.float32)
    stack = x + 2 * y + 3 * z
    params = MontageParameters(
        num_groups=3,
        num_points=4,
        end_replication=2,
        apodization=2,
        canvas_size=32,
        seed=17,
    )
    xz1, yz1, info1 = build_axial_montages(stack, params)
    xz2, yz2, info2 = build_axial_montages(stack, params)
    assert xz1.shape == (3, 32, 32)
    assert yz1.shape == (3, 32, 32)
    np.testing.assert_array_equal(xz1, xz2)
    np.testing.assert_array_equal(yz1, yz2)
    assert info1 == info2


def test_resolution_conversion():
    assert resolution_from_kc(30.0, 0.25) == 240.0
    assert np.isnan(resolution_from_kc(30.0, 0.0))


def test_global_decorrelation_preserves_refined_frequency_axis():
    image = np.random.RandomState(3).normal(size=(65, 65)).astype(np.float32)
    result = pipeline.imdecorr_global(image)

    assert result["refined_radii"].shape == (DecorrParameters().nr,)
    assert np.ptp(result["refined_radii"]) <= 0.4 + np.finfo(float).eps


def test_selected_sector_matches_full_and_skips_unused_refinement(monkeypatch):
    image = np.random.RandomState(7).normal(size=(65, 65)).astype(np.float32)
    params = DecorrParameters(nr=12, ng=3, na=4)
    original_curve = pipeline._RadialSampler.curve
    call_count = 0

    def counted_curve(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_curve(*args, **kwargs)

    monkeypatch.setattr(pipeline._RadialSampler, "curve", counted_curve)
    full = imdecorr_sectorial(image, params)
    full_calls = call_count
    call_count = 0
    sector = params.na // 2
    selected = imdecorr_sectorial(image, params, selected_sector=sector)

    assert full_calls == params.na + 2 * params.na * params.ng
    assert call_count == params.na + 2 * params.ng
    np.testing.assert_equal(selected["kcGM"][sector], full["kcGM"][sector])
    np.testing.assert_equal(
        selected["peak_snr"][sector], full["peak_snr"][sector]
    )
    assert selected["interior_valid"][sector] == full["interior_valid"][sector]
    np.testing.assert_array_equal(selected["d0"], full["d0"])


def test_cuda_selected_sector_matches_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image = np.random.RandomState(8).normal(size=(65, 65)).astype(np.float32)
    params = DecorrParameters(nr=12, ng=3, na=4)
    sector = params.na // 2

    cpu = imdecorr_sectorial(image, params, selected_sector=sector)
    cuda = imdecorr_sectorial(
        image,
        params,
        selected_sector=sector,
        device="cuda",
    )

    assert cuda["device"] == "cuda"
    np.testing.assert_equal(cuda["kcGM"][sector], cpu["kcGM"][sector])
    np.testing.assert_equal(cuda["peak_snr"][sector], cpu["peak_snr"][sector])
    assert cuda["interior_valid"][sector] == cpu["interior_valid"][sector]


def test_cuda_full_sectorial_matches_cpu_reported_outputs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    image = np.random.RandomState(9).normal(size=(65, 65)).astype(np.float32)
    params = DecorrParameters(nr=12, ng=3, na=4)

    cpu = imdecorr_sectorial(image, params)
    cuda = imdecorr_sectorial(image, params, device="cuda")

    np.testing.assert_equal(cuda["kcGM"], cpu["kcGM"])
    np.testing.assert_equal(cuda["interior_valid"], cpu["interior_valid"])
    np.testing.assert_allclose(cuda["d0"], cpu["d0"], rtol=0, atol=0.001)
    np.testing.assert_allclose(
        cuda["curves"], cpu["curves"], rtol=0, atol=0.001
    )


def test_torch_radial_sampler_batches_match_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    r2, _ = pipeline._frequency_geometry((17, 17))
    rng = np.random.RandomState(19)
    first = (
        rng.normal(size=(3, 17, 17)) + 1j * rng.normal(size=(3, 17, 17))
    ).astype(np.complex64)
    second = (
        rng.normal(size=(3, 17, 17)) + 1j * rng.normal(size=(3, 17, 17))
    ).astype(np.complex64)
    radii = np.stack(
        [
            np.linspace(0.0, 1.0, 12),
            np.linspace(0.1, 0.9, 12),
            np.linspace(0.2, 0.8, 12),
        ]
    )
    energies = np.sqrt(
        np.sum(np.abs(first) ** 2, axis=(1, 2), dtype=np.float64)
    )
    cpu_sampler = pipeline._RadialSampler(r2)
    expected = np.stack(
        [
            cpu_sampler.curve(first[i], second[i], energies[i], radii[i])
            for i in range(len(first))
        ]
    )
    cuda_sampler = pipeline._TorchRadialSampler(r2, torch)

    actual = cuda_sampler.curves(
        torch.as_tensor(first, device="cuda"),
        torch.as_tensor(second, device="cuda"),
        torch.as_tensor(energies, device="cuda"),
        radii,
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0.001)


def test_torch_sectorial_geometry_is_cached():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    first = pipeline._torch_sectorial_geometry((33, 33), 4, torch)
    second = pipeline._torch_sectorial_geometry((33, 33), 4, torch)

    assert first is second


def test_auto_device_falls_back_to_cpu(monkeypatch):
    image = np.ones((33, 33), dtype=np.float32)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("test CUDA failure")

    monkeypatch.setattr(pipeline, "_imdecorr_sectorial_cuda", unavailable)
    with pytest.warns(UserWarning, match="using CPU"):
        result = imdecorr_sectorial(
            image,
            DecorrParameters(nr=8, ng=2, na=2),
            selected_sector=1,
            device="auto",
        )

    assert result["device"] == "cpu"


def test_local_grid_excludes_low_contrast_padding(monkeypatch):
    image = np.zeros((40, 40), dtype=np.float32)
    image[9:27, ::2] = 10.0
    monkeypatch.setattr(
        pipeline,
        "imdecorr_global",
        lambda *_args, **_kwargs: {"kc": 0.5, "peak_snr": 1.0},
    )

    result = local_grid(image, 30.0, tile_size=8)

    assert result["resolution_nm"].notna().sum() == 8
    assert result.loc[result["grid_row"].isin([1, 4]), "resolution_nm"].isna().all()


def test_yx_analysis_skips_axial_outputs(tmp_path, monkeypatch):
    input_path = tmp_path / "image.tif"
    tifffile.imwrite(input_path, np.ones((32, 32), dtype=np.float32))
    monkeypatch.setattr(
        pipeline,
        "imdecorr_global",
        lambda *_args, **_kwargs: {
            "kc": 0.5,
            "peak_snr": 1.0,
            "d0": np.zeros(4),
            "curves": np.zeros((4, 4)),
        },
    )
    monkeypatch.setattr(
        pipeline,
        "local_grid",
        lambda *_args, **_kwargs: pd.DataFrame(
            {
                "grid_row": [1],
                "grid_column": [1],
                "y_first": [1],
                "y_last": [32],
                "x_first": [1],
                "x_last": [32],
                "kc": [0.5],
                "resolution_nm": [120.0],
                "peak_snr": [1.0],
            }
        ),
    )

    output = tmp_path / "output"
    summary = analyze_file(input_path, output, xy_pixel_size_nm=30.0)

    assert summary["data_axes"] == "YX"
    assert summary["axial_status"] == "not_applicable"
    assert not (output / "axial").exists()
    assert not (output / "lateral" / "plane_scores.csv").exists()


def test_sampling_must_be_positive(tmp_path):
    input_path = tmp_path / "image.tif"
    tifffile.imwrite(input_path, np.ones((32, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        analyze_file(input_path, tmp_path / "output", xy_pixel_size_nm=0)
