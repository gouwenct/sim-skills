import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_axial_peak_qc import (  # noqa: E402
    SPECTRAL_SNR_MIN_DB,
    axial_spectral_diagnostics,
    curve_peak_diagnostics,
    qc_status,
)


def _gaussian_curve(radii: np.ndarray, center: float) -> np.ndarray:
    return np.exp(-0.5 * ((radii - center) / 0.025) ** 2)


def test_curve_support_counts_refined_peaks_near_selected_kc():
    nr, na, ng = 50, 2, 10
    params = SimpleNamespace(Na=na, Ng=ng, dt=0.01, rmin=0.2, rmax=0.3)
    curves = np.zeros((nr, na, 2 * ng), dtype=np.float64)
    initial_radii = np.linspace(0.0, 1.0, nr)
    for index in range(ng):
        curves[:, 1, index] = _gaussian_curve(initial_radii, 0.7)
    refined_radii = np.linspace(0.5, 1.0, nr)
    selected_kc = float(refined_radii[np.argmin(np.abs(refined_radii - 0.82))])
    for index in range(ng):
        center = selected_kc if index < 7 else 0.7
        curves[:, 1, ng + index] = _gaussian_curve(refined_radii, center)
    core = SimpleNamespace(curves=curves, kcGM=np.array([0.5, selected_kc]))

    result = curve_peak_diagnostics(core, params)

    assert result["curve_support_count"] == 7
    assert np.isclose(result["selected_kc"], selected_kc)


def test_spectral_margin_is_higher_at_a_supported_frequency():
    rng = np.random.RandomState(12)
    rows, columns = 512, 128
    y = np.arange(rows, dtype=np.float64)
    image = rng.normal(0.0, 1.0, (rows, columns))
    for frequency in np.linspace(0.66, 0.74, 21):
        image += 0.8 * np.sin(np.pi * frequency * y)[:, None]

    supported = axial_spectral_diagnostics(image, 0.7)
    unsupported = axial_spectral_diagnostics(image, 0.85)

    assert supported["spectral_snr_db"] > SPECTRAL_SNR_MIN_DB
    assert supported["spectral_snr_db"] > unsupported["spectral_snr_db"]


def test_qc_status_requires_both_diagnostics_to_pass():
    assert qc_status(6, 6.0) == "PASS"
    assert qc_status(5, 8.0) == "REVIEW"
    assert qc_status(8, 5.9) == "REVIEW"
