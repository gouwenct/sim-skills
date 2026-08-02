from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_simcheck as _R


class SimcheckWorkflowTests(unittest.TestCase):
    def test_wrapper_preserves_simcheck_v2_output_contract(self) -> None:
        rng = np.random.default_rng(2)
        stack = rng.poisson(40, size=(9, 16, 18)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input_0001.tif"
            output = root / "SIMcheck_V2"
            tifffile.imwrite(source, stack)

            exit_code = _R.main(
                [
                    str(source),
                    "--sim-mode",
                    "2d-sim",
                    "--n-z",
                    "1",
                    "--n-directions",
                    "3",
                    "--n-phases",
                    "3",
                    "--page-order",
                    "AP",
                    "--output-root",
                    str(output),
                    "--device",
                    "cpu",
                ]
            )

            self.assertEqual(exit_code, 0)
            relative_files = {
                str(path.relative_to(output)).replace("\\", "/")
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                relative_files,
                {
                    "SIMcheck_V2_batch_MCNR.csv",
                    "SIMcheck_V2_batch_MCNR.md",
                    "SIMcheck_V2_report.md",
                    "SIMcheck_V2_stats.csv",
                    "raw/FPJ_C1-preview.png",
                    "raw/FPJ_C1.tif",
                    "raw/MCNR_C1_per_angle_compare_z1.png",
                    "raw/MCNR_C1_z1_angles_stack.tif",
                },
            )
            with (output / "SIMcheck_V2_batch_MCNR.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["D1_risk"], "UNCALIBRATED")
            self.assertEqual(row["overall_risk"], "UNCALIBRATED")

    def test_cli_defaults_to_known_protocol_auto_detection(self) -> None:
        args = _R.build_parser().parse_args(
            [
                "input.tif",
                "--n-z",
                "1",
                "--n-directions",
                "3",
                "--n-phases",
                "3",
                "--page-order",
                "AP",
            ]
        )
        self.assertEqual(args.sim_mode, "auto")

    def test_known_sim_modes_define_protocol_specific_checks(self) -> None:
        expected = {
            "2d-sim": (3, 3, 1, (1,), "background_modulation_std", False),
            "3d-sim": (
                3,
                5,
                2,
                (1, 2),
                "highest_frequency_spatial_std",
                True,
            ),
            "3d-mp-sim": (
                3,
                7,
                0,
                (1, 2, 3),
                "background_modulation_std",
                False,
            ),
            "4b-sim": (4, 13, 0, (), "", False),
        }
        for name, values in expected.items():
            with self.subTest(name=name):
                profile = _R.resolve_sim_mode(name, values[0], values[1])
                self.assertEqual(profile.n_directions, values[0])
                self.assertEqual(profile.n_phases, values[1])
                self.assertEqual(profile.fpj_peak_orders, values[2])
                self.assertEqual(profile.mcnr_orders, values[3])
                self.assertEqual(profile.mcnr_noise_method, values[4])
                self.assertEqual(profile.mcnr_calibrated, values[5])

        auto = _R.resolve_sim_mode("auto", 3, 3)
        self.assertEqual(auto.name, "2d-sim")
        with self.assertRaisesRegex(ValueError, "requires 3 directions and 3 phases"):
            _R.resolve_sim_mode("2d-sim", 3, 5)

    def test_2dsim_runs_uncalibrated_mcnr_and_labels_first_order_fpj(self) -> None:
        rng = np.random.default_rng(3)
        stack = rng.poisson(40, size=(9, 16, 18)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.tif"
            output = root / "quality"
            tifffile.imwrite(source, stack)
            exit_code = _R.main(
                [
                    str(source),
                    "--n-z",
                    "1",
                    "--n-directions",
                    "3",
                    "--n-phases",
                    "3",
                    "--page-order",
                    "AP",
                    "--sim-mode",
                    "2d-sim",
                    "--output-root",
                    str(output),
                    "--checks",
                    "mcnr,fpj",
                    "--device",
                    "cpu",
                ]
            )
            self.assertEqual(exit_code, 0)
            with (output / "SIMcheck_V2_stats.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                stats = list(csv.DictReader(handle))
            stat_names = [item["metric"] for item in stats]
            self.assertTrue(any("feature MCNR" in name for name in stat_names))
            self.assertTrue(
                all(np.isfinite(float(item["value"])) for item in stats)
            )
            self.assertTrue(any("1st peak" in name for name in stat_names))
            self.assertFalse(any("2nd peak" in name for name in stat_names))
            self.assertEqual(
                tifffile.imread(output / "raw" / "FPJ_C1.tif").shape[0],
                4,
            )
            self.assertEqual(
                tifffile.imread(
                    output / "raw" / "MCNR_C1_z1_angles_stack.tif"
                ).shape[0],
                3,
            )
            with (output / "SIMcheck_V2_batch_MCNR.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 1)
            self.assertTrue(summary_rows[0]["D1"])
            self.assertTrue(summary_rows[0]["D2"])
            self.assertTrue(summary_rows[0]["D3"])
            self.assertEqual(summary_rows[0]["D1_risk"], "UNCALIBRATED")
            self.assertEqual(summary_rows[0]["overall_risk"], "UNCALIBRATED")

    def test_three_five_seven_phase_mcnr_is_finite_and_modulation_sensitive(self) -> None:
        rng = np.random.default_rng(31)
        yy, xx = np.mgrid[:24, :24]
        feature = (yy - 12) ** 2 + (xx - 12) ** 2 <= 25

        for phase_count, orders in ((3, (1,)), (5, (1, 2)), (7, (1, 2, 3))):
            with self.subTest(phase_count=phase_count):
                phases = 2.0 * np.pi * np.arange(phase_count) / phase_count
                base = rng.poisson(40, size=(phase_count, 1, 24, 24)).astype(
                    np.float32
                )
                weak = base.copy()
                strong = base.copy()
                for phase_index, phase in enumerate(phases):
                    weak[phase_index, 0, feature] += 3.0 * np.cos(phase)
                    strong[phase_index, 0, feature] += 18.0 * np.cos(phase)

                weak_map, weak_noise = _R._SC._mcnr_volume_for_angle(
                    weak,
                    harmonic_orders=orders,
                    zw=0,
                )
                strong_map, strong_noise = _R._SC._mcnr_volume_for_angle(
                    strong,
                    harmonic_orders=orders,
                    zw=0,
                )

                self.assertTrue(np.isfinite(weak_map).all())
                self.assertTrue(np.isfinite(strong_map).all())
                self.assertTrue(np.isfinite(weak_noise).all())
                self.assertTrue(np.isfinite(strong_noise).all())
                self.assertGreater(
                    float(strong_map[0, feature].mean()),
                    float(weak_map[0, feature].mean()),
                )

    def test_five_phase_report_keeps_legacy_stats_schema(self) -> None:
        rng = np.random.default_rng(37)
        raw = rng.poisson(40, size=(1, 5, 1, 3, 1, 12, 14)).astype(
            np.float32
        )
        with tempfile.TemporaryDirectory() as temporary:
            report = _R._SC.check_mcnr(
                raw,
                Path(temporary),
                harmonic_orders=(1, 2),
                noise_method="highest_frequency_spatial_std",
                calibrated=True,
                zw=0,
            )
        self.assertFalse(
            any("background noise std" in stat.name for stat in report.stats)
        )

    def test_mcnr_background_excludes_constant_zero_padding(self) -> None:
        rng = np.random.default_rng(41)
        inner = rng.poisson(40, size=(3, 1, 20, 20)).astype(np.float32)
        phase = 2.0 * np.pi * np.arange(3) / 3
        for phase_index, value in enumerate(phase):
            inner[phase_index, 0, 6:14, 6:14] += 12.0 * np.cos(value)
        padded = np.zeros((3, 1, 40, 40), dtype=np.float32)
        padded[:, :, 10:30, 10:30] = inner

        inner_map, inner_noise = _R._SC._mcnr_volume_for_angle(
            inner,
            harmonic_orders=(1,),
            zw=0,
        )
        padded_map, padded_noise = _R._SC._mcnr_volume_for_angle(
            padded,
            harmonic_orders=(1,),
            zw=0,
        )

        np.testing.assert_allclose(padded_noise, inner_noise, rtol=1e-6)
        np.testing.assert_allclose(
            padded_map[:, 10:30, 10:30],
            inner_map,
            rtol=1e-6,
        )

    def test_4bsim_fpj_is_visual_only_and_mcnr_is_not_mislabeled(self) -> None:
        rng = np.random.default_rng(13)
        stack = rng.poisson(40, size=(4 * 13, 16, 18)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.tif"
            output = root / "quality"
            tifffile.imwrite(source, stack)
            exit_code = _R.main(
                [
                    str(source),
                    "--n-z",
                    "1",
                    "--n-directions",
                    "4",
                    "--n-phases",
                    "13",
                    "--page-order",
                    "AP",
                    "--sim-mode",
                    "4b-sim",
                    "--output-root",
                    str(output),
                    "--checks",
                    "mcnr,fpj",
                    "--device",
                    "cpu",
                    "--no-preview",
                ]
            )
            self.assertEqual(exit_code, 0)
            with (output / "SIMcheck_V2_stats.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                stats = list(csv.DictReader(handle))
            stat_names = [item["metric"] for item in stats]
            self.assertFalse(any("peak r(px)" in name for name in stat_names))
            self.assertFalse(
                list((output / "raw").glob("MCNR_C1_*.tif"))
            )
            fpj = tifffile.imread(output / "raw" / "FPJ_C1.tif")
            self.assertEqual(fpj.shape[0], 5)

    def test_3dmpsim_uses_legacy_output_names(self) -> None:
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.tif"
            output = root / "SIMcheck_V2"
            tifffile.imwrite(
                source,
                rng.poisson(40, size=(3 * 7, 12, 14)).astype(np.float32),
            )
            exit_code = _R.main(
                [
                    str(source),
                    "--sim-mode",
                    "3d-mp-sim",
                    "--n-z",
                    "1",
                    "--n-directions",
                    "3",
                    "--n-phases",
                    "7",
                    "--page-order",
                    "AP",
                    "--output-root",
                    str(output),
                    "--device",
                    "cpu",
                    "--no-preview",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                {
                    str(path.relative_to(output)).replace("\\", "/")
                    for path in output.rglob("*")
                    if path.is_file()
                },
                {
                    "SIMcheck_V2_batch_MCNR.csv",
                    "SIMcheck_V2_batch_MCNR.md",
                    "SIMcheck_V2_report.md",
                    "SIMcheck_V2_stats.csv",
                    "raw/FPJ_C1.tif",
                    "raw/MCNR_C1_per_angle_compare_z1.png",
                    "raw/MCNR_C1_z1_angles_stack.tif",
                },
            )

    def test_reshape_zap_with_seven_phases(self) -> None:
        spec = _R.AcquisitionSpec(
            n_channels=1,
            n_timepoints=1,
            n_z=2,
            n_directions=4,
            n_phases=7,
            page_order="ZAP",
        )
        stack = np.arange(56 * 2 * 3, dtype=np.float32).reshape(56, 2, 3)
        canonical = _R.reshape_pages_to_ctzapyx(stack, spec)
        self.assertEqual(canonical.shape, (1, 1, 2, 4, 7, 2, 3))
        for z_index in range(2):
            for direction_index in range(4):
                for phase_index in range(7):
                    page = (z_index * 4 + direction_index) * 7 + phase_index
                    np.testing.assert_array_equal(
                        canonical[
                            0,
                            0,
                            z_index,
                            direction_index,
                            phase_index,
                        ],
                        stack[page],
                    )

    def test_arbitrary_full_page_order(self) -> None:
        shape = (2, 2, 2, 3, 5, 2, 2)
        canonical = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        stack = np.transpose(canonical, (1, 3, 2, 4, 0, 5, 6)).reshape(
            -1,
            2,
            2,
        )
        spec = _R.AcquisitionSpec(
            n_channels=2,
            n_timepoints=2,
            n_z=2,
            n_directions=3,
            n_phases=5,
            page_order="TAZPC",
        )
        actual = _R.reshape_pages_to_ctzapyx(stack, spec)
        np.testing.assert_array_equal(actual, canonical)

    def test_rejects_ambiguous_or_missing_axis(self) -> None:
        spec = _R.AcquisitionSpec(
            n_channels=1,
            n_timepoints=1,
            n_z=2,
            n_directions=3,
            n_phases=5,
            page_order="ZP",
        )
        with self.assertRaisesRegex(ValueError, "omits non-singleton"):
            _R.reshape_pages_to_ctzapyx(
                np.zeros((30, 2, 2), dtype=np.float32),
                spec,
            )

    def test_recursive_inputs_preserve_relative_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = root / "a" / "same.tif"
            source_b = root / "b" / "same.tif"
            source_a.parent.mkdir()
            source_b.parent.mkdir()
            tifffile.imwrite(source_a, np.zeros((5, 2, 2), dtype=np.float32))
            tifffile.imwrite(source_b, np.zeros((5, 2, 2), dtype=np.float32))
            output = root / "results"
            sources = _R.discover_inputs(
                root,
                name_glob="*.tif",
                recursive=True,
                excluded_root=output,
            )
            self.assertEqual(len(sources), 2)
            self.assertEqual(
                _R._item_output_dir(source_a, root, output),
                output / "a" / "same",
            )
            self.assertEqual(
                _R._item_output_dir(source_b, root, output),
                output / "b" / "same",
            )
            _R._preflight_output_dirs(
                sources,
                root,
                output,
                overwrite=False,
            )

    def test_folder_input_copies_batch_summary_into_each_run(self) -> None:
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "inputs"
            inputs.mkdir()
            for name in ("sample_a.tif", "sample_b.tif"):
                tifffile.imwrite(
                    inputs / name,
                    rng.poisson(40, size=(9, 12, 14)).astype(np.float32),
                )
            output = root / "SIMcheck_V2"
            exit_code = _R.main(
                [
                    str(inputs),
                    "--sim-mode",
                    "2d-sim",
                    "--n-z",
                    "1",
                    "--n-directions",
                    "3",
                    "--n-phases",
                    "3",
                    "--page-order",
                    "AP",
                    "--output-root",
                    str(output),
                    "--device",
                    "cpu",
                    "--no-preview",
                ]
            )
            self.assertEqual(exit_code, 0)
            for filename in (
                "SIMcheck_V2_batch_MCNR.csv",
                "SIMcheck_V2_batch_MCNR.md",
            ):
                expected = (output / filename).read_bytes()
                for run_name in ("sample_a", "sample_b"):
                    self.assertEqual(
                        (output / run_name / filename).read_bytes(),
                        expected,
                    )

    def test_mcnr_fpj_support_four_directions_and_thirteen_phases(self) -> None:
        rng = np.random.default_rng(7)
        z_count, direction_count, phase_count = 2, 4, 13
        stack = rng.poisson(
            40,
            size=(z_count * direction_count * phase_count, 16, 18),
        ).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.tif"
            output = root / "quality"
            tifffile.imwrite(source, stack)
            exit_code = _R.main(
                [
                    str(source),
                    "--n-z",
                    str(z_count),
                    "--n-directions",
                    str(direction_count),
                    "--n-phases",
                    str(phase_count),
                    "--page-order",
                    "ZAP",
                    "--sim-mode",
                    "generic",
                    "--output-root",
                    str(output),
                    "--checks",
                    "mcnr,fpj",
                    "--device",
                    "cpu",
                    "--no-preview",
                ]
            )
            self.assertEqual(exit_code, 0)
            fpj = tifffile.imread(output / "raw" / "FPJ_C1.tif")
            self.assertEqual(fpj.shape[0], direction_count + 1)
            self.assertTrue(np.isfinite(fpj).all())
            self.assertTrue(
                (output / "SIMcheck_V2_stats.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
