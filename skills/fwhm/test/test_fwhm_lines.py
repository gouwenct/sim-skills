from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fwhm_common import compact_stem, physical_parameters
from fwhm_lines import (
    _local_geometry,
    _write_candidate_stacks,
    default_line_run_parameter,
    run_line_analysis,
)
from fwhm_reviewer import ReviewProject


class LineGeometryTests(unittest.TestCase):
    def test_diagonal_skeleton_uses_geodesic_length(self) -> None:
        points = np.array([[index, index] for index in range(21)], dtype=int)

        direction, curvature, local_count = _local_geometry(
            points,
            np.array([10.0, 10.0]),
            3.32,
        )

        self.assertIsNotNone(direction)
        self.assertIsNotNone(curvature)
        self.assertGreaterEqual(local_count, 7)
        self.assertAlmostEqual(abs(float(direction[0])), 2**-0.5, places=2)
        self.assertAlmostEqual(abs(float(direction[1])), 2**-0.5, places=2)
        self.assertLess(float(curvature), 1.0)

    def test_compact_stem_caps_long_names_and_keeps_identity(self) -> None:
        first = Path("A" * 120 + ".tif")
        second = Path("A" * 119 + "B.tif")

        first_name = compact_stem(first)
        second_name = compact_stem(second)

        self.assertLessEqual(len(first_name), 48)
        self.assertNotEqual(first_name, second_name)

    def test_candidate_stacks_use_thin_lines_and_roi_labels(self) -> None:
        image = np.zeros((5, 48, 48), dtype=np.float32)
        row = {
            "roi_id": 12,
            "z_px": 2,
            "y_px": 24.0,
            "x_px": 20.0,
            "profile_half_length_xy_px": 8,
            "profile_half_length_z_px": 2,
            "normal_x": 1.0,
            "normal_y": 0.0,
            "accepted_xy": True,
            "accepted_z": True,
        }

        with tempfile.TemporaryDirectory() as temporary:
            xy_path = Path(temporary) / "xy.tif"
            z_path = Path(temporary) / "z.tif"
            _write_candidate_stacks(image, [row], xy_path, z_path)
            xy_stack = tifffile.imread(xy_path)
            z_stack = tifffile.imread(z_path)

        green = np.array([0, 255, 132], dtype=np.uint8)
        self.assertTrue(np.all(xy_stack[2, 24, 12:29] == green))
        self.assertFalse(np.any(np.all(xy_stack[2, 23, 12:29] == green, axis=-1)))
        self.assertTrue(np.all(z_stack[2, 24, 17:24] == green))
        self.assertFalse(np.any(np.all(z_stack[2, 23, 17:24] == green, axis=-1)))
        for stack in (xy_stack, z_stack):
            label_pixels = (
                (stack[2, ..., 0] > 200)
                & (stack[2, ..., 1] > 180)
                & (stack[2, ..., 2] < 130)
            )
            self.assertTrue(np.any(label_pixels))

    def test_line_analysis_precomputes_review_assets(self) -> None:
        rng = np.random.default_rng(0)
        z, y, x = np.indices((16, 128, 128), dtype=np.float64)
        image = (
            20.0
            + 400.0
            * np.exp(-0.5 * ((x - 64.0) / 1.5) ** 2)
            * np.exp(-0.5 * ((z - 8.0) / 1.2) ** 2)
            + rng.normal(0.0, 1.0, size=(16, 128, 128))
        ).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "line.tif"
            output = root / "result"
            tifffile.imwrite(source, image, metadata={"axes": "ZYX"})
            parameters = default_line_run_parameter()
            parameters.update({"input_path": source, "output_path": output})

            summary = run_line_analysis(
                physical_parameters(30.1, 96.9, 100.0, 290.0),
                parameters,
            )

            self.assertGreater(summary["accepted_roi_count_xy"], 0)
            self.assertTrue((output / "candidates.png").is_file())
            with tifffile.TiffFile(output / "candidates-XY.tif") as xy_tiff:
                self.assertEqual(xy_tiff.series[0].axes, "ZYXS")
            with tifffile.TiffFile(output / "candidates-Z.tif") as z_tiff:
                self.assertEqual(z_tiff.series[0].axes, "ZYXS")
            self.assertTrue((output / "profiles.pdf").is_file())
            self.assertEqual(
                len(list((output / "review-assets" / "XY").glob("*.png")))
                + len(list((output / "review-assets" / "Z").glob("*.png"))),
                summary["review_asset_count"],
            )

            project = ReviewProject(output)
            manifest = project.manifest()
            xy_items = [
                item for item in manifest["items"] if item["direction"] == "xy"
            ]
            z_bases = {
                item["key"].rsplit("-", 1)[0]
                for item in manifest["items"]
                if item["direction"] == "z"
            }
            xy_item = next(
                item
                for item in xy_items
                if item["key"].rsplit("-", 1)[0] in z_bases
            )
            project.save_decision(xy_item["key"], "delete")
            reviewed = project.generate_reviewed()
            reviewed_summary = json.loads(
                (reviewed / "fwhm_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reviewed_summary["deleted_xy_count"], 1)
            self.assertEqual(reviewed_summary["deleted_z_count"], 0)


if __name__ == "__main__":
    unittest.main()
