from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import run_hyperstack as hyp  # noqa: E402


def _write_zyx(path: Path, value: float, shape=(4, 5, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full(shape, value, dtype=np.float32)
    tifffile.imwrite(path, data, photometric="minisblack")


class HyperstackTests(unittest.TestCase):
    def test_natural_key_orders_decimal_and_integers(self) -> None:
        opld = sorted(["OTF_OPLD_1.0.tif", "OTF_OPLD_0.9.tif"], key=hyp.natural_key)
        imgs = sorted(["img10.tif", "img2.tif"], key=hyp.natural_key)
        self.assertEqual(opld, ["OTF_OPLD_0.9.tif", "OTF_OPLD_1.0.tif"])
        self.assertEqual(imgs, ["img2.tif", "img10.tif"])

    def test_stack_self_writes_sibling_tzyx(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "ssim_map"
            _write_zyx(folder / "OTF_OPLD_0.0.tif", 1.0)
            _write_zyx(folder / "OTF_OPLD_0.1.tif", 2.0)
            dest = hyp.write_hyperstack(folder)
            self.assertEqual(dest, folder.parent / "ssim_map_hyperstack.tif")
            with tifffile.TiffFile(dest) as handle:
                series = handle.series[0]
                self.assertEqual(series.axes, "TZYX")
                self.assertEqual(series.shape, (2, 4, 5, 6))
                self.assertTrue(handle.is_imagej)
                self.assertTrue(handle.imagej_metadata.get("hyperstack"))
                stacked = series.asarray()
            self.assertEqual(float(stacked[0].max()), 1.0)
            self.assertEqual(float(stacked[1].max()), 2.0)

    def test_existing_target_gets_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "wiener"
            _write_zyx(folder / "a.tif", 1.0)
            _write_zyx(folder / "b.tif", 2.0)
            first = hyp.write_hyperstack(folder)
            second = hyp.write_hyperstack(folder)
            self.assertEqual(first.name, "wiener_hyperstack.tif")
            self.assertTrue(second.name.startswith("wiener_hyperstack-"))
            self.assertNotEqual(first, second)
            self.assertTrue(second.exists())

    def test_skips_named_hyperstack_and_4d(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "src"
            _write_zyx(folder / "vol_0.tif", 1.0)
            _write_zyx(folder / "vol_1.tif", 2.0)
            tifffile.imwrite(
                folder / "old_hyperstack.tif",
                np.zeros((2, 4, 5, 6), dtype=np.float32),
                imagej=True,
                metadata={"axes": "TZYX"},
            )
            tifffile.imwrite(folder / "already4d.tif", np.zeros((2, 4, 5, 6), dtype=np.float32))
            items = hyp.qualified_3d_tiffs(folder)
            self.assertEqual([item["path"].name for item in items], ["vol_0.tif", "vol_1.tif"])

    def test_majority_shape_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "mix"
            _write_zyx(folder / "a.tif", 1.0, shape=(4, 5, 6))
            _write_zyx(folder / "b.tif", 2.0, shape=(4, 5, 6))
            _write_zyx(folder / "odd.tif", 9.0, shape=(3, 3, 3))
            items = hyp.qualified_3d_tiffs(folder)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["shape"], (4, 5, 6))

    def test_parent_stacks_class_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "raw_OPLD_0.0"
            for name, value in (("wiener", 1.0), ("gaussian", 2.0)):
                _write_zyx(root / name / "t0.tif", value)
                _write_zyx(root / name / "t1.tif", value + 0.5)
            stackable, ambiguous, too_deep = hyp.discover_tiff_folders(root, max_depth=2)
            self.assertEqual(ambiguous, [])
            self.assertEqual(too_deep, [])
            self.assertEqual(sorted(p.name for p in stackable), ["gaussian", "wiener"])

    def test_ambiguous_when_both_layers_qualify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "both"
            _write_zyx(root / "top0.tif", 1.0)
            _write_zyx(root / "top1.tif", 2.0)
            _write_zyx(root / "child" / "c0.tif", 3.0)
            _write_zyx(root / "child" / "c1.tif", 4.0)
            stackable, ambiguous, too_deep = hyp.discover_tiff_folders(root, max_depth=2)
            self.assertEqual([p.name for p in ambiguous], ["both"])
            self.assertEqual(stackable, [])
            self.assertEqual(too_deep, [])
            self.assertEqual(hyp.main([str(root)]), 2)

    def test_three_level_reaches_grandchildren(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roi21-tifs"
            leaf = root / "raw_OPLD_0.0" / "ssim_map"
            _write_zyx(leaf / "a.tif", 1.0)
            _write_zyx(leaf / "b.tif", 2.0)
            stackable, ambiguous, too_deep = hyp.discover_tiff_folders(root, max_depth=2)
            self.assertEqual(ambiguous, [])
            self.assertEqual(too_deep, [])
            self.assertEqual([p.name for p in stackable], ["ssim_map"])

    def test_depth_three_needs_confirm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "top"
            leaf = root / "a" / "b" / "c" / "leaf"
            _write_zyx(leaf / "a.tif", 1.0)
            _write_zyx(leaf / "b.tif", 2.0)
            stackable, ambiguous, too_deep = hyp.discover_tiff_folders(root, max_depth=2)
            self.assertEqual(stackable, [])
            self.assertTrue(too_deep)


if __name__ == "__main__":
    unittest.main()
