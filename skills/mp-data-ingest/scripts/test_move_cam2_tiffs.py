from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from move_cam2_tiffs import move_cam2_tiffs


class MoveCam2TiffsTests(unittest.TestCase):
    def test_execute_numbers_and_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("mp10", "mp2", "peizhun2", "mp1"):
                cam2 = root / name / "cam2"
                cam2.mkdir(parents=True)
                (cam2 / f"{name}.tif").write_bytes(b"tiff")

            summary = move_cam2_tiffs(root, execute=True)

            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"01、mp1", "02、mp2", "03、mp10", "99、peizhun2"},
            )
            self.assertEqual(summary.renamed, 4)
            self.assertEqual(summary.moved, 4)

    def test_dry_run_keeps_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "mp1" / "cam2" / "data.tif"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"tiff")

            summary = move_cam2_tiffs(root, execute=False)

            self.assertTrue(source.is_file())
            self.assertEqual(summary.planned_renames, 1)
            self.assertEqual(summary.moved, 0)


if __name__ == "__main__":
    unittest.main()
