from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from slice_mp_dr_to_mf import slice_one_stack


class SliceMpDrTest(unittest.TestCase):
    def test_splits_z_major_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.tif"
            tifffile.imwrite(source, np.arange(24, dtype=np.float32).reshape(6, 2, 2))

            written = slice_one_stack(source, root / "out", n_z=2, n_frames=3)

            self.assertEqual(len(written), 2)
            self.assertEqual(tifffile.imread(written[0]).shape, (3, 2, 2))


if __name__ == "__main__":
    unittest.main()
