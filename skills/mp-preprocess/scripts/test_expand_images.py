from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from expand_images import expand_batch


class ExpandImagesTest(unittest.TestCase):
    def test_writes_padded_sibling_stack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "sample" / "Dr" / "01-488"
            source_dir.mkdir(parents=True)
            tifffile.imwrite(source_dir / "input.tif", np.ones((2, 2, 2), dtype=np.float32))

            written = expand_batch(root, target_x=4, target_y=4, target_z=3)

            self.assertEqual(len(written), 1)
            self.assertEqual(tifffile.imread(written[0]).shape, (3, 4, 4))


if __name__ == "__main__":
    unittest.main()
