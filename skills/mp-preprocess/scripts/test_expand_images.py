from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from expand_images import _center_pad_or_crop, expand_batch


class ExpandImagesTest(unittest.TestCase):
    def test_center_pad_and_crop_preserves_center(self) -> None:
        image = np.arange(27).reshape(3, 3, 3)

        adjusted = _center_pad_or_crop(image, (2, 4, 2))

        self.assertEqual(adjusted.shape, (2, 4, 2))
        np.testing.assert_array_equal(adjusted[:, 1:4, :], image[1:3, :, 1:3])

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
