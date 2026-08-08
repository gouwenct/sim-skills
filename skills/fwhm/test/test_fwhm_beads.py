from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fwhm_beads import build_summary, default_run_parameter
from fwhm_common import physical_parameters


class BeadMedianSelectionTests(unittest.TestCase):
    def test_keeps_the_same_rois_near_both_median_baselines(self) -> None:
        xy_values = [100, 160, 101, 99, 100, 102, 200, 201]
        z_values = [360, 300, 301, 299, 302, 298, 400, 401]
        rows = [
            {
                "source": "sample.tif",
                "candidate_id": index + 1,
                "accepted": True,
                "rejection_reasons": "",
                "fwhm_xy_nm": float(xy),
                "fwhm_z_nm": float(z),
            }
            for index, (xy, z) in enumerate(zip(xy_values, z_values))
        ]
        parameters = default_run_parameter()
        parameters["median_nearby_count"] = 2

        summary = build_summary(
            rows,
            physical_parameters(30.1, 96.9),
            parameters,
        )
        saved = [
            row
            for row in rows
            if row["median_selected_xy"] and row["median_selected_z"]
        ]

        self.assertEqual([row["candidate_id"] for row in saved], [3, 5])
        self.assertEqual(summary["median_selection"]["xy"]["baseline_nm"], 101.5)
        self.assertEqual(summary["median_selection"]["z"]["baseline_nm"], 301.5)
        self.assertEqual(summary["saved_row_count"], 2)
        self.assertEqual(summary["final_xy_roi_count"], 2)
        self.assertEqual(summary["final_z_roi_count"], 2)


if __name__ == "__main__":
    unittest.main()
