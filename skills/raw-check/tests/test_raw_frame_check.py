from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_raw_frames.py"


def _format_cam1_time(milliseconds: float) -> str:
    seconds = milliseconds / 1000.0
    return (
        "[ CAPTURE TIME ]\n"
        f" Time_From_Start = 00:00:{seconds:07.4f}\n"
    )


def _write_stack(
    path: Path,
    frame_numbers: list[int],
    *,
    duplicate_at: int | None = None,
    timestamps_ms: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    with tifffile.TiffWriter(path) as writer:
        for local_index, frame_number in enumerate(frame_numbers):
            frame = np.array(
                [[frame_number, frame_number + 1], [frame_number + 2, frame_number + 4]],
                dtype=np.uint16,
            )
            if duplicate_at == local_index:
                frame = previous.copy()
            description = None
            if timestamps_ms is not None:
                description = _format_cam1_time(timestamps_ms[local_index])
            writer.write(frame, description=description)
            previous = frame


class RawFrameCheckTests(unittest.TestCase):
    def _run(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--input-dir", str(root)],
            text=True,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )

    def test_clean_pair_reads_all_cam1_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_stack(root / "cam1_part1.tif", list(range(10)))
            _write_stack(root / "cam1_part2.tif", list(range(10, 30)))
            _write_stack(root / "cam2" / "cam2_stack.tif", list(range(30)))

            result = self._run(root)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("cam1: files=2 frames=30", result.stdout)
            self.assertIn("RESULT: CLEAN", result.stdout)

    def test_frame_count_mismatch_reports_incomplete_timepoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_stack(root / "cam1_stack.tif", list(range(29)))
            _write_stack(root / "cam2_stack.tif", list(range(30)))

            result = self._run(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("CAMERA_FRAME_COUNT_MISMATCH: cam1=29 cam2=30", result.stdout)
            self.assertIn(
                "INCOMPLETE_TIMEPOINT: camera=cam1 timepoint=2 frames=14/15",
                result.stdout,
            )

    def test_duplicate_stats_reports_frame_pair_and_timepoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_stack(root / "cam1_stack.tif", list(range(30)), duplicate_at=17)
            _write_stack(root / "cam2_stack.tif", list(range(30)))

            result = self._run(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(
                "DUPLICATE_STATS: camera=cam1 frames=17,18 timepoint=2",
                result.stdout,
            )

    def test_timestamp_gap_reports_estimated_missing_timepoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            timestamps = [index * 3.4 for index in range(30)]
            timestamps[17:] = [value + 3.4 for value in timestamps[17:]]
            _write_stack(
                root / "cam1_stack.tif",
                list(range(30)),
                timestamps_ms=timestamps,
            )
            _write_stack(root / "cam2_stack.tif", list(range(30)))

            result = self._run(root)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn(
                "TIMING_GAP: camera=cam1 after_frame=17 before_frame=18 "
                "estimated_missing=1 timepoint=2",
                result.stdout,
            )


if __name__ == "__main__":
    unittest.main()
