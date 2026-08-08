---
name: raw-check
description: Use when paired cam1/cam2 MP-SIM raw TIFF data may contain dropped, missing, duplicated, or repeated frames, or when camera frame counts and acquisition timing need verification before preprocessing.
---

# Raw Check

Scan every page in every discovered cam1/cam2 TIFF without modifying raw data.
Treat each camera as `n_z x directions x phases` frames per timepoint; the
default 3x5 acquisition is 15 frames per camera, or 30 across both cameras.

```powershell
& <python> -B <SKILL_DIR>/scripts/check_raw_frames.py `
  --input-dir <sample-with-cam1-and-cam2-tiffs>
```

The command checks all naturally sorted `cam1*` and `cam2*` TIFFs directly in
the sample directory and in optional `cam1/` or `cam2/` subdirectories. Use
`--n-z`, `--n-directions`, and `--n-phases` only when acquisition geometry is
not the default. Run `--help` for the clock-gap calibration option.

## Interpret Results

| Result | Meaning |
| --- | --- |
| `CAMERA_FRAME_COUNT_MISMATCH` | The two cameras stored different page counts. |
| `INCOMPLETE_TIMEPOINT` | A camera page count is not divisible by the configured acquisition block. |
| `FRAME_INDEX_GAP` / `TIMING_GAP` | Metadata suggests missing frames near the reported 1-based timepoint. |
| `DUPLICATE_STATS` | Adjacent frames have exactly equal mean, max, min, and standard deviation. |
| `RESULT: CLEAN` | No configured check found an anomaly. |

Exit code is `0` for clean data, `1` for detected or suspected anomalies, and
`2` for invalid inputs. File-size difference is informational because cam1 and
cam2 TIFF metadata overhead differs.

## Common Mistakes

- Do not call a timestamp gap proof of a dropped frame; report it as suspected.
- Do not claim a clean count can detect a complete timepoint missing from both
  cameras when metadata has no gap.
- Do not use this Skill for FPJ; use `raw-fpj`.
