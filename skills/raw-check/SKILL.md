---
name: raw-check
description: Compute preliminary FPJ directly from MP-SIM cam1/cam2 raw TIFF data before registration or correction. Use to inspect camera regions 4 and 5 for 3x3, 3x5, 3x7, or 4x13 acquisitions.
---

# Raw Check

Use the bundled entry point. It loads the sibling `simcheck` implementation;
override discovery with `SIMCHECK_SKILL_DIR` only when necessary.

```powershell
& <python> -B <SKILL_DIR>/scripts/fpj_raw_cam12.py `
  --input-dir <folder-with-cam1-and-cam2-tiffs>
```

Alternatively pass `--cam1` and `--cam2`. The default is 3 directions x 5
phases; set `--n-phases` for another acquisition. This is a preliminary raw
check without registration, background subtraction, direction correction, or
phase correction. Do not compare it directly with final processed metrics.

Outputs are TIFF and PNG previews under `raw_fpj/`.
