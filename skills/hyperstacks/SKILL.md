---
name: hyperstacks
description: Use when the user wants to turn a folder of same-type 3D TIFFs into an ImageJ hyperstack, mentions /hyperstacks, hyperstack / TZYX / 同类 TIFF 文件夹, or asks to stack Wiener, Gaussian, SSIM, or OPLD volumes for ImageJ T and Z sliders.
---

# hyperstacks

Stack same-type 3D TIFFs into one sibling ImageJ hyperstack. Do not put the hyperstack inside the single-volume folder.

## Run

```powershell
& $env:SIM_SKILLS_PYTHON -B "$SKILL_DIR\scripts\run_hyperstack.py" "<folder>"
```

If `SIM_SKILLS_PYTHON` is unset, resolve Python with the sim-skills `core` profile (`environment.ps1` / `3DMPSIM*`). Run `--help` for `--output-dir`, `--max-depth`, and `--dry-run`.

Default output is beside the folder being stacked: `<parent>\<folder-name>_hyperstack.tif`. If that file exists, write `<folder-name>_hyperstack-YYYYMMDD-HHMMSS.tif` instead. Never overwrite.

## Rules

- Auto-route: this layer is mostly same-shape 3D TIFFs → stack this layer. This layer is mostly TIFF subfolders → stack each such subfolder. Both layers have qualified 3D TIFFs → stop and ask.
- Keep only 3D volumes that share the majority shape. Skip `*hyperstack*`, 2D, 4D, and ImageJ `TZYX`/`TZCYX`. Need at least two files.
- T follows natural filename order. Unlabeled 3D is `ZYX`. Output is `float32` ImageJ `TZYX`. Do not call `_U.save_hyperstack` (it stretches to uint16).
- Default depth is the specified folder plus children plus grandchildren (`--max-depth 2`). Deeper TIFF folders require confirmation.

## Check

After a write, confirm ImageJ metadata has `hyperstack=true`, `frames=T`, `slices=Z`, and dtype `float32`. Report the output path and any `NEED_CONFIRM` lines.
