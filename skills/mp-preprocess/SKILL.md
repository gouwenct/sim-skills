---
name: mp-preprocess
description: Convert MP-SIM cam1/cam2 acquisitions into registered and brightness-corrected Dr stacks, with optional registration-only output and 462x462x120 padding. Use for the real preprocessing and padding stages after raw layout is normalized.
---

# MP Preprocess

Use the bundled algorithm modules in `scripts/`; do not import a project-local
`code/real` directory. Use the MP-SIM Python environment with NumPy, tifffile,
OpenCV, SciPy, scikit-image, matplotlib, and Torch.

## Real Preprocessing

1. Copy `assets/run_real_template.py` to a run-specific working directory.
2. Set `MP_SIM_RAW_ROOT` to the normalized raw batch.
3. Put `<SKILL_DIR>/scripts` on `PYTHONPATH`.
4. Edit only the template's marked experiment parameters: `peizhun_file`,
   `FANWEI`, `W_order`, and `save_reg_only`.
5. Run the copied entry and retain it with the experiment record.

```powershell
$env:MP_SIM_RAW_ROOT = '<raw-batch>'
$env:PYTHONPATH = '<SKILL_DIR>/scripts'
& <python> -B <copied-run-entry.py>
```

`save_reg_only=0` writes the corrected A track. `save_reg_only=1` additionally
writes the registration-only B track. Reconstruction consumes only A.

## Padding

Run the bundled CLI against the generated batch root:

```powershell
& <python> -B <SKILL_DIR>/scripts/expand_images.py <dataOut-batch> `
  --x 462 --y 462 --z 120
```

The command writes sibling directories suffixed `_padding` and skips existing
outputs. Confirm the root before running because it writes TIFF files.

## Contract

Input is a normalized camera batch. Output is `<batch>/<sample>/Dr/<track>/`
with timepoint TIFF stacks, plus optional `<track>_padding/` directories.
