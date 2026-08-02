---
name: mp-reconstruct
description: Estimate illumination parameters and optionally reconstruct padded 3D MP-SIM Dr stacks with the bundled CPU/GPU algorithm modules. Use after preprocessing and SIMcheck when running the 3D reconstruction stage.
---

# MP Reconstruct

Use the bundled modules in `scripts/`; never load reconstruction code from an
external research-project folder. Keep physical parameters in the template's
`argparse.Namespace` and paths, switches, filenames, and run choices in
`run_parameter`.

## Run

1. Copy `assets/run_mp_reconstruct_template.py` to a run-specific directory.
2. Set `MP_SIM_RECON_DATA_ROOT` to the `dataOut/<batch>` root.
3. Set `MP_SIM_OTF_ROOT` to the MP OTF directory.
4. Put `<SKILL_DIR>/scripts` on `PYTHONPATH`.
5. Set `MP_SIM_FANWEI`, `MP_SIM_PARAMETER_DEVICE`, and
   `MP_SIM_PARAMETER_OUTPUT` as needed, then run the copied entry.

```powershell
$env:MP_SIM_RECON_DATA_ROOT = '<dataOut-batch>'
$env:MP_SIM_OTF_ROOT = '<OTF-MP-8-directory>'
$env:MP_SIM_FANWEI = '1'
$env:MP_SIM_PARAMETER_DEVICE = 'auto'
$env:MP_SIM_PARAMETER_OUTPUT = 'cfai_png'
$env:PYTHONPATH = '<SKILL_DIR>/scripts'
& <python> -B <copied-run-entry.py>
```

Default to `just_cfai=1` for parameter estimation. Use `just_cfai=0` only when
the user explicitly requests full reconstruction. Use only the corrected A
track, normally `01-488_padding`.

## Contract

Input is a padded Dr batch plus calibrated OTF data. Output is parameter files,
logs, optional `P/lihe_*.tif`, and full reconstructed volumes only when enabled.
