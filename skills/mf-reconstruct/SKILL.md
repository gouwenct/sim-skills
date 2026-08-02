---
name: mf-reconstruct
description: Split one 120-frame 3D-MP Dr timepoint into eight 15-frame MF-SIM slices and run 2D MF reconstruction with bundled algorithms. Use for MF-SIM or 2D-MF processing after a 3D-MP Dr stack exists.
---

# MF Reconstruct

## Slice

Validate `(120,H,W) = 8 x 15` and write eight `(15,H,W)` stacks:

```powershell
& <python> -B <SKILL_DIR>/scripts/slice_mp_dr_to_mf.py `
  --src <single-timepoint.tif> --out-dir <Dr/01-488_padding_slice>
```

Do not reshape an unexpected frame count. Use `--overwrite` only after explicit
confirmation.

## Reconstruct

1. Copy `assets/run_mf_reconstruct_template.py` to a run-specific directory.
2. Set `MF_SIM_DATA_ROOT` to the batch root and `MF_SIM_OTF_ROOT` to the 2D-MF
   OTF directory.
3. Put `<SKILL_DIR>/scripts` on `PYTHONPATH`.
4. Confirm `FANWEI` using the actual unsorted `Path.iterdir()` order.
5. Mirror dx, optical settings, notch, `mear`, W, `loadflag`, and `just_cfai`
   from the selected 3D run where applicable.

```powershell
$env:MF_SIM_DATA_ROOT = '<dataOut-batch>'
$env:MF_SIM_OTF_ROOT = '<OTF-2D-MF-directory>'
$env:PYTHONPATH = '<SKILL_DIR>/scripts'
& <python> -B <copied-run-entry.py>
```

MF mode requires `flag_N=5`, `Nshifts=5`, and `Nangles=3`. Do not substitute
the classic 2D-SIM `flag_N=3` mode.
