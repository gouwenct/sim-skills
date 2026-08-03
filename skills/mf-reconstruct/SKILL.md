---
name: mf-reconstruct
description: Split one 120-frame 3D-MP Dr timepoint into eight 15-frame MF-SIM slices, then locate and run the research project's MF reconstruction entrypoint. Use for MF-SIM or 2D-MF processing after a 3D-MP Dr stack exists.
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

1. Use the project path supplied by the user. Otherwise search upward from the
   current directory for `code/2D-MF-recon`.
2. If neither method finds the project, ask the user for its path. Do not scan
   whole drives or use a machine-specific default.
3. Select a matching `Run_MF*.py` project entry. If none matches, copy the
   nearest matching project entry to a new run-specific filename and edit only
   its parameter section; never copy or modify algorithm modules.
4. Confirm `FANWEI` using the actual unsorted `Path.iterdir()` order.
5. Mirror dx, optical settings, notch, `mear`, W, `loadflag`, and `just_cfai`
   from the selected 3D project entry where applicable.
6. Run with the project-selected `3DMPSIM*` interpreter.

```powershell
& <project-python> -B <project-root>/code/2D-MF-recon/<Run_MF-entry.py>
```

MF mode requires `flag_N=5`, `Nshifts=5`, and `Nangles=3`. Do not substitute
the classic 2D-SIM `flag_N=3` mode.
