---
name: mp-reconstruct
description: Locate, configure, and run a research project's 3D MP-SIM reconstruction entrypoint. Use after preprocessing and SIMcheck for parameter estimation or explicitly requested full reconstruction.
---

# MP Reconstruct

Keep the formal entrypoint, reconstruction algorithms, calibrated OTF data, and
physical parameters in the research project. This skill does not bundle copies.

## Run

1. Use the project path supplied by the user. Otherwise search upward from the
   current directory for `code/recon`.
2. If neither method finds the project, ask the user for its path. Do not scan
   whole drives or use a machine-specific default.
3. Select a matching `0-run5*.py` project entry. If none matches, copy the
   nearest matching project entry to a new run-specific filename and edit only
   its parameter section; never modify algorithm modules.
4. Print the actual sorted sample-directory mapping before confirming `FANWEI`.
5. Confirm the absolute input/output paths, OTF, `Dr_sub`, `loadflag`,
   `just_cfai`, parameter device, and output level.
6. Run the entry with the project-selected `3DMPSIM*` interpreter.

```powershell
& <project-python> -B <project-root>/code/recon/<0-run5-entry.py>
```

Default to `just_cfai=1` for parameter estimation. Use `just_cfai=0` only when
the user explicitly requests full reconstruction. Use only the corrected A
track, normally `01-488_padding`.

## Contract

Input is a padded Dr batch plus calibrated OTF data. Output is parameter files,
logs, optional `P/lihe_*.tif`, and full reconstructed volumes only when enabled.
