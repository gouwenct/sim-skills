---
name: r-3dmpsim
description: Orchestrate the complete 3D MP-SIM workflow from acquisition through preprocessing, SIMcheck, parameter estimation, and optional reconstruction. Use when the user says run a 3DMP experiment, invokes R-3DMPSIM, or supplies an MP-SIM raw-data link.
---

# R-3DMPSIM

This is an orchestration skill. Use sibling skills for reusable utilities and
the research project's formal real/reconstruction entrypoints.

## Confirm First

Show and confirm: data source and destination, batch, samples, `peizhun_file`,
FANWEI mapping, `W_order`, `save_reg_only`, padding size, SIMcheck checks and
timepoint, reconstruction FANWEI, Dr track, OTF path, `just_cfai`, load mode,
device, output level, and permission to write.

Default to corrected A only, 462x462x120 padding, SIMcheck MCNR+FPJ at timepoint
3, GPU auto selection, `just_cfai=1`, and no full reconstruction.

## Stages

1. Follow `../mp-data-ingest/SKILL.md` to acquire and normalize raw data.
2. Follow `../mp-preprocess/SKILL.md` to generate Dr and padded Dr stacks.
3. Follow `../simcheck/SKILL.md` to run MCNR and FPJ on padded A; B is optional.
4. Follow `../mp-reconstruct/SKILL.md` to estimate parameters or reconstruct A.
5. Offer `../decorr/SKILL.md` for optional effective-resolution measurement.
6. Offer `../fwhm/SKILL.md` only for confirmed beads or thin line structures.

Stop at the failing stage. If project discovery from an explicit path or upward
search fails, ask the user for the project path. Preserve the path identity:

```text
raw batch X = preprocessing input = dataOut/X = SIMcheck batch = recon data root
```

After Stage 3, report MCNR and FPJ before starting Stage 4. If a lihe artifact
exists after Stage 4, create its preview and report it; never invent a missing
artifact. Upload results only when the user authorizes the configured Tencent
Docs destination; that integration is bundled under the `simcheck` skill.
