# sim-skills

A Git-managed collection of reusable MP-SIM and MF-SIM skills for Codex,
Claude, OpenCode, and other agents that discover `~/.agents/skills`.

## Skills

- `r-3dmpsim`: orchestrates acquisition through 3D reconstruction.
- `r-mfsim`: orchestrates MP Z slicing and MF reconstruction.
- `mp-data-ingest`: downloads and normalizes camera data.
- `mp-preprocess`: locates the project real entry and provides padding.
- `simcheck`: computes phase-aware SIM quality metrics.
- `mp-reconstruct`: locates and runs the project 3D reconstruction entry.
- `mf-reconstruct`: slices MP stacks and runs the project MF entry.
- `raw-check`: computes preliminary FPJ from camera raw data.
- `fwhm`: measures bead or thin-line FWHM with auditable ROI evidence.
- `decorr`: measures lateral and axial decorrelation resolution.

Reusable preprocessing, inspection, and measurement scripts are bundled under
their owning skills. Formal `Run_real_MP*.py`, `0-run5*.py`, and `Run_MF*.py`
entries, their imported algorithms, physical parameters, calibrated OTF files,
runtime data, credentials, and GPU drivers remain in the research project.

## Python environment

Utility skills use this order:

1. `SIM_SKILLS_PYTHON` when the user explicitly supplies an interpreter.
2. One dependency-complete `3DMPSIM*` Conda environment.
3. This repository's shared `.venv`.

If multiple compatible `3DMPSIM*` environments exist, ask the user to choose.
If none exists, ask before creating the CPU-only shared environment:

```powershell
& (Join-Path $repo 'environment.ps1') -CreateUv
```

The UV environment does not install PyTorch, CuPy, CUDA, or reconstruction
dependencies. `decorr` can use CUDA only when a selected existing environment
already provides a compatible CUDA-enabled PyTorch. Formal project entries
always use the project-selected environment, never this UV fallback.

## Install or update

The recommended checkout is `$env:USERPROFILE\.agents\sim-skills`, but the
repository may be cloned anywhere. Do not hardcode a full user-profile path.
The installer resolves skill sources from its own location and creates one
directory junction per skill under the current user's `.agents\skills`, so
every supported agent uses the same Git-managed files. Pass `-TargetRoot` when
an agent uses a different shared skill directory.

```powershell
$repo = Join-Path $env:USERPROFILE '.agents\sim-skills'

if (Test-Path -LiteralPath (Join-Path $repo '.git')) {
    git -C $repo pull --ff-only
} else {
    gh repo clone gouwenct/sim-skills $repo
}

& (Join-Path $repo 'install.ps1')
```

The GitHub repository is private, so GitHub CLI authentication is required.
The installer is idempotent for links that already target this checkout. By
default it never replaces a real directory or a link to another location. After
the user explicitly confirms migration, back up conflicts and switch all links:

```powershell
& (Join-Path $repo 'install.ps1') -MigrateExisting
```

Backups are written below `.agents/skills-backup/<timestamp>/` and are never
deleted by the installer. Pass `-BackupRoot` to choose another location.

An AI installer should follow the same sequence: clone or fast-forward pull,
run `install.ps1`, report conflicts without changing them, then verify that all
ten links resolve to this checkout. It must not use `-MigrateExisting` or
`-CreateUv` without explicit user confirmation.
