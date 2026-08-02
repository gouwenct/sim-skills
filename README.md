# sim-skills

A self-contained collection of MP-SIM and MF-SIM skills for Codex, Claude,
OpenCode, and other agents that discover `~/.agents/skills`.

## Skills

- `r-3dmpsim`: orchestrates acquisition through 3D reconstruction.
- `r-mfsim`: orchestrates MP Z slicing and MF reconstruction.
- `mp-data-ingest`: downloads and normalizes camera data.
- `mp-preprocess`: registers, corrects, and pads raw data.
- `simcheck`: computes phase-aware SIM quality metrics.
- `mp-reconstruct`: estimates parameters and reconstructs 3D MP-SIM.
- `mf-reconstruct`: slices MP stacks and reconstructs MF-SIM.
- `raw-check`: computes preliminary FPJ from camera raw data.

Algorithms and entry templates are bundled under their owning skills. Runtime
data, calibrated OTF files, credentials, Python packages, and GPU drivers remain
external inputs.

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
The installer is idempotent for links that already target this checkout. It
never replaces a real directory or a link to another location. If a legacy
skill conflicts, stop, back it up after explicit user confirmation, and rerun
the installer.

An AI installer should follow the same sequence: clone or fast-forward pull,
run `install.ps1`, report conflicts without changing them, then verify that all
eight links resolve to this checkout.
