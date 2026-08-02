---
name: r-mfsim
description: Orchestrate MF-SIM processing from an existing 3D-MP Dr timepoint through Z slicing and 2D MF reconstruction. Use when the user invokes R-MFSIM or asks to run MF-SIM, 2D-MF, or padding-slice processing.
---

# R-MFSIM

This is an orchestration skill. Use `../mf-reconstruct/SKILL.md`; do not call
project-local `code/tool` or `code/2D-MF-recon` scripts.

Before writing, confirm the batch, samples, source Dr track, timepoints, output
slice directory, overwrite policy, OTF path, FANWEI mapping, mirrored 3D optical
parameters, `loadflag`, `just_cfai`, and whether to run slicing, reconstruction,
or both.

The source timepoint must have 120 frames arranged as 8 Z layers x 3 directions
x 5 phases. The leaf skill owns both the deterministic slicing script and MF
reconstruction modules. Stop rather than reshaping invalid input or using the
classic 3-phase 2D-SIM mode.
