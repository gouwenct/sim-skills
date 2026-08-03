---
name: mp-preprocess
description: Locate and run a research project's MP-SIM real-preprocessing entrypoint, then pad generated Dr stacks with the bundled deterministic utility. Use for the real preprocessing and padding stages after raw layout is normalized.
---

# MP Preprocess

Keep formal preprocessing entrypoints, algorithms, and physical parameters in
the research project. This skill owns only project discovery and padding.

## Real Preprocessing

1. Use the project path supplied by the user. Otherwise search upward from the
   current directory for `code/real`.
2. If neither method finds the project, ask the user for its path. Do not scan
   whole drives and do not use a machine-specific default.
3. Select a matching `Run_real_MP*.py` entry in the project. If none matches
   the experiment, copy the nearest matching project entry to a new run-specific
   filename and edit only its parameter section.
4. Confirm `path_input`, `peizhun_file`, `FANWEI`, `W_order`, and
   `save_reg_only`.
5. Run the project entry with the project-selected `3DMPSIM*` interpreter.

```powershell
& <project-python> -B <project-root>/code/real/<Run_real_MP-entry.py>
```

`save_reg_only=0` writes the corrected A track. `save_reg_only=1` additionally
writes the registration-only B track. Reconstruction consumes only A.

## Padding

Run the bundled CPU utility against the generated batch root:

```powershell
& <python> -B <SKILL_DIR>/scripts/expand_images.py <dataOut-batch> `
  --x 462 --y 462 --z 120
```

The command writes sibling directories suffixed `_padding` and skips existing
outputs. Confirm the root before running because it writes TIFF files.

## Contract

Input is a normalized camera batch. Output is `<batch>/<sample>/Dr/<track>/`
with timepoint TIFF stacks, plus optional `<track>_padding/` directories.
