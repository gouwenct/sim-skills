---
name: simcheck
description: Analyze phase-resolved 2D-SIM, 3D-SIM, 3DMP-SIM, or 4B-SIM TIFF stacks with explicit acquisition dimensions and protocol-aware FPJ/MCNR boundaries. Use when Codex must validate page ordering, reduce the phase axis, compute FPJ, compute phase-count-aware MCNR, or process 3x3, 3x5, 3x7, 4x13, or generic SIM data.
---

# SIMcheck

Analyze a phase-resolved stack without hardcoding frame count, direction count, or phase count.

## Terminology

- In prose, always write **相位** in Chinese or **phase** in English.
- Never use standalone `P` to mean 相位; it can be confused with a wave vector.
- Single-letter `p` is allowed only inside a dimension signature, index, formula, or code, such as `I[c,t,z,a,p,y,x]`.
- Write **方向** for the acquisition direction. Use `a` only in dimensions, formulas, or code.
- Call the input a phase-resolved Dr/SIM stack unless it is confirmed to be camera raw data.

## Establish The Acquisition Contract

Before processing, obtain:

- imaging mode;
- channel count;
- timepoint count;
- Z count;
- direction count;
- phase count;
- stored page order;
- whether phases are uniformly spaced.

Do not infer a factorization from page count alone. Verify:

```text
pages = channels * timepoints * Z * directions * phases
```

Read [references/acquisition-contract.md](references/acquisition-contract.md) when page order, singleton axes, multiple channels/timepoints, or output serialization needs interpretation.

## Preflight

1. Resolve this skill directory as `SKILL_DIR`.
2. Select an interpreter through the repository environment policy: explicit `SIM_SKILLS_PYTHON`, one compatible `3DMPSIM*`, then the confirmed shared UV environment. FPJ PNG preview additionally needs Pillow. Torch CUDA is optional and is never installed by the CPU fallback.
3. Inspect the input TIFF metadata without writing data.
4. Show a short checklist containing input scope, all acquisition dimensions, page order, requested outputs, device, and destination.
5. Stop for missing or ambiguous acquisition dimensions. A matching page-count product does not prove the order.
6. Check destination conflicts. Never overwrite unless the user explicitly authorizes it.

## Run The Workflow

Use the bundled entry point:

```powershell
& $python -B "$SKILL_DIR\scripts\run_simcheck.py" "<file-or-folder>" `
  --sim-mode <2d-sim|3d-sim|3d-mp-sim|4b-sim|generic|auto> `
  --n-z <Z> `
  --n-directions <directions> `
  --n-phases <phases> `
  --page-order <slowest-to-fastest-axes> `
  --output-root "<destination>" `
  --checks mcnr,fpj `
  --device auto
```

Use an explicit mode when the acquisition is known:

| Mode | Directions x phases | FPJ peak policy | MCNR policy |
| --- | ---: | --- | --- |
| `2d-sim` | 3 x 3 | first-order candidates | 1st harmonic, extended/unvalidated |
| `3d-sim` | 3 x 5 | first- and second-order candidates | 1st+2nd harmonics, SIMcheck thresholds |
| `3d-mp-sim` | 3 x 7 | visual FPJ, no automatic order labels | 1st+2nd+3rd harmonics, extended/unvalidated |
| `4b-sim` | 4 x 13 | visual FPJ, no automatic order labels | skipped |
| `generic` | explicit dimensions | legacy two-shell FPJ | legacy 1st+2nd harmonics, unvalidated |

`auto` selects a known mode only from the already supplied direction and phase counts; it does not infer dimensions or page order from page count.

For a folder, use `--name-glob "*_0001.tif"` and add `--recursive` only when nested traversal is intended.

Run `--dry-run` first when scope or destination is broad. Use:

- `--checks mcnr,fpj` for the normal legacy workflow;
- `--checks fpj` or `--checks mcnr` only when explicitly requested;
- `--overwrite` only after explicit approval.

The page-order argument uses `C,T,Z,A,P` as code symbols and means slowest-to-fastest stored page axes. For example, `ZAP` means Z outermost and phase fastest.

## Preserve The SIMcheck_V2 Contract

`run_simcheck.py` is only a model/acquisition adapter. It reshapes the input and selects the phase-count-aware policy, then calls the legacy functions in `simcheck_core.py`. Do not duplicate MCNR, FPJ, report, or batch-summary calculations in the wrapper.

For a direct TIFF, the output root contains:

```text
SIMcheck_V2_batch_MCNR.csv
SIMcheck_V2_batch_MCNR.md
SIMcheck_V2_report.md
SIMcheck_V2_stats.csv
raw/
  FPJ_C1.tif
  FPJ_C1-preview.png
  MCNR_C1_per_angle_compare_z*.png
  MCNR_C1_z*_angles_stack.tif
```

For folder input, each source gets one run subdirectory with the same per-run files, while identical batch MCNR documents are written both at the output root and into every run directory.

Do not add manifests, phase-mean outputs, new summary filenames, or separate per-direction FPJ/MCNR PNGs to this wrapper. Phase reduction is a separate preprocessing workflow. Never compute MCNR or FPJ from a phase-mean TIFF.

## Interpret And Validate

Read [references/metrics.md](references/metrics.md) before interpreting MCNR, comparing different phase counts, or reporting FPJ peaks.

After execution, verify:

1. FPJ has `directions + 1` pages per channel;
2. MCNR contains one score per direction plus aggregate statistics when enabled;
3. all quantitative outputs are finite;
4. the fixed files above exist and no alternative output protocol was created.

A skipped metric is not a zero score and must not be reported as a pass or failure.

Report per-direction MCNR, mean, angle-averaged-map score, weakest/strongest direction, and imbalance ratio separately. Report threshold verdicts, Wiener estimates, and overall risk only for the calibrated five-phase `3d-sim` mode. Do not collapse these into one unlabeled “overall” number.

## Bundled Scripts

- `scripts/run_simcheck.py`: thin file/folder adapter for model selection, acquisition validation, axis mapping, and legacy-core calls.
- `scripts/simcheck_core.py`: the sole MCNR/FPJ/report/batch-summary implementation.
- `scripts/fpj_preview.py`: FPJ PNG preview.
- `scripts/preview_for_chat.py`: FPJ and lihe chat previews.
- `scripts/push_p_highlight_to_tencent_docs.py`: optional, explicitly authorized Tencent Docs export; configure roots through `MP_SIM_PROJECT_ROOT`, `MP_SIM_BATCH_ROOT`, and `MP_SIM_PREVIEW_ROOT`.
- `scripts/test_simcheck_workflow.py`: deterministic synthetic regression tests.

Do not use a fixed D1/D2/D3 export schema when direction count differs from three. Treat external document upload as a separate, explicitly authorized integration and require a destination schema that supports the actual direction count.
