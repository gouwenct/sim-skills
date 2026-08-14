---
name: fwhm
description: Offline, auditable FWHM measurement for fluorescent bead TIFF stacks and thin biological line structures. Use when Codex needs to calculate lateral or axial resolution from beads or lines, automate ROI/profile selection, fit Gaussian widths, batch TIFF files, generate verification plots, or open the local ROI reviewer.
---

# FWHM

Use the bundled deterministic scripts. The measurement itself is fully offline and does not require AI.

## Choose the mode

- Use `beads` only for isolated fluorescent beads or point-like emitters.
- Use `lines` only for sufficiently thin, bright biological line structures.
- Ask the user which mode applies when the sample type is not explicit. Never infer it from image content.

## Collect required inputs

Always obtain:

- A TIFF file or a folder containing TIFF files.
- XY sampling in nm/pixel and Z sampling in nm/layer.

For `lines`, also require:

- Theoretical lateral resolution in nm.
- Theoretical axial resolution in nm.

For consecutive data under the same parent folder, allow reuse of the latest confirmed sampling values. Use `--reuse-last` only when the user explicitly identifies the data as the same or a consecutive batch.

Use `--preset mp-sim-beads-v1` only for confirmed MP-SIM beads. It supplies XY 30.1 nm, Z 96.9 nm, and excludes the final two Z layers. Generic analysis excludes the first and last Z layer.

## Run the analysis

Use the environment launcher so it can prefer a compatible `3DMPSIM*` environment and fall back to the repository's shared `.venv`. If neither exists, ask the user before creating the UV environment:

```powershell
python "<skill-root>\scripts\run_fwhm.py" beads "<input>" `
  --pixel-size-xy-nm 30.1 --pixel-size-z-nm 96.9

python "<skill-root>\scripts\run_fwhm.py" lines "<input>" `
  --pixel-size-xy-nm 30.1 --pixel-size-z-nm 96.9 `
  --theoretical-lateral-resolution-nm 100 `
  --theoretical-axial-resolution-nm 290
```

对于 beads，如果只需要保留同一批接近箱图中位数的 ROI，可增加：

```powershell
python "<skill-root>\scripts\run_fwhm.py" beads "<input>" `
  --pixel-size-xy-nm 30.1 --pixel-size-z-nm 96.9 `
  --median-nearby-count 5
```

`--median-nearby-count N` 默认是 `0`，保留旧行为；设为正数时，在 Tukey 排除之后分别计算横向和轴向箱图中位数，只从同时具备两个方向合格值的同一 ROI 中联合排序，并保留前 `N` 个。输出 CSV、候选图、箱图和 `review-assets/` 只包含这批 ROI；没有轴向合格值的 2D ROI 不会被选入。

Pass `--recursive` only when the user wants subfolders included. For `CZYX`, `TZYX`, or `TCZYX`, require `--channel` and/or `--timepoint`; never silently select a non-singleton channel or time point. Missing or generic three-dimensional axes default to the confirmed `ZYX` convention.

Do not choose an existing output directory. The default is:

```text
<input-data-parent>\FWHM-results\<input-name>-<timestamp>\
```

## Interpret the result

- Read `SUMMARY.md` first. It contains the short result summary and an `Index` of every output.
- Report the number of accepted ROIs, median lateral/axial FWHM, IQR, and bootstrap 95% CI when present.
- Treat `status="too_many_candidates"` as `候选太多`; diagnostics are usable, but the result is not formal.
- Treat 10–29 accepted ROIs as a warning and fewer than 10 as `样本量不足`.
- Apply the standard Tukey upper outer fence only when at least 10 quality-accepted ROIs exist: `Q3 + 3×IQR`. In the default mode, keep excluded ROI data and images; with `--median-nearby-count`, outputs contain only the shared retained ROI set.
- Do not claim line FWHM is system resolution unless the biological line diameter is negligible. It is otherwise observed structure width convolved with the PSF.

Read [references/method.md](references/method.md) when explaining formulas, screening rules, output fields, or scientific limitations.

## Review ROIs

Open an existing result in the local reviewer:

```powershell
python "<skill-root>\scripts\run_fwhm.py" review "<result-directory>"
```

The reviewer binds only to `127.0.0.1`. Automatically accepted ROIs are kept by default; the user only deletes obvious failures. Decisions are saved immediately to `roi-decisions.json`. Generating reviewed results creates a new `reviewed/<timestamp>/` directory and never overwrites automatic results.

Analysis pre-generates `review-assets/`, so opening and switching ROIs in the reviewer does not render images on demand. `candidates.png` uses the same candidate IDs shown in the reviewer.

For line data, the reviewer separates lateral and axial items. Lateral fitting and review use exactly the candidate peak-Z layer. Inspect per-layer annotations in `candidates-XY.tif` and `candidates-Z.tif`; their axes are `ZYXS`. Direction-specific evidence is stored in `review-assets/XY/` and `review-assets/Z/`, and XY/Z decisions are independent. The reviewer shows three items per row, caps a page at 150 items, initially loads 30, and loads 30 more near the page bottom.

Do not manually re-include automatically rejected ROIs. Manual ROI addition is provisional: use the local subpixel refinement only when available and require the same automatic quality checks before inclusion.
