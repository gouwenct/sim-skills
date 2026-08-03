---
name: decorr
description: Run the versioned Python decorrelation-resolution workflow for reconstructed microscopy TIFF data. Use when Codex needs to measure lateral or axial decorrelation resolution, select an XY plane automatically, batch final MP-SIM reconstructions, inspect axial XZ/YZ montage groups, export Excel results, or render decorrelation QC figures. Keep this workflow separate from FWHM and do not use MATLAB for new measurements.
---

# Decorrelation Resolution

Use the bundled Python scripts. Treat reconstructed TIFF input as read-only and write every run to a separate output directory.

## Preflight

Confirm before measuring:

1. Input is one final reconstructed TIFF or the formal batch root.
2. A 2D TIFF is `YX`; a 3D TIFF is `ZYX`.
3. XY sampling is provided in nm/pixel.
4. Z sampling is provided in nm/layer for `ZYX`.
5. Output does not overlap the input tree.
6. Plane selection is Tenengrad unless a documented manual override is required.
7. The main axial montage seed is `20250124`.
8. XY dimensions are at least 262 by 262 pixels for the fixed 4 by 4 grid.

Never infer sampling from a filename. Keep physical parameters separate from input, output, seed, and rendering options.

## Choose the output

Do not choose an existing output directory for an ordinary `measure` run.
When the user does not specify an output location, use:

```text
<input-data-folder-parent>\decorr-results\<input-stem>-<timestamp>\
```

This keeps results beside the input data folder rather than inside it. Never
derive the output from the current working directory, and never write external
data to a repository's `code/output` merely because that repository is active.
Report the resolved input and output paths before measurement starts.

For a non-formal folder batch, create one new batch root:

```text
<input-folder-parent>\decorr-results\<input-folder-name>-<timestamp>\
```

Place each measurement below its `samples/` directory so the root can be
rendered as one batch. Use an output elsewhere only when the user explicitly
requests it or an isolated writable location is required.

The formal 27-sample runner is the only default exception: it keeps its stable
input-adjacent path so complete samples can be reused after interruption. Do
not use that resumable directory for a run with changed physical parameters,
seeds, or output schema; pass a new `--output-root` instead.

## Run

Use the skill launcher:

```powershell
& "<skill-root>\decorr.cmd" measure "<input.tif>" `
  --xy-pixel-nm <xy-nm> `
  --z-pixel-nm <z-nm> `
  --axial-device cuda `
  --seed 20250124
```

Pass `--output "<new-output-dir>"` only for an explicit isolated override. The
directory must not already exist.

For the formal 27-sample MP-SIM batch:

```powershell
& "<skill-root>\decorr.cmd" batch `
  --input-root "<formal-input-root>" `
  --axial-device cuda
```

Use the formal runner only when the input matches:

`*/n-3.0-01-488_padding/留负MP-新建文件夹 (10)-mear1去相关*.tif`

It requires exactly 27 TIFFs and uses XY `30.0 nm`, Z `96.9 nm`, Tenengrad selection, and seed `20250124`.
By default it writes beside the input folder at
`<input-parent>/decorr-results/<input-folder-name>/`. Pass `--output-root`
only for an explicit isolated override.

Axial measurement defaults to `--axial-device cpu`. Use `cuda` only when the
installed PyTorch runtime has CUDA support. Use `auto` when CPU fallback is
preferred over failing the run if CUDA is unavailable. Lateral measurement,
montage construction, TIFF I/O, and workbook generation remain on CPU.

Render existing results with:

```powershell
& "<skill-root>\decorr.cmd" render "<batch-output-root>"
```

The Python measurement persists native diagnostic curves, so rendering produces
both lateral and axial paper figures without requiring historical MATLAB files.
Batch measurement writes `decorr_results.xlsx`. Rendering writes `INDEX.md`,
the combined batch summary, root-level fixed-size lateral 5 by 2 and axial
3 by 2 QC pages, individual QC figures in `lateral_qc_pages/` and
`axial_qc_pages/`, and paper figures in `lateral_decorrelation_paper/` and
`axial_decorrelation_paper/`. For a batch, rendering also writes
`axial-peak-qc/`, appends `axial_peak_qc.csv` as the `轴向峰值QC` workbook
sheet, writes flat per-sample PNG files directly in `axial-peak-qc/samples/`
for ImageJ import, and adds raw solid and 6 dB-filtered dashed axial results
to separate lower panels in `batch_decorrelation_summary.png`. Its upper
panels show lateral resolution and the lateral/filtered-axial/raw-axial box
plots. The same render command accepts either a batch root or one direct
measurement output.

Use one batch-wide color range per comparable metric. Keep lateral and axial
resolution ranges separate, but use the same `viridis_r` direction so colors
have the same better-to-worse meaning. Fix spectral-margin heatmaps at
0 to 15 dB and curve-support heatmaps at 0 to 10.

Additional axial audit seeds are disabled by default. Pass explicit seeds only
for a requested stability audit, for example
`--audit-seeds 20250125 20250126 20250127 20250128`.

## Interpret

- `YX`: report lateral resolution only.
- `ZYX`: report lateral and axial results.
- Do not issue a formal axial result for fewer than eight native Z planes.
- Exclude local tiles whose robust contrast is below 5% of the selected plane's
  robust contrast before calculating the lateral sample median. Render them as
  `N/A`.
- Use sample-level medians for batch statistics; do not pool all local blocks as independent samples.
- Treat decorrelation as effective image resolution, not bead calibration or proof of biological fidelity.
- Treat FFT support as diagnostic evidence, not a resolution value by itself.
- Retain raw axial values. In batch summaries, filter the dashed axial series
  at spectral margin `>= 6 dB`, equivalent to selected-band power at least
  3.98 times the outer-frequency noise-floor power. Label this as a fixed
  workflow QC gate, not an original decorrelation-paper threshold.
- Do not clip negative reconstructed values before measurement.
- Do not tune thresholds after seeing experimental-group results.

Read [references/method.md](references/method.md) when explaining the montage, peak selection, output schema, known baseline limitations, or scientific boundaries.

## Validate

Run:

```powershell
& '<resolved-sim-skills-python>' -m pytest `
  "<skill-root>\tests" -q
```

Also run the skill validator after changing `SKILL.md` or `agents/openai.yaml`.

For each successful TIFF, verify `summary.json` where applicable, exactly one
selected plane for `ZYX`, 16 lateral grid rows with excluded rows left blank, 12
axial groups for eligible `ZYX`, float32 TIFF output, and
`resolution_nm = 2 * pixel_size_nm / kc`. For a completed batch also verify
`INDEX.md`, the six post-render workbook sheets, `axial-peak-qc/`, both QC page
directories, the root-level contact pages, and the four centralized
per-sample figure sets for eligible `ZYX`.
