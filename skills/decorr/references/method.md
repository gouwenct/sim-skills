# Decorrelation Method Reference

## Contents

- Measurement roles
- Input and sampling
- Lateral workflow
- Axial workflow
- Axial spectral-support QC
- Resolution conversion
- Baseline limitations
- Output checks

## Measurement roles

Use decorrelation for rapid effective-resolution screening of reconstructed images. Use FWHM separately for bead or sufficiently thin line-profile measurements. Neither metric proves biological fidelity.

## Input and sampling

Read TIFF data with `tifffile` as float32 and preserve negative reconstructed values. Interpret two-dimensional input as `YX` and confirmed three-dimensional input as `ZYX`. Require explicit XY sampling and, for `ZYX`, explicit Z sampling.

The formal MP-SIM batch uses:

- XY: 30.0 nm/pixel
- Z: 96.9 nm/layer
- Main axial seed: 20250124

## Lateral workflow

Select the XY layer with Tenengrad gradient energy. Record all plane scores and the top candidates so close or low-SNR selections can be reviewed.

Use a 4 by 4 grid of local regions. Before decorrelation, retain only tiles
whose 0.5% to 99.5% robust contrast is at least 5% of the selected plane's
robust contrast. Store excluded tile resolutions as blank and render them as
`N/A`. A batch summary uses one valid-tile median per sample rather than
treating local regions as independent samples.

## Axial workflow

Construct XZ and YZ montage groups without native-Z interpolation:

- Copy the first and last Z planes four times.
- Build 12 groups.
- Combine 32 strips per group.
- Use seed 20250124 for the main result.

Random seeds control montage strip offsets; they do not change the reconstructed
TIFF. Additional seeds are disabled by default. When a stability audit is
explicitly requested, keep its fixed seeds separate from the main statistical
population.

The main seed retains all sector-refinement diagnostics. Audit seeds retain the
same all-sector initial search used to establish the filter range, then refine
only the reported middle sector. The selected-sector calculation order and
thresholds are unchanged.

CPU remains the default reference backend. The optional CUDA backend preserves
the stable radial ordering and float64 accumulations, but FFT and reduction
rounding can move an intermediate quantized curve value by 0.001. Validate
selected `kc`, validity flags, and sample medians against CPU before using CUDA
for a new formal dataset.

CUDA caches fixed-shape frequency geometry and processes up to 16 correlation
curves per batch. The batch size is intentionally capped below the fastest
measured setting to retain VRAM headroom for shared reconstruction workloads.

Do not issue a formal axial conclusion when native Z has fewer than eight layers.

## Axial spectral-support QC

Preserve every raw axial `kc` and resolution. For the batch comparison, also
calculate the spectral margin at the selected cutoff:

$$
M_{\mathrm{dB}} =
10\log_{10}\left(\frac{P(k_c)}{P_{\mathrm{noise}}}\right)
$$

Use the median power within `kc +/- 0.02` for $P(k_c)$ and the median power at
normalized axial frequencies `>= 0.9` for $P_{\mathrm{noise}}$. The fixed
workflow gate is:

$$
M_{\mathrm{dB}} \ge 6
\quad\Longleftrightarrow\quad
\frac{P(k_c)}{P_{\mathrm{noise}}} \ge 10^{6/10} \approx 3.98
$$

Show raw sample medians with solid lines and the 6 dB-filtered medians with
dashed lines and a dashed box plot. This power-ratio definition is the
theoretical basis for the gate. The value `6 dB` is a fixed workflow QC choice,
not a universal threshold from the parameter-free decorrelation paper. Do not
delete or overwrite raw values.

## Resolution conversion

Convert normalized cutoff frequency using:

$$
\mathrm{resolution}_{nm} = \frac{2\,\mathrm{pixel\ size}_{nm}}{k_c}
$$

A smaller resolution value means the selected cutoff frequency is larger. This alone does not prove that the selected peak is physically valid.

## Baseline limitations

The committed baseline preserves the existing peak selector. Its axial sector
refinement selects the candidate maximizing the square root of `kc` times SNR
and can switch between the observed modes near 230 nm and 270 nm. The 6 dB
render-stage gate compares spectral support without hardcoding a minimum
resolution or maximum `kc`; it does not change the raw selector.

The Python pipeline stores compressed lateral and axial diagnostic curves.
Render paper-style figures from these diagnostics; use historical
`axial_result.mat` only as a compatibility fallback.

The formal output root contains `INDEX.md`, `decorr_results.xlsx`, one combined
batch summary, root-level lateral 5 by 2 and axial 3 by 2 contact pages,
centralized individual QC and paper-figure directories, and
`axial-peak-qc/`. After rendering, the workbook sheets are `样本汇总`,
`横向局部块`, `轴向分组`, `轴向种子稳定性`, `运行信息`, and `轴向峰值QC`.
Use one batch-wide color range for every set of comparable sample heatmaps;
record the ranges in `qc_run_info.json`.

## Output checks

Check:

- Input and output paths.
- Physical sampling.
- Selected XY layer and plane scores.
- Lateral grid row and column.
- Valid axial XZ/YZ group counts.
- Sample-level lateral and axial medians.
- Raw and 6 dB-filtered axial medians and valid-group counts.
- Batch-global color ranges and the `轴向峰值QC` workbook sheet.
- Seed and montage sampling metadata.
- Workbook and figure paths.
- Failures, warnings, and skipped paper figures.
