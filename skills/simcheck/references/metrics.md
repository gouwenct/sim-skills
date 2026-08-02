# Metrics And Interpretation

## MCNR

Compute MCNR independently for every acquisition direction on the original phase-resolved data.

The bundled implementation applies an Anscombe transform, subtracts the sequence mean, performs a Fourier transform along the local Z-window/phase sequence, and uses:

$$
\mathrm{MCNR} =
\frac{\sqrt{\sum_{h \in H}|F_h|^2}}
{\sigma_{\mathrm{noise}}}.
$$

The harmonic set is $H=\{1\}$ for three phases, $H=\{1,2\}$ for five phases, and $H=\{1,2,3\}$ for seven phases. The feature score is measured on the middle-Z MCNR map after Otsu feature selection.

Noise normalization depends on the protocol:

- five-phase `3d-sim` preserves SIMcheck's highest-frequency amplitude standard deviation over XY;
- three- and seven-phase modes use the standard deviation of the combined harmonic amplitude in the lowest-intensity 20% of spatial pixels that vary across the local phase/Z sequence; constant external padding is excluded;
- the report metadata records the harmonic orders, noise method, and calibration state; uncalibrated three- and seven-phase stats also retain the per-direction noise estimate.

Report these separately:

- one feature MCNR per direction;
- mean of per-direction feature MCNR values;
- feature MCNR of the direction-averaged MCNR map;
- weakest and strongest directions;
- maximum/minimum direction ratio;
- estimated Wiener value;
- overall risk, defined as the worst per-direction risk.

The direction-averaged-map score is not the same quantity as the arithmetic mean of direction scores.

### Calibrated Five-Phase Risk Bands

| Feature MCNR | Project risk | SIMcheck-style verdict |
| ---: | --- | --- |
| `< 4` | HIGH | NO |
| `4 to <6` | MED | MAYBE |
| `6 to <8` | LOW | MAYBE |
| `>= 8` | OK | YES |

Do not apply this table, the SoftWoRx Wiener estimate, or an overall risk label to three- or seven-phase extended MCNR.

### Phase-Count Constraints

- Require uniformly spaced phases for the current FFT method.
- Three-phase 2D-SIM has no independent phase-frequency noise bin. Its first-harmonic/background-normalized result is an extended, unvalidated MCNR.
- Five-phase `3d-sim` uses the first and second harmonics and retains the traditional SIMcheck-style interpretation.
- Seven-phase `3d-mp-sim` uses the first three harmonics. Its background-normalized result is an extended, unvalidated MCNR.
- `4b-sim` remains unsupported because no calibrated 13-phase harmonic policy has been established.
- Do not apply ordinary FFT MCNR to nonuniform phase positions; use the actual phase angles with a fitted harmonic model or nonuniform transform.
- Different phase counts change sampling and noise aggregation. Treat cross-protocol MCNR comparisons as uncalibrated unless controlled data demonstrate comparability.

## FPJ

For each direction:

1. apply an edge window to every spatial plane;
2. zero-pad to the next power-of-two square;
3. compute the two-dimensional FFT;
4. compute `log1p(|FFT|^2)`;
5. maximum-project over Z and phase;
6. mask the central low-frequency disk;
7. optionally detect the number of radial order shells allowed by the imaging mode.

For `A` directions, write `A + 1` FPJ pages:

- page 0: maximum across all directions;
- pages 1 through `A`: one page per data direction.

Peak policy:

- `2d-sim`: label only one symmetric first-order candidate pair per direction;
- `3d-sim`: label first- and second-order candidate pairs;
- `3d-mp-sim` and `4b-sim`: do not assign automatic order labels without a calibrated modality-specific peak model;
- `generic`: retain the legacy two-shell heuristic for backward compatibility.

Always report:

- number of phase-resolved planes used;
- padded FPJ size;
- number of labeled peaks.

Direction labels follow acquisition order, not geometric sorting in Fourier space.
Automatic peaks are candidates, not pass/fail results. FPJ remains primarily a visual frequency-domain check.

## Validation

- FPJ shape must be `(directions + 1, side, side)`.
- FPJ and MCNR numerical outputs must be finite.
- MCNR must contain exactly one direction score for each acquisition direction.
- A large maximum/minimum direction ratio indicates directional imbalance but is not itself a pass/fail threshold.
- Sparse samples can lack visually dominant FPJ spots even when other quality measures are acceptable.
