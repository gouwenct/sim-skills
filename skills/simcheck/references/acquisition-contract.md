# Acquisition Contract

## Canonical Tensor

Represent a phase-resolved stack internally as:

```text
I[c,t,z,a,p,y,x]
```

The letters are permitted here because this is a dimension signature:

- `c`: channel index;
- `t`: timepoint index;
- `z`: axial index;
- `a`: acquisition-direction index;
- `p`: phase index;
- `y,x`: spatial indices.

In prose, write 相位 or phase rather than standalone `P`.

## Frame Count

For a flattened TIFF stack:

$$
N_{\mathrm{pages}} = N_c N_t N_z N_a N_{\mathrm{phase}}.
$$

Page count validates a supplied contract but does not identify one. For example, 168 pages do not prove `8 Z x 3 directions x 7 phases`.

## Page Order

`--page-order` lists stored non-spatial axes from slowest to fastest. The fastest axis changes on consecutive TIFF pages.

Examples:

| Argument | Stored nesting |
| --- | --- |
| `ZAP` | Z outermost, direction next, phase fastest |
| `AZP` | direction outermost, Z next, phase fastest |
| `TAZPC` | timepoint outermost, channel fastest |

Use only `C,T,Z,A,P`. Every non-singleton axis must occur exactly once. A singleton axis may be omitted.

Do not translate a legacy name such as `CPZAT` by appearance alone. Confirm its actual loop nesting, then express that nesting using the slowest-to-fastest rule.

## Phase Reduction

Phase reduction is a separate preprocessing workflow and is not performed by the thin SIMcheck wrapper. The mapping remains:

For pixelwise mean reduction:

$$
M[c,t,z,a,y,x] =
\frac{1}{N_{\mathrm{phase}}}
\sum_{p=0}^{N_{\mathrm{phase}}-1}
I[c,t,z,a,p,y,x].
$$

The output page count is:

$$
N_{\mathrm{out}} = N_c N_t N_z N_a.
$$

Examples:

| Z | Directions | Phases | Input pages | Output pages |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 3 | 5 | 120 | 24 |
| 8 | 3 | 7 | 168 | 24 |
| 8 | 3 | 13 | 312 | 24 |
| 8 | 4 | 7 | 224 | 32 |

The default output order omits singleton axes and otherwise follows `C,T,Z,A`. With one channel and one timepoint, this becomes `ZA`.

## Multiple Timepoints And Channels

Phase reduction can serialize all channels and timepoints into one labeled TIFF.

The legacy `SIMcheck_V2` layout accepts one timepoint per input TIFF. Split multiple timepoints before invoking the wrapper. The bundled core loops over all channels within that timepoint. Do not combine different timepoints into one phase FFT.

## Folder Inputs

- Default folder traversal is non-recursive.
- `--recursive` must be explicit.
- Use `--name-glob` to select intended timepoints or tracks.
- Precompute the input list before creating outputs.
- Exclude the output root from discovery.
- Reject non-empty destination directories unless overwrite was explicitly approved.

## Required Provenance

Record at least:

- channel, timepoint, Z, direction, and phase counts;
- page order and its slowest-to-fastest semantics;
- requested checks and compute device;
- elapsed time and output paths.

The wrapper prints the execution plan and records the acquisition/check policy in `SIMcheck_V2_report.md`; the batch documents record source, output path, and elapsed time.
