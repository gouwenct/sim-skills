---
title: Direction-Specific Review Contracts for 3D Line FWHM
date: 2026-07-31
category: conventions
module: 3D line FWHM analysis and review
problem_type: convention
component: tooling
severity: high
applies_when:
  - Measuring lateral and axial FWHM from ZYX line data
  - Reviewing per-layer candidate evidence for 3D line ROIs
  - Persisting manual decisions for multiple measurement directions
tags:
  - 3d-line-fwhm
  - peak-z-layer
  - direction-specific-review
  - manual-review
kb_status: candidate
---

# Direction-Specific Review Contracts for 3D Line FWHM

## Context

A `ZYX` line candidate can yield a lateral XY fit and an axial Z fit. Earlier review output treated a candidate as one item and used combined or projected evidence. That breaks traceability because the two fits use different pixels and can pass or fail independently.

## Guidance

### Match XY review to the fitted plane

The line pipeline selects `z_center` from the candidate Z profile and passes exactly `volume[z_center]` to the perpendicular-profile extractor. The reviewer must display that same layer when `lateral_plane_mode` is `single_peak_layer`; a three-layer or maximum projection is not equivalent evidence.

See [scripts/fwhm_lines.py](../../../scripts/fwhm_lines.py) and [scripts/fwhm_reviewer.py](../../../scripts/fwhm_reviewer.py).

### Keep direction artifacts independent

- `candidates-XY.tif` is a `ZYXS` stack whose one-pixel lateral normal and ROI ID occur only on `z_center`.
- `candidates-Z.tif` is a `ZYXS` stack whose axial marker spans the saved Z-profile interval and whose ROI ID occurs on `z_center`.
- The runtime output's `review-assets` directory stores the single-layer XY crop, normal and fit under `XY`.
- The same runtime output directory stores XY, XZ, YZ and axial-fit evidence under `Z`.
- Manual keys end in `-xy` or `-z`; reviewed inclusion is stored in `reviewed_included_xy` and `reviewed_included_z`.

An older whole-ROI deletion may migrate to both directions for compatibility, but new decisions remain direction-specific.

### Keep loading behavior out of measurement semantics

The reviewer may paginate or incrementally load assets. Those controls must not change candidate identity, order, evidence, saved decisions or reviewed statistics.

## Why This Matters

Review evidence must be the same evidence that produced the measurement. A projection includes pixels from layers that did not contribute to the lateral fit, so it is not measurement-matched evidence. Independent decisions also prevent a rejected lateral fit from silently deleting a valid axial fit.

## When to Apply

- Apply the single-layer rule whenever line data are `ZYX` and `lateral_plane_mode` is `single_peak_layer`.
- Apply direction-specific keys whenever one candidate produces independently accepted measurements.
- Preserve separate candidate stacks in automatic and per-file batch outputs, and preserve separate XY/Z asset directories in automatic and reviewed outputs.

## Examples

If ROI 12 passes both automatic checks but its lateral normal crosses a neighboring line, delete only the `-xy` item. The reviewed XY statistics exclude ROI 12 while the untouched `-z` item remains eligible for axial statistics.

## Related

- [ADR-001](../../decisions/ADR-001-direction-specific-line-review.md)
- [Method and output reference](../../../references/method.md)
- [Regression test](../../../test/test_fwhm_lines.py)
