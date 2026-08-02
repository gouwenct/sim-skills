---
name: mp-data-ingest
description: Download MP-SIM raw data from PKU shares, Baidu shares, direct URLs, or local paths, then normalize sample numbering and move cam2 TIFF files beside cam1. Use for the acquisition and raw-layout stages before MP preprocessing.
---

# MP Data Ingest

Keep acquisition separate from image processing. Confirm the source, destination,
sample name, extraction code, and whether filesystem normalization is allowed.

## Download

Run `scripts/download_netdisk.py` for PKU, Baidu, direct HTTP, or local sources:

```powershell
& <python> -B <SKILL_DIR>/scripts/download_netdisk.py `
  --url <url-or-local-path> --dest <date-directory> --name <sample-name>
```

Add `--password <code>` for protected shares and `--extract` for archives.
Baidu downloads require an authenticated `BaiduPCS-Go` executable on `PATH`.

## Normalize Raw Layout

Preview first, then execute only after confirming the planned renames and moves:

```powershell
& <python> -B <SKILL_DIR>/scripts/move_cam2_tiffs.py <raw-batch>
& <python> -B <SKILL_DIR>/scripts/move_cam2_tiffs.py <raw-batch> --execute
```

Normal samples receive natural-order `NN、` prefixes; `peizhun*` uses `99、`.
Never overwrite a conflicting target.
