#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Push 新建文件夹(10) P-highlight records to Tencent Docs via official MCP HTTP API."""

from __future__ import annotations

import base64
import csv
import json
import os
import re
import ssl
import time
import urllib.request
from pathlib import Path

FILE_ID = "DUFhCd3ZUTk9qeGFT"
SHEET_ID = "BB08J2"  # tab=BB08J2 工作表1
DOC_URL = f"https://docs.qq.com/sheet/{FILE_ID}?tab={SHEET_ID}"
ROOT = Path(os.environ.get("MP_SIM_PROJECT_ROOT") or Path.cwd())
BATCH = Path(os.environ.get("MP_SIM_BATCH_ROOT") or ROOT / "dataOut")
PREV = Path(os.environ.get("MP_SIM_PREVIEW_ROOT") or ROOT / "previews")
HEADERS = [
    "数据标题",
    "批次",
    "D1",
    "D1风险",
    "D2",
    "D2风险",
    "D3",
    "D3风险",
    "mean",
    "总判",
    "MCNR表格",
    "FPJ图",
    "lihe图",
    "备注",
]
COL = {
    "title": 0,
    "batch": 1,
    "D1": 2,
    "D1_risk": 3,
    "D2": 4,
    "D2_risk": 5,
    "D3": 6,
    "D3_risk": 7,
    "mean": 8,
    "overall": 9,
    "mcnr_text": 10,
    "fpj": 11,
    "lihe": 12,
    "note": 13,
}


def batch_data_label(batch: str, title: str) -> str:
    """腾讯文档「批次」列：批次名/数据名，如 新建文件夹 (10)/16、38。"""
    b = (batch or "").strip().rstrip("/\\")
    t = (title or "").strip().lstrip("/\\")
    if not b:
        return t
    if not t:
        return b
    return f"{b}/{t}"

MCP_URL = "https://docs.qq.com/openapi/mcp"
CTX = ssl._create_unverified_context()
_id = 0


def _token() -> str:
    cfg = Path.home() / ".grok" / "config.toml"
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r'Authorization\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError("tencent-docs Authorization not found in ~/.grok/config.toml")
    return m.group(1)


TOKEN = _token()


def _post(payload: dict, retries: int = 5) -> dict | str:
    headers = {
        "Authorization": TOKEN,
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    data = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            MCP_URL,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180, context=CTX) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            obj = None
            for line in body.splitlines():
                if line.startswith("data:"):
                    try:
                        obj = json.loads(line[5:].strip())
                    except Exception:
                        pass
            if obj is None:
                try:
                    obj = json.loads(body)
                except Exception:
                    return body
            return obj
        except Exception as exc:  # SSL / network flakes
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"MCP POST failed after retries: {last_err}")


def mcp_init() -> None:
    _post(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "p-highlight-export", "version": "1.0"},
            },
        }
    )
    try:
        _post({"jsonrpc": "2.0", "method": "notifications/initialized"})
    except Exception:
        pass


def call_tool(name: str, arguments: dict, retries: int = 4):
    global _id
    last = None
    for _ in range(retries):
        _id += 1
        obj = _post(
            {
                "jsonrpc": "2.0",
                "id": _id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        if not isinstance(obj, dict):
            last = obj
            time.sleep(1.2)
            continue
        if obj.get("error"):
            last = obj["error"]
            time.sleep(1.2)
            continue
        result = obj.get("result") or obj
        if isinstance(result, dict) and result.get("isError"):
            last = result
            time.sleep(1.2)
            continue
        if isinstance(result, dict) and "content" in result:
            texts = []
            for c in result.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "text":
                    texts.append(c.get("text") or "")
            joined = "\n".join(texts)
            try:
                return json.loads(joined)
            except Exception:
                return {"raw": joined, "result": result}
        return result
    return {"error": last}


def parse_mcnr(sample_dir: Path) -> dict | None:
    csv_path = sample_dir / "Dr" / "SIMcheck_V2" / "SIMcheck_V2_batch_MCNR.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    name = sample_dir.name
    cand = [
        r
        for r in rows
        if r.get("sample") == name and "regOnly" not in (r.get("track") or "")
    ]
    if not cand:
        cand = [r for r in rows if r.get("sample") == name]
    if not cand:
        return None
    r = cand[-1]

    def f(k: str) -> float:
        try:
            return float(r.get(k) or "nan")
        except Exception:
            return float("nan")

    return {
        "sample": name,
        "D1": f("D1"),
        "D1_risk": r.get("D1_risk", ""),
        "D2": f("D2"),
        "D2_risk": r.get("D2_risk", ""),
        "D3": f("D3"),
        "D3_risk": r.get("D3_risk", ""),
        "mean": f("mean"),
        "overall": r.get("overall_risk", ""),
        "track": r.get("track", ""),
    }


def collect_records() -> list[dict]:
    samples = sorted(
        [
            p
            for p in BATCH.iterdir()
            if p.is_dir()
            and not p.name.startswith("99")
            and "peizhun" not in p.name.lower()
        ],
        key=lambda p: p.name,
    )
    out = []
    for sd in samples:
        name = sd.name
        m = parse_mcnr(sd)
        fpj = PREV / f"{name}__FPJ_C1.png"
        lihe = PREV / f"{name}__lihe_maxproj.png"
        out.append(
            {
                "title": name,
                "mcnr": m,
                "fpj": fpj if fpj.exists() else None,
                "lihe": lihe if lihe.exists() else None,
            }
        )
    return out


def cell_string(row: int, col: int, text: str) -> dict:
    return {
        "row": row,
        "col": col,
        "value_type": "STRING",
        "string_value": text if text is not None else "",
    }


def cell_number(row: int, col: int, num: float) -> dict:
    return {
        "row": row,
        "col": col,
        "value_type": "NUMBER",
        "number_value": float(num),
    }


def write_text(records: list[dict]) -> None:
    values: list[dict] = []
    for col, h in enumerate(HEADERS):
        values.append(cell_string(0, col, h))

    for i, rec in enumerate(records):
        row = i + 1
        m = rec["mcnr"] or {}
        if m:
            mcnr_text = (
                f"| 样本 | D1 | 风险 | D2 | 风险 | D3 | 风险 | mean | 总判 |\n"
                f"| {rec['title']} | {m['D1']:.3f} | {m['D1_risk']} | "
                f"{m['D2']:.3f} | {m['D2_risk']} | {m['D3']:.3f} | {m['D3_risk']} | "
                f"{m['mean']:.3f} | {m['overall']} |"
            )
            note = f"track={m.get('track', '')}; A轨 01-488_padding t0"
        else:
            mcnr_text = ""
            note = "无 MCNR CSV"

        def put(col: int, v):
            if isinstance(v, (int, float)) and v == v:
                values.append(cell_number(row, col, float(v)))
            else:
                if isinstance(v, float) and v != v:
                    sv = ""
                else:
                    sv = "" if v is None else str(v)
                values.append(cell_string(row, col, sv))

        put(0, rec["title"])
        put(1, batch_data_label(rec.get("batch") or "新建文件夹 (10)", rec["title"]))
        put(2, m.get("D1"))
        put(3, m.get("D1_risk", ""))
        put(4, m.get("D2"))
        put(5, m.get("D2_risk", ""))
        put(6, m.get("D3"))
        put(7, m.get("D3_risk", ""))
        put(8, m.get("mean"))
        put(9, m.get("overall", ""))
        put(10, mcnr_text)
        put(11, "")  # image
        put(12, "")  # image
        put(13, note)

    print(f"writing {len(values)} cells ...")
    chunk = 60
    for i in range(0, len(values), chunk):
        part = values[i : i + chunk]
        res = call_tool(
            "sheet.set_range_value",
            {"file_id": FILE_ID, "sheet_id": SHEET_ID, "values": part},
        )
        err = res.get("error") if isinstance(res, dict) else None
        print(f"  cells {i}-{i+len(part)} => {('OK' if not err else err)}")
        time.sleep(0.25)


def _shrink_png(png: Path, max_side: int = 720, quality_like: int = 85) -> bytes:
    """Return JPEG bytes (smaller than full PNG) for more reliable upload."""
    from io import BytesIO

    from PIL import Image as PILImage

    im = PILImage.open(png).convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), PILImage.Resampling.LANCZOS)
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality_like, optimize=True)
    return buf.getvalue()


def insert_image(png: Path | None, row: int, col: int, label: str) -> None:
    if not png or not png.exists():
        print(f"  skip image {label}")
        return
    raw = _shrink_png(png)
    b64 = base64.b64encode(raw).decode("ascii")
    size_mb = len(raw) / (1024 * 1024)
    fname = png.stem + ".jpg"
    print(f"  image {label} row={row} col={col} {size_mb:.2f}MB")
    up = call_tool(
        "upload_image",
        {"file_name": fname, "image_base64": b64},
    )
    image_id = None
    if isinstance(up, dict):
        image_id = (
            up.get("image_id")
            or up.get("id")
            or (up.get("data") or {}).get("image_id")
        )
        if not image_id and isinstance(up.get("raw"), str):
            try:
                image_id = json.loads(up["raw"]).get("image_id")
            except Exception:
                pass
    if image_id:
        res = call_tool(
            "sheet.insert_image",
            {
                "file_id": FILE_ID,
                "sheet_id": SHEET_ID,
                "row_index": row,
                "col_index": col,
                "image_id": image_id,
            },
        )
        print(f"    insert_id ok => {str(res)[:120]}")
        return

    res = call_tool(
        "sheet.insert_image",
        {
            "file_id": FILE_ID,
            "sheet_id": SHEET_ID,
            "row_index": row,
            "col_index": col,
            "content": b64,
        },
    )
    print(f"    insert_content => {str(res)[:120]}")


def ensure_header() -> None:
    """Write header row if missing."""
    data = call_tool(
        "sheet.get_cell_data",
        {
            "file_id": FILE_ID,
            "sheet_id": SHEET_ID,
            "start_row": 0,
            "end_row": 0,
            "start_col": 0,
            "end_col": 0,
            "return_csv": True,
        },
    )
    csv_data = ""
    if isinstance(data, dict):
        csv_data = data.get("csv_data") or ""
    first = (csv_data.splitlines()[0] if csv_data else "").strip()
    if first.startswith("数据标题"):
        return
    values = [cell_string(0, c, h) for c, h in enumerate(HEADERS)]
    call_tool(
        "sheet.set_range_value",
        {"file_id": FILE_ID, "sheet_id": SHEET_ID, "values": values},
    )


def list_title_rows(max_row: int = 200) -> dict[str, int]:
    """Map 数据标题 -> row index (0-based). Header is row 0."""
    data = call_tool(
        "sheet.get_cell_data",
        {
            "file_id": FILE_ID,
            "sheet_id": SHEET_ID,
            "start_row": 0,
            "end_row": max_row,
            "start_col": 0,
            "end_col": 0,
            "return_csv": True,
        },
    )
    csv_data = ""
    if isinstance(data, dict):
        csv_data = data.get("csv_data") or ""
    mapping: dict[str, int] = {}
    lines = csv_data.splitlines()
    for row_idx, ln in enumerate(lines):
        title = ln.split(",", 1)[0].strip().strip('"')
        if not title or title == "数据标题":
            continue
        mapping[title] = row_idx
    return mapping


def find_or_append_row(title: str) -> int:
    """Return 0-based row for sample title; append empty row if new."""
    ensure_header()
    mapping = list_title_rows()
    if title in mapping:
        return mapping[title]
    # append after max existing data row
    next_row = (max(mapping.values()) + 1) if mapping else 1
    call_tool(
        "sheet.set_range_value",
        {
            "file_id": FILE_ID,
            "sheet_id": SHEET_ID,
            "values": [
                cell_string(next_row, COL["title"], title),
                cell_string(
                    next_row,
                    COL["batch"],
                    batch_data_label("新建文件夹 (10)", title),
                ),
            ],
        },
    )
    return next_row


def upsert_mcnr(
    title: str,
    mcnr: dict,
    batch: str = "新建文件夹 (10)",
    *,
    init: bool = True,
) -> int:
    """Immediately write MCNR fields for one sample. Returns row index."""
    if init:
        mcp_init()
    row = find_or_append_row(title)
    batch_cell = batch_data_label(batch, title)
    mcnr_text = (
        f"| 样本 | D1 | 风险 | D2 | 风险 | D3 | 风险 | mean | 总判 |\n"
        f"| {title} | {mcnr['D1']:.3f} | {mcnr.get('D1_risk','')} | "
        f"{mcnr['D2']:.3f} | {mcnr.get('D2_risk','')} | "
        f"{mcnr['D3']:.3f} | {mcnr.get('D3_risk','')} | "
        f"{mcnr['mean']:.3f} | {mcnr.get('overall') or mcnr.get('overall_risk','')} |"
    )
    values = [
        cell_string(row, COL["title"], title),
        cell_string(row, COL["batch"], batch_cell),
        cell_number(row, COL["D1"], float(mcnr["D1"])),
        cell_string(row, COL["D1_risk"], str(mcnr.get("D1_risk", ""))),
        cell_number(row, COL["D2"], float(mcnr["D2"])),
        cell_string(row, COL["D2_risk"], str(mcnr.get("D2_risk", ""))),
        cell_number(row, COL["D3"], float(mcnr["D3"])),
        cell_string(row, COL["D3_risk"], str(mcnr.get("D3_risk", ""))),
        cell_number(row, COL["mean"], float(mcnr["mean"])),
        cell_string(
            row,
            COL["overall"],
            str(mcnr.get("overall") or mcnr.get("overall_risk") or ""),
        ),
        cell_string(row, COL["mcnr_text"], mcnr_text),
        cell_string(
            row,
            COL["note"],
            f"track={mcnr.get('track', '')}; A轨 01-488_padding t0",
        ),
    ]
    res = call_tool(
        "sheet.set_range_value",
        {"file_id": FILE_ID, "sheet_id": SHEET_ID, "values": values},
    )
    print(f"[tencent] MCNR upsert {title} row={row} => {str(res)[:120]}")
    return row


def upsert_fpj(title: str, png: Path | str, *, init: bool = True) -> int:
    """Immediately attach FPJ preview for one sample."""
    if init:
        mcp_init()
    row = find_or_append_row(title)
    insert_image(Path(png), row, COL["fpj"], f"{title}-FPJ")
    return row


def upsert_lihe(title: str, png: Path | str, *, init: bool = True) -> int:
    """Immediately attach lihe preview for one sample."""
    if init:
        mcp_init()
    row = find_or_append_row(title)
    insert_image(Path(png), row, COL["lihe"], f"{title}-lihe")
    return row


def main(images_only: bool = False, start_row: int = 1) -> None:
    mcp_init()
    info = call_tool("sheet.get_sheet_info", {"file_id": FILE_ID})
    print("sheet_info:", json.dumps(info, ensure_ascii=False)[:500])

    records = collect_records()
    print("records:", [r["title"] for r in records])
    if not records:
        raise SystemExit("no sample records found")

    if not images_only:
        write_text(records)

    for i, rec in enumerate(records):
        row = i + 1
        if row < start_row:
            continue
        print(f"images for {rec['title']} (row {row})")
        try:
            insert_image(rec["fpj"], row, 11, f"{rec['title']}-FPJ")
            time.sleep(0.5)
            insert_image(rec["lihe"], row, 12, f"{rec['title']}-lihe")
            time.sleep(0.5)
        except Exception as exc:
            print(f"  ERROR on {rec['title']}: {exc}")
            # continue remaining rows
            time.sleep(2)
            continue

    data = call_tool(
        "sheet.get_cell_data",
        {
            "file_id": FILE_ID,
            "sheet_id": SHEET_ID,
            "start_row": 0,
            "end_row": min(3, len(records)),
            "start_col": 0,
            "end_col": 10,
            "return_csv": True,
        },
    )
    print("verify:", json.dumps(data, ensure_ascii=False)[:800])
    print("DONE")
    print(f"URL: https://docs.qq.com/sheet/{FILE_ID}?tab={SHEET_ID}")


if __name__ == "__main__":
    import sys

    images_only = "--images-only" in sys.argv
    start_row = 1
    for a in sys.argv[1:]:
        if a.startswith("--start-row="):
            start_row = int(a.split("=", 1)[1])
    main(images_only=images_only, start_row=start_row)
