from __future__ import annotations

import csv
from datetime import datetime
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse
import webbrowser

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fwhm_beads
import fwhm_lines
from fwhm_common import (
    apply_tukey_outer_fence,
    plot_resolution_boxplot,
    read_rows_csv,
    read_tiff,
    sample_status,
    summary_statistics,
    write_json,
    write_rows_csv,
    write_summary_markdown,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FWHM ROI 审核</title>
<style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#17191d;background:#f4f5f7}
*{box-sizing:border-box}
body{margin:0}
button,input,select{font:inherit}
button{cursor:pointer}
button:disabled{cursor:not-allowed;opacity:.48}
button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #8bb8f5;outline-offset:2px}
header{position:sticky;top:0;z-index:20;min-height:56px;padding:10px 18px;background:#fff;border-bottom:1px solid #d9dde3;display:flex;align-items:center;gap:14px}
h1{font-size:18px;margin:0}
.muted{color:#68707c;font-size:13px}
.controls{position:sticky;top:56px;z-index:19;padding:10px 18px;background:#fff;border-bottom:1px solid #d9dde3;display:grid;grid-template-columns:minmax(220px,1fr) auto auto minmax(220px,auto);gap:8px;align-items:center}
input,select,.control-button{height:40px;border:1px solid #aeb6c2;border-radius:6px;background:#fff;padding:0 12px}
.tabs{display:flex;height:40px;border:1px solid #aeb6c2;border-radius:6px;overflow:hidden}
.tabs[hidden]{display:none}
.tab{min-width:72px;border:0;border-right:1px solid #aeb6c2;background:#fff;padding:0 12px}
.tab:last-child{border-right:0}
.tab.active{background:#1769e0;color:#fff}
.pager{display:flex;align-items:center;gap:8px}
#page-label{min-width:150px;text-align:center;color:#435064;font-size:13px}
.primary{height:40px;border:1px solid #1769e0;border-radius:6px;background:#1769e0;color:#fff;padding:0 14px}
.message{min-height:20px;color:#435064;font-size:13px;text-align:right}
main{width:min(1800px,100%);margin:0 auto;padding:14px 18px 28px}
#grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start}
.roi-card{min-width:0;background:#fff;border:1px solid #cfd5dd;border-radius:6px;overflow:hidden}
.roi-card.deleted{border-color:#c8cbd0;background:#f0f1f3}
.roi-card.deleted img{opacity:.48;filter:grayscale(1)}
.roi-header{min-height:52px;padding:8px 10px;border-bottom:1px solid #e0e4e9;display:flex;align-items:center;gap:10px}
.roi-copy{min-width:0;flex:1}
.roi-title{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:700}
.metric{margin-top:2px;color:#5d6672;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.status{display:inline-flex;align-items:center;min-height:22px;padding:0 7px;border:1px solid #b9c0ca;border-radius:4px;color:#435064;background:#f7f8fa;font-size:11px;font-weight:600}
.status.deleted{border-color:#e6a49d;color:#8f1d14;background:#fff1ef}
.decision{min-width:76px;height:38px;border-radius:6px;padding:0 14px;font-weight:700}
.decision.delete{border:1px solid #a61b12;background:#b42318;color:#fff}
.decision.restore{border:1px solid #75808e;background:#fff;color:#343b45}
.roi-card img{display:block;width:100%;aspect-ratio:13/7;object-fit:contain;background:#111}
.empty{grid-column:1/-1;padding:56px 20px;text-align:center;color:#68707c;background:#fff;border:1px solid #d9dde3;border-radius:6px}
@media(max-width:980px){
  .controls{grid-template-columns:1fr auto;top:56px}
  .pager{justify-self:end}
  .message{text-align:left}
  #grid{grid-template-columns:1fr}
}
@media(max-width:600px){
  header{align-items:flex-start;flex-direction:column;gap:2px}
  .controls{position:static;grid-template-columns:1fr;padding:10px}
  .pager{justify-self:stretch;justify-content:space-between}
  #page-label{min-width:0}
  main{padding:10px}
  .message{text-align:left}
}
</style>
</head>
<body>
<header><h1>FWHM ROI 审核</h1><span id="summary" class="muted"></span></header>
<div class="controls">
  <input id="search" aria-label="筛选 ROI" placeholder="筛选 ROI 编号或文件">
  <select id="view" aria-label="审核状态">
    <option value="all">全部 ROI</option>
    <option value="keep">仅保留</option>
    <option value="delete">仅删除</option>
  </select>
  <div id="direction-tabs" class="tabs" role="tablist" aria-label="测量方向" hidden>
    <button class="tab" data-direction="xy" role="tab">横向</button>
    <button class="tab" data-direction="z" role="tab">轴向</button>
  </div>
  <div class="pager" aria-label="ROI 分组">
    <button id="previous" class="control-button">上一组</button>
    <span id="page-label" aria-live="polite"></span>
    <button id="next" class="control-button">下一组</button>
  </div>
  <button id="generate" class="primary">生成审核结果</button>
  <span id="message" class="message" role="status"></span>
</div>
<main>
  <div id="grid" aria-live="polite"></div>
</main>
<script>
const PAGE_SIZE=150,LOAD_BATCH_SIZE=30;
let manifest=null,page=0,direction="all",loadedCount=LOAD_BATCH_SIZE,scrollTick=false;
const pending=new Set();
const $=id=>document.getElementById(id);
async function api(url,options){const response=await fetch(url,options);if(!response.ok)throw new Error(await response.text());return response.json()}
function state(item){return manifest.decisions[item.key]||"keep"}
function visibleItems(){
 const q=$("search").value.trim().toLowerCase(),view=$("view").value;
 return manifest.items.filter(item=>(direction==="all"||item.direction===direction)&&(!q||item.label.toLowerCase().includes(q)||item.source.toLowerCase().includes(q))&&(view==="all"||view===state(item)))
}
function fileName(path){return path.split(/[\\/]/).pop()||path}
function element(tag,className,text){
 const node=document.createElement(tag);
 if(className)node.className=className;
 if(text!==undefined)node.textContent=text;
 return node
}
function card(item){
 const deleted=state(item)==="delete";
 const article=element("article","roi-card"+(deleted?" deleted":""));
 const header=element("div","roi-header");
 const copy=element("div","roi-copy");
 const title=element("div","roi-title");
 title.appendChild(element("span","",item.label));
 title.appendChild(element("span","status"+(deleted?" deleted":""),deleted?"已删除":"保留"));
 const metric=element("div","metric",item.metric+" · "+fileName(item.source));
 metric.title=item.source;
 copy.append(title,metric);
 const button=element("button","decision "+(deleted?"restore":"delete"),deleted?"恢复":"删除");
 button.type="button";
 button.disabled=pending.has(item.key);
 button.setAttribute("aria-label",(deleted?"恢复 ":"删除 ")+item.label);
 button.onclick=()=>decide(item,deleted?"keep":"delete");
 const image=document.createElement("img");
 image.src="/api/roi-image?key="+encodeURIComponent(item.key);
 image.alt=item.label+" 原图与 profile";
 image.loading="lazy";
 image.onerror=()=>{image.alt=item.label+" 图像加载失败"};
 header.append(copy,button);
 article.append(header,image);
 return article
}
function render(){
 const items=visibleItems(),pages=Math.max(1,Math.ceil(items.length/PAGE_SIZE));
 page=Math.min(page,pages-1);
 const start=page*PAGE_SIZE,pageItems=items.slice(start,start+PAGE_SIZE);
 const currentItems=pageItems.slice(0,loadedCount),grid=$("grid");
 grid.replaceChildren();
 if(currentItems.length){for(const item of currentItems)grid.appendChild(card(item))}
 else{grid.appendChild(element("div","empty","没有符合当前筛选条件的 ROI"))}
 const deleted=items.filter(item=>state(item)==="delete").length;
 $("summary").textContent=`${items.length} 个自动合格 ROI · 已删除 ${deleted}`;
 $("page-label").textContent=`第 ${page+1} / ${pages} 组 · 已加载 ${currentItems.length} / ${pageItems.length} 个`;
 $("previous").disabled=page===0;
 $("next").disabled=page>=pages-1;
}
function resetPaging(){
 page=0;loadedCount=LOAD_BATCH_SIZE;window.scrollTo({top:0});render()
}
function changePage(offset){
 page+=offset;loadedCount=LOAD_BATCH_SIZE;window.scrollTo({top:0});render()
}
window.addEventListener("scroll",()=>{
 if(scrollTick)return;
 scrollTick=true;
 requestAnimationFrame(()=>{
  scrollTick=false;
  const nearBottom=window.innerHeight+window.scrollY>=document.documentElement.scrollHeight-400;
  const pageItemCount=visibleItems().slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE).length;
  if(nearBottom&&loadedCount<pageItemCount){
   loadedCount=Math.min(pageItemCount,loadedCount+LOAD_BATCH_SIZE);render()
  }
 })
});
async function decide(item,value){
 const previous=state(item);
 pending.add(item.key);
 if(value==="keep")delete manifest.decisions[item.key];else manifest.decisions[item.key]=value;
 render();
 try{
  await api("/api/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key:item.key,value})});
  $("message").textContent=item.label+(value==="delete"?" 已删除":" 已恢复")+"，已自动保存"
 }catch(error){
  if(previous==="keep")delete manifest.decisions[item.key];else manifest.decisions[item.key]=previous;
  $("message").textContent="保存失败："+error.message
 }finally{pending.delete(item.key);render()}
}
$("search").oninput=resetPaging;
$("view").onchange=resetPaging;
for(const tab of document.querySelectorAll(".tab")){
 tab.onclick=()=>{
  direction=tab.dataset.direction;
  for(const item of document.querySelectorAll(".tab")){
   const active=item.dataset.direction===direction;
   item.classList.toggle("active",active);
   item.setAttribute("aria-selected",active?"true":"false")
  }
  resetPaging()
 }
}
$("previous").onclick=()=>{if(page>0)changePage(-1)};
$("next").onclick=()=>changePage(1);
$("generate").onclick=async()=>{
 const button=$("generate");
 try{
  button.disabled=true;$("message").textContent="正在生成...";
  const result=await api("/api/generate",{method:"POST"});
  $("message").textContent="已生成："+result.output
 }catch(error){$("message").textContent="生成失败："+error.message}
 finally{button.disabled=false}
};
(async()=>{
 manifest=await api("/api/manifest");
 if(manifest.mode==="lines"){
  direction="xy";$("direction-tabs").hidden=false;
  document.querySelector('.tab[data-direction="xy"]').classList.add("active");
  document.querySelector('.tab[data-direction="xy"]').setAttribute("aria-selected","true")
 }
 render()
})().catch(error=>$("message").textContent="加载失败："+error.message);
</script>
</body>
</html>"""


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def _row_key(row: dict[str, Any], mode: str) -> str:
    source = str(row.get("source") or row.get("source_file"))
    identifier = str(row.get("candidate_id") or row.get("roi_id"))
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return f"{mode}-{digest}-{identifier}"


def _direction_key(row: dict[str, Any], mode: str, direction: str) -> str:
    base = _row_key(row, mode)
    return base if mode == "beads" else f"{base}-{direction}"


def _quality_accepted(row: dict[str, Any], mode: str) -> bool:
    if mode == "beads":
        return _as_bool(row.get("accepted"))
    return _as_bool(row.get("accepted_xy")) or _as_bool(row.get("accepted_z"))


class ReviewProject:
    def __init__(self, result_dir: Path):
        self.result_dir = result_dir.resolve()
        summary_path = self.result_dir / "fwhm_summary.json"
        measurements_path = self.result_dir / "fwhm_measurements.csv"
        if not summary_path.is_file() or not measurements_path.is_file():
            raise ValueError("result directory must contain fwhm_summary.json and fwhm_measurements.csv")
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.mode = str(self.summary.get("mode") or "beads")
        self.rows = read_rows_csv(measurements_path)
        self.rows_by_key: dict[str, dict[str, Any]] = {}
        self.direction_by_key: dict[str, str | None] = {}
        for row in self.rows:
            if self.mode == "beads" and _quality_accepted(row, self.mode):
                key = _row_key(row, self.mode)
                self.rows_by_key[key] = row
                self.direction_by_key[key] = None
            elif self.mode == "lines":
                for direction in ("xy", "z"):
                    if _as_bool(row.get(f"accepted_{direction}")):
                        key = _direction_key(row, self.mode, direction)
                        self.rows_by_key[key] = row
                        self.direction_by_key[key] = direction
        self.decision_path = self.result_dir / "roi-decisions.json"
        self.decisions: dict[str, str] = {}
        if self.decision_path.is_file():
            loaded = json.loads(self.decision_path.read_text(encoding="utf-8"))
            for key, value in loaded.get("decisions", {}).items():
                if value not in {"keep", "delete"}:
                    continue
                if key in self.rows_by_key:
                    if value == "delete":
                        self.decisions[key] = value
                    continue
                if self.mode == "lines" and value == "delete":
                    for direction in ("xy", "z"):
                        migrated = f"{key}-{direction}"
                        if migrated in self.rows_by_key:
                            self.decisions[migrated] = value
        self.asset_dir = self.result_dir / "review-assets"
        self.asset_dir.mkdir(exist_ok=True)
        self._image_cache: dict[str, tuple[np.ndarray, str]] = {}
        self.lock = threading.Lock()

    def manifest(self) -> dict[str, Any]:
        items = []
        for key, row in self.rows_by_key.items():
            identifier = row.get("candidate_id") or row.get("roi_id")
            source = str(row.get("source") or row.get("source_file"))
            direction = self.direction_by_key[key]
            if direction is None:
                xy = _as_float(row.get("fwhm_xy_nm"))
                z = _as_float(row.get("fwhm_z_nm"))
                metric = f"XY {xy:.1f} nm" if xy is not None else "XY rejected"
                if z is not None:
                    metric += f" · Z {z:.1f} nm"
            else:
                value = _as_float(row.get(f"fwhm_{direction}_nm"))
                metric = f"{direction.upper()} {value:.1f} nm"
            items.append(
                {
                    "key": key,
                    "label": f"ROI {identifier}",
                    "source": source,
                    "metric": metric,
                    "direction": direction,
                }
            )
        return {"mode": self.mode, "items": items, "decisions": self.decisions}

    def save_decision(self, key: str, value: str) -> None:
        if key not in self.rows_by_key or value not in {"keep", "delete"}:
            raise ValueError("invalid ROI decision")
        with self.lock:
            if value == "keep":
                self.decisions.pop(key, None)
            else:
                self.decisions[key] = value
            write_json(
                {
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "decisions": self.decisions,
                },
                self.decision_path,
            )

    def _load_source(self, source: str) -> np.ndarray:
        if source not in self._image_cache:
            parameters = self.summary.get("run_parameter", {})
            image, axes = read_tiff(
                Path(source),
                str(parameters.get("axes", "auto")),
                parameters.get("channel"),
                parameters.get("timepoint"),
            )
            self._image_cache[source] = (image, axes)
        return self._image_cache[source][0]

    def roi_asset(self, key: str) -> Path:
        if key not in self.rows_by_key:
            raise KeyError(key)
        direction = self.direction_by_key[key]
        asset_dir = (
            self.asset_dir
            if direction is None
            else self.asset_dir / direction.upper()
        )
        asset_dir.mkdir(exist_ok=True)
        path = asset_dir / f"{key}.png"
        if path.is_file():
            return path
        row = self.rows_by_key[key]
        source = str(row.get("source") or row.get("source_file"))
        image = self._load_source(source)
        if self.mode == "beads":
            self._render_bead_asset(image, row, path)
        else:
            assert direction is not None
            self._render_line_asset(image, row, path, direction)
        return path

    def _render_bead_asset(
        self,
        image: np.ndarray,
        row: dict[str, Any],
        path: Path,
    ) -> None:
        parameters = dict(self.summary.get("run_parameter", {}))
        for key in (
            "roi_radius_xy_px",
            "profile_half_width_px",
            "exclude_z_start",
            "exclude_z_end",
            "roi_radius_z_px",
        ):
            if key in parameters:
                parameters[key] = int(parameters[key])
        x, y = int(float(row["x_px"])), int(float(row["y_px"]))
        z = int(float(row["z_px"])) if image.ndim == 3 else None
        radius = int(parameters.get("roi_radius_xy_px", 6))
        half_width = int(parameters.get("profile_half_width_px", 1))
        crop_radius = radius + 4
        x0, x1 = max(0, x - crop_radius), min(image.shape[-1], x + crop_radius + 1)
        y0, y1 = max(0, y - crop_radius), min(image.shape[-2], y + crop_radius + 1)
        coordinate = (z, y, x) if z is not None else (y, x)
        profiles, centers = fwhm_beads._axis_profiles(image, coordinate, parameters)

        columns = 3 if image.ndim == 3 else 2
        figure, axes = plt.subplots(2, columns, figsize=(4.3 * columns, 7), constrained_layout=True)
        xy_axis = axes[0, 0]
        plane = image[z] if z is not None else image
        crop = plane[y0:y1, x0:x1]
        low, high = np.percentile(crop[np.isfinite(crop)], [1.0, 99.8])
        xy_axis.imshow(crop, cmap="gray", vmin=low, vmax=high)
        xy_axis.plot([x - radius - x0, x + radius - x0], [y - y0, y - y0], color="#00b7ff")
        xy_axis.plot([x - x0, x - x0], [y - radius - y0, y + radius - y0], color="#ff9f1c")
        xy_axis.set_title(f"ROI {row.get('candidate_id')} original XY")

        for column, axis_name in enumerate(("x", "y", "z")):
            if column >= columns or axis_name not in profiles:
                continue
            axis = axes[1, column]
            profile = profiles[axis_name]
            spacing = (
                float(self.summary["opt"]["pixel_size_z_nm"])
                if axis_name == "z"
                else float(self.summary["opt"]["pixel_size_xy_nm"])
            )
            coordinate_nm = np.arange(profile.size) * spacing
            axis.plot(coordinate_nm, profile, "o", markersize=3, label="data")
            try:
                fit = fwhm_beads.fit_gaussian_profile(profile, spacing, centers[axis_name])
                dense = np.linspace(0, coordinate_nm[-1], 300)
                axis.plot(
                    dense,
                    fwhm_beads.gaussian_with_offset(
                        dense, fit.amplitude, fit.center_nm, fit.sigma_nm, fit.offset
                    ),
                    label=f"FWHM {fit.fwhm_nm:.1f} nm",
                )
                axis.legend(fontsize=8)
            except (RuntimeError, ValueError):
                axis.set_title("fit failed")
            axis.set_xlabel("nm")
            axis.set_title(axis_name.upper())

        if image.ndim == 3:
            z_start = int(parameters.get("exclude_z_start", 1))
            z_stop = image.shape[0] - int(parameters.get("exclude_z_end", 1))
            xz = np.mean(image[z_start:z_stop, y - half_width : y + half_width + 1, x0:x1], axis=1)
            yz = np.mean(image[z_start:z_stop, y0:y1, x - half_width : x + half_width + 1], axis=2)
            for axis, view, title in ((axes[0, 1], xz, "XZ"), (axes[0, 2], yz, "YZ")):
                low, high = np.percentile(view[np.isfinite(view)], [1.0, 99.8])
                axis.imshow(view, cmap="gray", vmin=low, vmax=high, aspect="auto")
                axis.axhline(z - z_start, color="#fff176")
                axis.set_title(title)
        figure.savefig(path, dpi=150)
        plt.close(figure)

    def _render_line_asset(
        self,
        image: np.ndarray,
        row: dict[str, Any],
        path: Path,
        direction: str,
    ) -> None:
        parameters = dict(self.summary.get("run_parameter", {}))
        z = int(float(row["z_px"])) if image.ndim == 3 else None
        if direction == "xy":
            if image.ndim == 2:
                plane = image
            elif parameters.get("lateral_plane_mode") == "single_peak_layer":
                plane = image[z]
            elif "lateral_z_half_width_layers" in parameters:
                excluded_start = int(parameters.get("exclude_z_start", 1))
                excluded_stop = image.shape[0] - int(
                    parameters.get("exclude_z_end", 1)
                )
                slab_half = int(parameters["lateral_z_half_width_layers"])
                slab_start = max(excluded_start, z - slab_half)
                slab_stop = min(excluded_stop, z + slab_half + 1)
                plane = np.max(image[slab_start:slab_stop], axis=0)
            else:
                plane = np.max(image, axis=0)
            self._render_line_xy_asset(plane, row, path, parameters)
            return

        if image.ndim != 3:
            raise ValueError("axial review requires ZYX data")
        self._render_line_z_asset(image, row, path, parameters)

    def _render_line_xy_asset(
        self,
        plane: np.ndarray,
        row: dict[str, Any],
        path: Path,
        parameters: dict[str, Any],
    ) -> None:
        x, y = float(row["x_px"]), float(row["y_px"])
        nx, ny = float(row["normal_x"]), float(row["normal_y"])
        half = int(float(row["profile_half_length_xy_px"]))
        profile, _, _ = fwhm_lines._extract_perpendicular_profile(
            plane,
            np.array([y, x]),
            np.array([float(row["tangent_y"]), float(row["tangent_x"])]),
            half,
            int(parameters.get("profile_average_half_width_px", 1)),
        )
        crop_radius = half + 4
        x0, x1 = max(0, int(x) - crop_radius), min(
            plane.shape[1], int(x) + crop_radius + 1
        )
        y0, y1 = max(0, int(y) - crop_radius), min(
            plane.shape[0], int(y) + crop_radius + 1
        )
        crop = plane[y0:y1, x0:x1]
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        low, high = np.percentile(crop[np.isfinite(crop)], [1.0, 99.8])
        axes[0].imshow(crop, cmap="gray", vmin=low, vmax=high)
        axes[0].plot(
            [x - nx * half - x0, x + nx * half - x0],
            [y - ny * half - y0, y + ny * half - y0],
            color="#00d084",
        )
        axes[0].set_title(
            f"ROI {row.get('roi_id')} XY at z={row.get('z_px')}"
        )
        spacing = float(self.summary["opt"]["pixel_size_xy_nm"])
        coordinate_nm = np.arange(profile.size) * spacing
        axes[1].plot(coordinate_nm, profile, "o", markersize=3)
        try:
            fit = fwhm_beads.fit_gaussian_profile(profile, spacing, half)
            dense = np.linspace(0, coordinate_nm[-1], 300)
            axes[1].plot(
                dense,
                fwhm_beads.gaussian_with_offset(
                    dense, fit.amplitude, fit.center_nm, fit.sigma_nm, fit.offset
                ),
                label=f"FWHM {fit.fwhm_nm:.1f} nm",
            )
            axes[1].legend()
        except (RuntimeError, ValueError):
            pass
        axes[1].set_title("Perpendicular XY profile")
        axes[1].set_xlabel("nm")
        figure.savefig(path, dpi=150)
        plt.close(figure)

    def _render_line_z_asset(
        self,
        image: np.ndarray,
        row: dict[str, Any],
        path: Path,
        parameters: dict[str, Any],
    ) -> None:
        x, y = int(float(row["x_px"])), int(float(row["y_px"]))
        z = int(float(row["z_px"]))
        half_xy = int(float(row["profile_half_length_xy_px"]))
        half_z = int(float(row["profile_half_length_z_px"]))
        crop_radius = half_xy + 4
        x0, x1 = max(0, x - crop_radius), min(
            image.shape[2], x + crop_radius + 1
        )
        y0, y1 = max(0, y - crop_radius), min(
            image.shape[1], y + crop_radius + 1
        )
        excluded_start = int(parameters.get("exclude_z_start", 1))
        excluded_stop = image.shape[0] - int(parameters.get("exclude_z_end", 1))
        z0 = max(excluded_start, z - half_z)
        z1 = min(excluded_stop, z + half_z + 1)
        profile = np.mean(
            image[
                z0:z1,
                max(0, y - 1) : min(image.shape[1], y + 2),
                max(0, x - 1) : min(image.shape[2], x + 2),
            ],
            axis=(1, 2),
        )
        xz = np.mean(
            image[z0:z1, max(0, y - 1) : min(image.shape[1], y + 2), x0:x1],
            axis=1,
        )
        yz = np.mean(
            image[z0:z1, y0:y1, max(0, x - 1) : min(image.shape[2], x + 2)],
            axis=2,
        )
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        xy = image[z, y0:y1, x0:x1]
        low, high = np.percentile(xy[np.isfinite(xy)], [1.0, 99.8])
        axes[0, 0].imshow(xy, cmap="gray", vmin=low, vmax=high)
        axes[0, 0].plot(x - x0, y - y0, marker="+", color="#00d084", ms=12)
        axes[0, 0].set_title(f"ROI {row.get('roi_id')} XY at z={z}")
        for axis, view, title in (
            (axes[0, 1], xz, "XZ"),
            (axes[1, 0], yz, "YZ"),
        ):
            view_low, view_high = np.percentile(
                view[np.isfinite(view)], [1.0, 99.8]
            )
            axis.imshow(
                view,
                cmap="gray",
                vmin=view_low,
                vmax=view_high,
                aspect="auto",
            )
            axis.axhline(z - z0, color="#00d084")
            axis.set_title(title)

        spacing = float(self.summary["opt"]["pixel_size_z_nm"])
        coordinate_nm = np.arange(profile.size) * spacing
        axes[1, 1].plot(coordinate_nm, profile, "o", markersize=3)
        try:
            fit = fwhm_beads.fit_gaussian_profile(profile, spacing, z - z0)
            dense = np.linspace(0, coordinate_nm[-1], 300)
            axes[1, 1].plot(
                dense,
                fwhm_beads.gaussian_with_offset(
                    dense, fit.amplitude, fit.center_nm, fit.sigma_nm, fit.offset
                ),
                label=f"FWHM {fit.fwhm_nm:.1f} nm",
            )
            axes[1, 1].legend()
        except (RuntimeError, ValueError):
            pass
        axes[1, 1].set_title("Axial Z profile")
        axes[1, 1].set_xlabel("nm")
        figure.savefig(path, dpi=150)
        plt.close(figure)

    def generate_reviewed(self) -> Path:
        reviewed_rows = []
        for row in self.rows:
            row_copy = dict(row)
            if self.mode == "beads":
                key = _row_key(row, self.mode)
                row_copy["manual_review"] = self.decisions.get(key, "keep")
                row_copy["reviewed_included"] = (
                    _quality_accepted(row, self.mode)
                    and row_copy["manual_review"] != "delete"
                )
            else:
                for direction in ("xy", "z"):
                    key = _direction_key(row, self.mode, direction)
                    decision = self.decisions.get(key, "keep")
                    row_copy[f"manual_review_{direction}"] = decision
                    row_copy[f"reviewed_included_{direction}"] = (
                        _as_bool(row.get(f"accepted_{direction}"))
                        and decision != "delete"
                    )
                row_copy["reviewed_included"] = bool(
                    row_copy["reviewed_included_xy"]
                    or row_copy["reviewed_included_z"]
                )
            reviewed_rows.append(row_copy)

        numeric_keys = {
            "fwhm_xy_nm",
            "fwhm_z_nm",
            "fwhm_x_nm",
            "fwhm_y_nm",
        }
        for row in reviewed_rows:
            for key in numeric_keys:
                if key in row:
                    row[key] = _as_float(row[key])
            for key in (
                "accepted",
                "accepted_xy",
                "accepted_z",
                "reviewed_included_xy",
                "reviewed_included_z",
            ):
                if key in row:
                    row[key] = _as_bool(row[key])
            row["tukey_excluded_fwhm_xy_nm"] = False
            row["tukey_excluded_fwhm_z_nm"] = False
        retained = [row for row in reviewed_rows if row["reviewed_included"]]
        accepted_key_xy = (
            "accepted" if self.mode == "beads" else "reviewed_included_xy"
        )
        accepted_key_z = (
            "accepted" if self.mode == "beads" else "reviewed_included_z"
        )
        tukey_xy = apply_tukey_outer_fence(
            retained,
            "fwhm_xy_nm",
            accepted_key=accepted_key_xy,
        )
        tukey_z = apply_tukey_outer_fence(
            retained,
            "fwhm_z_nm",
            accepted_key=accepted_key_z,
        )
        xy_values = [
            row["fwhm_xy_nm"]
            for row in retained
            if row.get("fwhm_xy_nm") is not None
            and row.get(accepted_key_xy)
            and not row["tukey_excluded_fwhm_xy_nm"]
        ]
        z_values = [
            row["fwhm_z_nm"]
            for row in retained
            if row.get("fwhm_z_nm") is not None
            and row.get(accepted_key_z)
            and not row["tukey_excluded_fwhm_z_nm"]
        ]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        reviewed = self.result_dir / "reviewed" / timestamp
        suffix = 1
        while reviewed.exists():
            reviewed = self.result_dir / "reviewed" / f"{timestamp}-{suffix:02d}"
            suffix += 1
        reviewed.mkdir(parents=True)
        if self.mode == "lines":
            xy_groups = [
                f"{row.get('source_file')}::{row.get('line_id')}"
                for row in retained
                if row.get("fwhm_xy_nm") is not None
                and row.get(accepted_key_xy)
                and not row["tukey_excluded_fwhm_xy_nm"]
            ]
            z_groups = [
                f"{row.get('source_file')}::{row.get('line_id')}"
                for row in retained
                if row.get("fwhm_z_nm") is not None
                and row.get(accepted_key_z)
                and not row["tukey_excluded_fwhm_z_nm"]
            ]
        else:
            xy_groups = None
            z_groups = None
        reviewed_summary = {
            "mode": self.mode,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "retained_roi_count": len(retained),
            "deleted_roi_count": len(self.decisions),
            "final_xy_roi_count": len(xy_values),
            "final_z_roi_count": len(z_values),
            "status_xy": sample_status(len(xy_values)),
            "status_z": sample_status(len(z_values)),
            "dimensions": {
                "fwhm_xy_nm": summary_statistics(xy_values, xy_groups),
                "fwhm_z_nm": summary_statistics(z_values, z_groups),
            },
            "tukey": {"xy": tukey_xy, "z": tukey_z},
            "source_result": str(self.result_dir),
        }
        if self.mode == "lines":
            reviewed_summary.update(
                {
                    "deleted_xy_count": sum(
                        value == "delete"
                        for key, value in self.decisions.items()
                        if key.endswith("-xy")
                    ),
                    "deleted_z_count": sum(
                        value == "delete"
                        for key, value in self.decisions.items()
                        if key.endswith("-z")
                    ),
                }
            )
        write_rows_csv(reviewed_rows, reviewed / "fwhm_measurements.csv")
        write_json(reviewed_summary, reviewed / "fwhm_summary.json")
        plot_resolution_boxplot(
            xy_values,
            z_values,
            reviewed / "fwhm_boxplot.png",
        )
        reviewed_assets = reviewed / "review-assets"
        reviewed_assets.mkdir()
        if self.mode == "beads":
            for row in retained:
                key = _row_key(row, self.mode)
                source_asset = self.roi_asset(key)
                shutil.copy2(source_asset, reviewed_assets / source_asset.name)
        else:
            for direction in ("xy", "z"):
                direction_dir = reviewed_assets / direction.upper()
                direction_dir.mkdir()
                for row in retained:
                    if not row.get(f"reviewed_included_{direction}"):
                        continue
                    key = _direction_key(row, self.mode, direction)
                    source_asset = self.roi_asset(key)
                    shutil.copy2(source_asset, direction_dir / source_asset.name)
        write_summary_markdown(
            reviewed_summary,
            reviewed / "SUMMARY.md",
            [
                ("SUMMARY.md", "审核汇总与文件索引。"),
                ("fwhm_measurements.csv", "全部 ROI 数值与审核判定。"),
                ("fwhm_summary.json", "审核后统计。"),
                ("fwhm_boxplot.png", "审核后箱线图。"),
                ("review-assets/", "每个人工保留 ROI 的核验图。"),
            ],
        )
        return reviewed


class ReviewHandler(BaseHTTPRequestHandler):
    project: ReviewProject

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            payload = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/api/manifest":
            self._send_json(self.project.manifest())
            return
        if parsed.path == "/api/roi-image":
            key = parse_qs(parsed.query).get("key", [""])[0]
            try:
                path = self.project.roi_asset(key)
                payload = path.read_bytes()
            except (KeyError, ValueError, OSError) as error:
                self.send_error(HTTPStatus.NOT_FOUND, str(error))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/decision":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.project.save_decision(str(payload["key"]), str(payload["value"]))
                self._send_json({"ok": True})
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/api/generate":
            try:
                output = self.project.generate_reviewed()
                self._send_json({"output": str(output)})
            except (ValueError, OSError) as error:
                self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


def serve_review(
    result_dir: Path,
    open_browser: bool = True,
    port: int = 0,
) -> None:
    project = ReviewProject(result_dir)
    handler = type("BoundReviewHandler", (ReviewHandler,), {"project": project})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"FWHM ROI reviewer: {url}")
    print("Press Ctrl+C to stop the local reviewer.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
