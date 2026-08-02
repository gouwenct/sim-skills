#!/usr/bin/env python3
"""Download shared netdisk folders/files into a local date directory.

Supported sources (Stage -1 of /R-3DMPSIM):
  - PKU NetDisk (disk.pku.edu.cn / Anyshare anonymous share links)
  - Baidu NetDisk share links via BaiduPCS-Go (if installed and logged in)
  - Direct HTTP(S) URLs to zip/rar/7z/files
  - Local paths (already on disk / BaiduSyncdisk)

Examples:
  python -B download_netdisk.py ^
    --url "https://disk.pku.edu.cn/link/AAB5..." ^
    --dest "E:/.../天津数据/20260725" ^
    --name 35

  python -B download_netdisk.py --url "https://..." --dest <date_dir> --password xxxx
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import ssl
import subprocess
import sys
import time
import zipfile
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import (
    HTTPCookieProcessor,
    HTTPSHandler,
    Request,
    build_opener,
    urlopen,
)


def _ssl_context() -> ssl.SSLContext:
    """Windows cert-store can be broken; fall back to unverified context."""
    try:
        ctx = ssl.create_default_context()
        # probe load
        ctx.cert_store_stats()
        return ctx
    except Exception:
        return ssl._create_unverified_context()

try:
    import rarfile  # type: ignore
except Exception:  # pragma: no cover
    rarfile = None


USER_AGENT = "Mozilla/5.0 GrokBuild-Netdisk/1.0"
PKU_HOST_MARKERS = ("disk.pku.edu.cn", "pku.edu.cn/link", "pku.edu.cn/anyshare")
BAIDU_HOST_MARKERS = ("pan.baidu.com", "yun.baidu.com", "baidu.com/s/")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _safe_name(name: str) -> str:
    name = (name or "download").strip().replace("\x00", "")
    name = re.sub(r'[<>:"|?*]', "_", name)
    name = name.rstrip(" .")
    return name or "download"


# ---------------------------------------------------------------------------
# PKU NetDisk (Anyshare)
# ---------------------------------------------------------------------------

class PkuNetdiskClient:
    def __init__(self, link_id: str, password: str = "", base: str = "https://disk.pku.edu.cn"):
        self.link_id = link_id
        self.password = password or ""
        self.base = base.rstrip("/")
        self.cj = CookieJar()
        ctx = _ssl_context()
        self.opener = build_opener(
            HTTPCookieProcessor(self.cj),
            HTTPSHandler(context=ctx),
        )
        self.opener.addheaders = [("User-Agent", USER_AGENT)]
        self.token: Optional[str] = None

    def open_link(self) -> dict[str, Any]:
        req = Request(f"{self.base}/link/{self.link_id}", method="GET")
        with self.opener.open(req, timeout=60) as resp:
            resp.read()
        token = None
        for c in self.cj:
            if c.name == f"link_token:{self.link_id}" or c.name.startswith("link_token:"):
                token = c.value
                break
        if not token:
            raise RuntimeError("PKU link_token cookie missing; link may require login/password")
        self.token = token
        meta = self.api("GET", f"/shared-link/v1/links/{self.link_id}")
        # password flow: not always required; if needed, caller can extend
        if meta.get("password_required") and self.password:
            # Best-effort; exact password endpoint may vary by Anyshare version.
            try:
                self.api(
                    "POST",
                    f"/shared-link/v1/links/{self.link_id}/password",
                    {"password": self.password},
                )
            except Exception as exc:  # pragma: no cover
                _log(f"[warn] password submit may have failed: {exc}")
        return meta

    def api(self, method: str, path: str, body: Any = None) -> Any:
        if not self.token:
            raise RuntimeError("call open_link() first")
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        req = Request(f"{self.base}/api{path}", data=data, headers=headers, method=method)
        with self.opener.open(req, timeout=120) as resp:
            raw = resp.read()
            if not raw:
                return None
            return json.loads(raw.decode("utf-8"))

    def entry_items(self) -> list[dict[str, Any]]:
        data = self.api("GET", "/efast/v1/entry-item")
        if not isinstance(data, list) or not data:
            raise RuntimeError("entry-item empty; cannot resolve share root")
        return data

    def list_dir(self, docid: str) -> dict[str, Any]:
        return self.api("POST", "/efast/v1/dir/list", {"docid": docid}) or {}

    def download_file(self, docid: str, name: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and out_path.stat().st_size > 0:
            _log(f"[skip] exists {out_path} ({out_path.stat().st_size} bytes)")
            return
        osd = self.api(
            "POST",
            "/efast/v1/file/osdownload",
            {
                "docid": docid,
                "authtype": "QUERY_STRING",
                "savename": name,
                "usehttps": True,
            },
        )
        authreq = osd["authrequest"]
        url = authreq[1] if isinstance(authreq, list) else authreq
        _log(f"[get] {name} -> {out_path}")
        t0 = time.time()
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with self.opener.open(url, timeout=600) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(out_path)
        _log(f"[ok] {out_path.stat().st_size} bytes in {time.time() - t0:.1f}s")

    def walk_download(self, docid: str, base: Path) -> None:
        data = self.list_dir(docid)
        for d in data.get("dirs") or []:
            self.walk_download(d["docid"], base / _safe_name(d["name"]))
        for f in data.get("files") or []:
            self.download_file(f["docid"], f["name"], base / _safe_name(f["name"]))


def parse_pku_link(url: str) -> str:
    # https://disk.pku.edu.cn/link/AAB5...
    # https://disk.pku.edu.cn/anyshare/link/AAB5...
    m = re.search(r"/link/([A-Za-z0-9]+)", url)
    if not m:
        raise ValueError(f"cannot parse PKU link id from: {url}")
    return m.group(1)


def download_pku(url: str, dest_dir: Path, name: str = "", password: str = "") -> Path:
    link_id = parse_pku_link(url)
    client = PkuNetdiskClient(link_id, password=password)
    meta = client.open_link()
    title = _safe_name(name or meta.get("title") or "pku_share")
    entries = client.entry_items()
    root = entries[0]
    root_id = root["id"]
    out = dest_dir / title
    out.mkdir(parents=True, exist_ok=True)
    _log(f"[pku] title={title} root={root.get('name')} -> {out}")
    client.walk_download(root_id, out)
    return out


# ---------------------------------------------------------------------------
# Baidu via BaiduPCS-Go
# ---------------------------------------------------------------------------

def find_baidupcs() -> Optional[str]:
    for cand in ("BaiduPCS-Go", "baidupcs-go", "BaiduPCS-Go.exe"):
        p = shutil.which(cand)
        if p:
            return p
    # common manual install locations
    home = Path.home()
    for p in (
        home / ".local" / "bin" / "BaiduPCS-Go.exe",
        home / ".local" / "bin" / "BaiduPCS-Go",
        Path(r"C:\tools\BaiduPCS-Go\BaiduPCS-Go.exe"),
        Path(r"D:\tools\BaiduPCS-Go\BaiduPCS-Go.exe"),
    ):
        if p.exists():
            return str(p)
    return None


def download_baidu(url: str, dest_dir: Path, name: str = "", password: str = "") -> Path:
    exe = find_baidupcs()
    if not exe:
        raise RuntimeError(
            "BaiduPCS-Go not found. Install from "
            "https://github.com/qjfoidnh/BaiduPCS-Go/releases and login once "
            "(`BaiduPCS-Go login`), then re-run."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_name = _safe_name(name) if name else ""
    # BaiduPCS-Go transfer share
    # common: BaiduPCS-Go transfer <url> <code>  then download
    cmd = [exe, "transfer", url]
    if password:
        cmd.append(password)
    _log(f"[baidu] {' '.join(cmd)}")
    subprocess.run(cmd, check=False)
    # After transfer, user often needs `download`. Prefer share download command if available.
    # Fallback: try `BaiduPCS-Go download` with share link support in newer forks.
    dl_cmd = [exe, "download", url, "--saveto", str(dest_dir)]
    if password:
        dl_cmd.extend(["--password", password])
    _log(f"[baidu] {' '.join(dl_cmd)}")
    r = subprocess.run(dl_cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(
            "BaiduPCS-Go download failed. Ensure you are logged in "
            f"(`{exe} login`) and the share is accessible."
        )
    if out_name:
        return dest_dir / out_name
    return dest_dir


# ---------------------------------------------------------------------------
# Direct URL / local / archive extract
# ---------------------------------------------------------------------------

def download_direct(url: str, dest_dir: Path, name: str = "") -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    fname = name or Path(unquote(parsed.path)).name or "download.bin"
    fname = _safe_name(fname)
    out = dest_dir / fname
    if out.exists() and out.stat().st_size > 0:
        _log(f"[skip] exists {out}")
        return out
    _log(f"[get] {url} -> {out}")
    t0 = time.time()
    tmp = out.with_suffix(out.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=600, context=_ssl_context()) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(out)
    _log(f"[ok] {out.stat().st_size} bytes in {time.time() - t0:.1f}s")
    return out


def extract_archive(path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(dest_dir)
        _log(f"[unzip] {path.name} -> {dest_dir}")
        return dest_dir
    if suffix in {".rar", ".7z"}:
        if rarfile is None:
            raise RuntimeError(f"need rarfile/7z support to extract {path}")
        with rarfile.RarFile(path) as rf:
            rf.extractall(dest_dir)
        _log(f"[unrar] {path.name} -> {dest_dir}")
        return dest_dir
    return path


def detect_kind(url_or_path: str) -> str:
    s = url_or_path.strip()
    p = Path(s)
    if p.exists():
        return "local"
    low = s.lower()
    if any(m in low for m in PKU_HOST_MARKERS) or "/link/" in low and "pku" in low:
        return "pku"
    if any(m in low for m in BAIDU_HOST_MARKERS):
        return "baidu"
    if low.startswith("http://") or low.startswith("https://"):
        return "direct"
    return "unknown"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage -1 netdisk download for /R-3DMPSIM")
    ap.add_argument("--url", required=True, help="share URL / direct URL / local path")
    ap.add_argument("--dest", required=True, type=Path, help="target date directory")
    ap.add_argument("--name", default="", help="folder name under dest (e.g. 35)")
    ap.add_argument("--password", default="", help="share password / extraction code")
    ap.add_argument("--extract", action="store_true", help="extract zip/rar after download")
    ap.add_argument(
        "--kind",
        default="auto",
        choices=["auto", "pku", "baidu", "direct", "local"],
        help="force source type",
    )
    args = ap.parse_args(argv)

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    kind = args.kind if args.kind != "auto" else detect_kind(args.url)
    _log(f"[kind] {kind}")
    _log(f"[dest] {dest}")

    if kind == "local":
        src = Path(args.url)
        if not src.exists():
            raise SystemExit(f"local path not found: {src}")
        target_name = _safe_name(args.name or src.name)
        out = dest / target_name
        if src.is_dir():
            if out.resolve() != src.resolve():
                if out.exists():
                    _log(f"[skip] dest exists {out}")
                else:
                    shutil.copytree(src, out)
            result = out
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out if out.suffix else out / src.name)
            result = out if out.suffix else out / src.name
            if args.extract and result.suffix.lower() in {".zip", ".rar", ".7z"}:
                extract_archive(result, dest / target_name)
                result = dest / target_name
        _log(f"[done] {result}")
        print(result)
        return 0

    if kind == "pku":
        result = download_pku(args.url, dest, name=args.name, password=args.password)
        _log(f"[done] {result}")
        print(result)
        return 0

    if kind == "baidu":
        result = download_baidu(args.url, dest, name=args.name, password=args.password)
        _log(f"[done] {result}")
        print(result)
        return 0

    if kind == "direct":
        result = download_direct(args.url, dest, name=args.name)
        if args.extract and result.suffix.lower() in {".zip", ".rar", ".7z"}:
            folder = dest / _safe_name(args.name or result.stem)
            extract_archive(result, folder)
            result = folder
        _log(f"[done] {result}")
        print(result)
        return 0

    raise SystemExit(f"unsupported source kind for: {args.url}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (HTTPError, URLError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
