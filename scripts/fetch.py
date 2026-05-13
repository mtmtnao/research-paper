#!/usr/bin/env python3
"""arXiv TeX ソースを papers/ 配下にダウンロードする。

Usage:
    python3 scripts/fetch.py [path/to/arxiv_ids.txt]

仕様:
- バージョン未指定の ID は arXiv API で最新版を解決
- papers/arXiv-<id>v<N>/ がすでに存在すれば skip（冪等）
- TeX ソース (.tar.gz) を展開。PDF だけの場合は warn して source.pdf 保存
- arXiv ポリシーに従い 1 件ごとに 3 秒待つ（並列 DL はしない）
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIST = ROOT / "scripts" / "arxiv_ids.txt"
PAPERS_DIR = ROOT / "papers"
ARXIV_DELAY_SEC = 5.0
UA = "research-paper-fetch/1.0 (mailto:noreply@example.com)"
MAX_RETRIES = 5


def http_get(url: str, timeout: int = 60) -> bytes:
    """GET with retry on 429/5xx/timeout (exponential backoff)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                wait = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 sec
                print(f"  [retry] HTTP {e.code}, sleep {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            wait = (2 ** attempt) * 5
            print(f"  [retry] {type(e).__name__}: {e}, sleep {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
            time.sleep(wait)
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_err}")


def load_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            ids.append(s)
    return ids


def resolve_latest_version(arxiv_id: str) -> str:
    """ID に vN が無ければ最新版を取って付与する（abs ページ HTML を使う、API より緩い）。"""
    if re.search(r"v\d+$", arxiv_id):
        return arxiv_id
    body = http_get(f"https://arxiv.org/abs/{arxiv_id}", timeout=30).decode("utf-8", errors="replace")
    # 例: <meta name="citation_arxiv_id" content="2305.14325v1">
    m = re.search(r'citation_arxiv_id"\s+content="' + re.escape(arxiv_id) + r'v(\d+)"', body)
    if m:
        return f"{arxiv_id}v{m.group(1)}"
    # フォールバック: ページ内の [vN] / abs/<id>v<N> 表記を探す
    m = re.search(rf"{re.escape(arxiv_id)}v(\d+)", body)
    if m:
        return f"{arxiv_id}v{m.group(1)}"
    return f"{arxiv_id}v1"


def extract(blob: bytes, dest: Path) -> str:
    """blob を dest に展開して種類を返す: 'tex' / 'pdf' / 'unknown'."""
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "_download.bin"
    tmp.write_bytes(blob)

    head = blob[:8]

    # PDF 単体（TeX ソースが公開されていないケース）
    if head[:4] == b"%PDF":
        tmp.rename(dest / "source.pdf")
        return "pdf"

    # gzip → tar.gz か単体 .tex.gz
    if head[:2] == b"\x1f\x8b":
        r = subprocess.run(
            ["tar", "xzf", str(tmp), "-C", str(dest)],
            capture_output=True,
        )
        if r.returncode == 0:
            tmp.unlink()
            return "tex"
        # tar じゃない → 単体 gzip
        r = subprocess.run(["gunzip", "-c", str(tmp)], capture_output=True)
        if r.returncode == 0 and b"\\documentclass" in r.stdout[:5000]:
            (dest / "main.tex").write_bytes(r.stdout)
            tmp.unlink()
            return "tex"
        tmp.unlink()
        return "unknown"

    # bare TeX
    if b"\\documentclass" in blob[:5000] or blob.startswith(b"%"):
        tmp.rename(dest / "main.tex")
        return "tex"

    tmp.unlink(missing_ok=True)
    return "unknown"


def fetch_one(arxiv_id: str) -> str:
    """戻り値: 'ok' / 'skip' / 'warn-pdf' / 'fail'."""
    versioned = resolve_latest_version(arxiv_id)
    dest = PAPERS_DIR / f"arXiv-{versioned}"
    if dest.exists():
        print(f"[skip] {dest.name}")
        return "skip"

    print(f"[fetch] {versioned}")
    blob = http_get(f"https://arxiv.org/e-print/{versioned}", timeout=120)

    kind = extract(blob, dest)
    if kind == "tex":
        print(f"[ok]   {dest.name}")
        return "ok"
    if kind == "pdf":
        print(f"[warn] {dest.name} (PDF only, no TeX source)")
        return "warn-pdf"
    print(f"[fail] {dest.name} (unknown format)")
    shutil.rmtree(dest, ignore_errors=True)
    return "fail"


def main(argv: list[str]) -> int:
    list_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_LIST
    ids = load_ids(list_path)
    PAPERS_DIR.mkdir(exist_ok=True)
    print(f"Fetching {len(ids)} arXiv IDs → {PAPERS_DIR}/")

    summary: dict[str, int] = {}
    for i, aid in enumerate(ids):
        try:
            result = fetch_one(aid)
        except Exception as e:
            print(f"[err]  {aid}: {e}")
            result = "fail"
        summary[result] = summary.get(result, 0) + 1
        if i < len(ids) - 1:
            time.sleep(ARXIV_DELAY_SEC)

    print("\n=== summary ===")
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
