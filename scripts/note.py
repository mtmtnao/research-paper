#!/usr/bin/env python3
"""papers/arXiv-*/ に対して、対応する notes/*.md が無いものを並列に Claude で生成する。

Usage:
    python3 scripts/note.py            # デフォルト 3 並列
    python3 scripts/note.py 5          # 並列度を変更
    MODEL=claude-opus-4-6 python3 scripts/note.py

仕様:
- `claude -p` ヘッドレスを subprocess で起動
- 各ワーカーは独立した Claude セッション（コンテキストは混ざらない）
- 既存ノートは skip（冪等）
- stdout/stderr は logs/<folder>.log に保存
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"
NOTES_DIR = ROOT / "notes"
LOGS_DIR = ROOT / "logs"

# 最新の Opus。新しい版が出たらここを更新する。
# 2026-05-12 時点: claude-opus-4-7 (Anthropic 公式モデル一覧で確認済み)
MODEL = os.environ.get("MODEL", "claude-opus-4-7")
DEFAULT_PARALLEL = 3
PER_PAPER_TIMEOUT_SEC = 60 * 60  # 1 時間

TEMPLATE_REL = "notes/_template.md"
EXAMPLE_REL = "notes/arXiv-2305.14325v1.md"  # few-shot 用の手本ノート

# プラン使用制限（session/weekly/Opus limit）の検出パターン。
# 例: "You've hit your session limit · resets 3:45pm"
# 検出したら以降のワーカー起動を止めて追加課金を避ける。
LIMIT_RE = re.compile(r"You've hit your .*?limit", re.IGNORECASE)
LIMIT_HIT = threading.Event()


def build_prompt(folder: str) -> str:
    return f"""\
あなたは ML 論文を読んで日本語の読書ノートを書くアシスタントです。

【対象論文】 papers/{folder}/  （arXiv TeX ソース）
【出力先】   notes/{folder}.md
【構造テンプレ】 {TEMPLATE_REL}
【スタイル手本】 {EXAMPLE_REL}

手順:
1. README.md の「TeX ソースの読み方 > 読む順序」に従って papers/{folder}/ の TeX を読む
   （main.tex → text/abstract → introduction → method → experiments → discussion → tables → figText/figure 参照 → main.bbl）
2. {TEMPLATE_REL} を読んでセクション構造を厳密に把握する
3. {EXAMPLE_REL} を読んで粒度・口調・具体性のレベルを把握する
4. notes/{folder}.md を新規作成して書く

絶対要件:
- 1行目は実際の論文タイトルを `# {{正式タイトル}}` で
- テンプレの全セクションを埋める（空欄を残さない）
- 数値・データセット名・baseline 名・指標名は TeX から直接拾って書く
- 推測で書かない。TeX に書かれていない事は書かない、もしくは「TeX 中には明示されていない」と明記
- Critical Thoughts は率直に：強み / 弱み・疑問 / 次に試したいこと を分けて書く
- 著者自身が論文中で limitations として認めている点があれば必ず拾う
- 日本語で書く
- まず Read だけで全体把握してから Write すること
"""


def run_one(folder: str) -> tuple[str, bool, str]:
    if LIMIT_HIT.is_set():
        return folder, False, "skip (plan limit hit)"

    note_path = NOTES_DIR / f"{folder}.md"
    log_path = LOGS_DIR / f"{folder}.log"

    if note_path.exists():
        return folder, True, "skip (already exists)"

    cmd = [
        "claude", "-p", build_prompt(folder),
        "--model", MODEL,
        "--allowed-tools", "Read,Write,Glob,Grep",
        "--permission-mode", "acceptEdits",
    ]
    try:
        r = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=PER_PAPER_TIMEOUT_SEC,
        )
        log_path.write_text(
            f"# {folder}\n"
            f"$ claude -p <prompt> --model {MODEL} ...\n\n"
            f"--- stdout ---\n{r.stdout}\n\n"
            f"--- stderr ---\n{r.stderr}\n"
        )
        m = LIMIT_RE.search(r.stderr or "") or LIMIT_RE.search(r.stdout or "")
        if m:
            LIMIT_HIT.set()
            return folder, False, f"PLAN LIMIT HIT: {m.group(0)}"
        ok = r.returncode == 0 and note_path.exists()
        msg = f"rc={r.returncode}, note_written={'yes' if note_path.exists() else 'no'}"
        return folder, ok, msg
    except subprocess.TimeoutExpired:
        log_path.write_text(f"# {folder}\nTIMEOUT after {PER_PAPER_TIMEOUT_SEC}s\n")
        return folder, False, "timeout"
    except FileNotFoundError:
        return folder, False, "claude CLI not found in PATH"
    except Exception as e:
        return folder, False, f"error: {e}"


def main(argv: list[str]) -> int:
    parallel = int(argv[1]) if len(argv) > 1 else DEFAULT_PARALLEL
    LOGS_DIR.mkdir(exist_ok=True)

    all_papers = sorted(p.name for p in PAPERS_DIR.glob("arXiv-*") if p.is_dir())
    todo = [f for f in all_papers if not (NOTES_DIR / f"{f}.md").exists()]

    print(f"Model: {MODEL}")
    print(f"Parallel: {parallel}")
    print(f"Total papers: {len(all_papers)}, missing notes: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    counts = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(run_one, f): f for f in todo}
        for fut in as_completed(futures):
            folder, ok, msg = fut.result()
            tag = "[ok]  " if ok else "[fail]"
            print(f"{tag} {folder}: {msg}")
            counts["ok" if ok else "fail"] += 1
            if LIMIT_HIT.is_set():
                # 以降のワーカーは run_one 冒頭で即 skip される。
                # 走行中のものはそのまま完了を待つ（途中 kill はしない）。
                print("!! Plan usage limit detected. Stopping new work. !!")
                for pending in futures:
                    pending.cancel()

    if LIMIT_HIT.is_set():
        print(f"\n=== summary === ok={counts['ok']}, fail={counts['fail']} (STOPPED: plan limit)")
        return 2
    print(f"\n=== summary === ok={counts['ok']}, fail={counts['fail']}")
    return 0 if counts["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
