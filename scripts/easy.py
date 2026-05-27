#!/usr/bin/env python3
"""papers/arXiv-*/ から「初学者研究者向け」の橋渡し解説 Markdown を生成し、HTML として公開する。

一次ソース: papers/{folder}/   （TeX。最終的な真実）
補助ソース: notes/{folder}.md  （既にあれば方向性アンカーとして使う・無くても可）
出力:       notes_easy/{folder}.md, docs/easy/{folder}.html

Usage:
    python3 scripts/easy.py                            # 不足分を並列生成（default 3 並列）
    python3 scripts/easy.py 5                          # 並列度変更
    python3 scripts/easy.py 5 --force                  # 既存 notes_easy/*.md も上書き再生成
    python3 scripts/easy.py --only arXiv-1312.6114v11  # 1 本だけ生成
    python3 scripts/easy.py --publish                  # notes_easy/*.md → docs/easy/*.html
    python3 scripts/easy.py --all                      # 生成 → publish を続けて実行
    MODEL=claude-opus-4-7 python3 scripts/easy.py
    AGENT=codex MODEL=gpt-5.5 python3 scripts/easy.py

仕様:
- `claude -p` または `codex exec` ヘッドレスを subprocess で並列起動（note.py と同じ機構を踏襲）
- 既存 notes_easy/*.md は skip（冪等、--force で上書き再生成）
- プラン制限を検出したら新規ワーカー起動停止 → 終了コード 2
- HTML レンダリングは preview.py の純粋関数を import 経由で再利用（preview.py は改変しない）
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

# preview.py の純粋関数を再利用（読み取りのみ・preview.py は改変しない）
sys.path.insert(0, str(SCRIPTS_DIR))
from preview import (  # noqa: E402
    extract_macros_from_tex,
    preprocess_latex,
    HTML_TEMPLATE,
    paper_dir_for,
)

PAPERS_DIR = ROOT / "papers"
NOTES_DIR = ROOT / "notes"
NOTES_EASY_DIR = ROOT / "notes_easy"
DOCS_EASY_DIR = ROOT / "docs" / "easy"
LOGS_DIR = ROOT / "logs"

AGENT = os.environ.get("AGENT", "claude").strip().lower()
DEFAULT_MODELS = {
    "claude": "claude-opus-4-7",
    "codex": "gpt-5.5",
}
MODEL = os.environ.get("MODEL", DEFAULT_MODELS.get(AGENT, ""))
DEFAULT_PARALLEL = 3
PER_PAPER_TIMEOUT_SEC = 60 * 60  # 1 時間

TEMPLATE_REL = "notes_easy/_template.md"

# プラン使用制限の検出パターン（note.py と同じ）
LIMIT_RE = re.compile(
    r"^(?:error:\s*)?(?:You've hit your (?!\.\.\.)[^\n]*limit|usage limit[^\n]*|rate limit[^\n]*|quota exceeded[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
LIMIT_HIT = threading.Event()


# ============================== プロンプト ==============================

def build_cmd(prompt: str) -> list[str]:
    if AGENT == "claude":
        return [
            "claude", "-p", prompt,
            "--model", MODEL,
            "--allowed-tools", "Read,Write,Glob,Grep",
            "--permission-mode", "acceptEdits",
        ]
    if AGENT == "codex":
        return [
            "codex", "-a", "never", "exec",
            "-C", str(ROOT),
            "-m", MODEL,
            "-s", "workspace-write",
            prompt,
        ]
    raise ValueError(f"unsupported AGENT={AGENT!r} (expected 'claude' or 'codex')")


def build_prompt(folder: str) -> str:
    notes_path = NOTES_DIR / f"{folder}.md"
    if notes_path.exists():
        aux = (
            f"【補助ソース】notes/{folder}.md  ← 研究者向けの既存正規ノート。"
            f"最初に Read して方向性のアンカーとして使う。"
            f"内容は鵜呑みにせず、数値・名称・主張は必ず TeX で裏取りすること。"
        )
    else:
        aux = (
            f"【補助ソース】notes/{folder}.md は **存在しない**。TeX のみが情報源。"
        )

    return f"""\
あなたは ML 論文を「初学者の研究者が原論文と正規ノートを読めるようになる」ために整理するアシスタントです。
読み物として簡単にしすぎず、論文が実際に何を問題にし、何を仮定し、どの根拠で何を主張しているかを保ってください。

【一次ソース】papers/{folder}/  （arXiv TeX。最終的な真実）
{aux}
【出力先】     notes_easy/{folder}.md
【テンプレ】   {TEMPLATE_REL}

# 手順
1. notes/{folder}.md が存在すれば最初に Read する（補助ソース）。
2. papers/{folder}/ の TeX を README の「読む順序」に沿って読む:
   main.tex → text/abstract → introduction → method → experiments → discussion
   → tables → figText → main.bbl
3. {TEMPLATE_REL} を読んでセクション構造を厳密に把握する。
4. notes_easy/{folder}.md を Write で作成する。既存ファイルがある場合は、全体を読み直したうえで上書き再生成する。

# 読者像
- ML / CS / 数学の基礎を学び始めた研究者・大学院生。
- 線形代数・確率・微積分・最適化の基本語は見たことがあるが、この論文のサブフィールドや固有の記法には慣れていない。
- 過度な比喩や抽象化は不要。専門用語は避けず、論文中の意味が分かるように定義・前提・役割を補う。
- 目的は「雰囲気をつかむこと」ではなく、正規ノート `notes/{folder}.md` や原論文の該当箇所を読めるようにすること。

# 事実性・引用・妥当性のルール
- **TeX に書かれていない事は書かない**。推測したくなったら書かないか、`（TeX 中には明示されていない）` と明記する。
- 数値（精度・loss・パラメータ数）・データセット名・ベンチマーク名・baseline 名・指標名は TeX から直接拾う。
- 定義・手法名・モデル名・データセット名・評価指標・表中の値・主要な式は、なるべく TeX の表記をそのまま保つ。
- 重要な主張や定義は、長い英文コピーではなく、短い原文フレーズ・式番号・表番号・ファイル名を添えて根拠が分かるようにする。
- 著者が contribution / limitation / future work として書いていることは、著者の主張として扱う。自分の評価と混ぜない。
- 研究者向けに、手法の妥当性・仮定・評価設計・比較対象が主張を支えているかを明示する。
- 日本語で書く。

# 数式ルール
- 中核になる定義・目的関数・更新式・評価式を 2〜5 個程度に絞って載せる。論文の式を全部写す必要はない。
- TeX にある式は、可能な限り原式を保つ。記号を勝手に変えない。
- 数式を出したら、直後に以下を必ず書く:

```
$$ ...数式... $$

**式の意味**: 何を定義・最適化・比較している式かを 1〜3 文で説明する。

**記号の定義**:
- $記号$ ... 論文中での意味
- $記号$ ... 論文中での意味

**この論文での役割**: この式が手法・実験・主張のどこに効いているかを書く。
```

- 直感説明は入れてよいが、比喩で技術的説明を置き換えない。
- 論文に重要な式がほぼ無い場合は、無理に作らず「TeX 中に中核的な明示式は少ない」と書く。

# 出力構造（テンプレと完全一致）
1 行目: `# {{正式タイトル}}（研究上の位置づけ）`

## 一言で言うと
研究上の問い・提案・主な結論を 1〜2 文で正確に書く。

## 何を議論する論文か
- 問題設定
- 対象範囲 / 仮定
- 既存研究との差分
- この論文で答えたい問い

## 背景と前提
- この論文を読む前に必要な概念
- この論文での用語の使われ方
- 先行研究や baseline との関係

## 提案手法
### コアアイデア
抽象化しすぎず、論文中の用語・定義・仮定を保って説明する。
### 重要な定義・数式
中核 2〜5 個。各式に「式の意味 / 記号の定義 / この論文での役割」を付ける。
### 実装 / アルゴリズム上の要点
必要なら step 形式で整理する。

## 実験・結果
- データセット / ベンチマーク
- 比較対象 / baseline
- 指標
- 主な結果
- 著者が主張する貢献

## 妥当性と限界
- この主張を支える根拠
- 著者が認めている limitations / future work
- 読者として注意すべき点
- 追加で確認したい実験 / 疑問

## 用語メモ
一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

## 読む順番の提案
まず見るべき節・図表・式と、正規ノート `notes/{folder}.md` のどこにつながるかを書く。

## もとの論文・正規ノート
- 論文 TeX: `papers/{folder}/`
- 正規ノート: `notes/{folder}.md`（存在する場合のみ）

# 最後にセルフチェック
書き終えたら、自分でセクションを 1 つずつ見返して以下を点検してから完了する:
- 論文の問題設定・仮定・主張が抽象化でぼやけていないか
- 数値・固有名詞・baseline・指標・式が TeX と一致しているか
- 重要な定義・主張に TeX 由来の表記や短い根拠フレーズが残っているか
- 手法の妥当性、限界、評価設計へのコメントが事実と評価に分かれているか
- TeX に書かれていない事を推測で断定していないか
"""


# ============================== 並列実行（生成） ==============================

def run_one(folder: str, force: bool = False) -> tuple[str, bool, str]:
    if LIMIT_HIT.is_set():
        return folder, False, "skip (plan limit hit)"

    out_path = NOTES_EASY_DIR / f"{folder}.md"
    log_path = LOGS_DIR / f"easy-{folder}.log"

    if out_path.exists() and not force:
        return folder, True, "skip (already exists)"

    try:
        cmd = build_cmd(build_prompt(folder))
    except ValueError as e:
        return folder, False, str(e)

    try:
        r = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=PER_PAPER_TIMEOUT_SEC,
        )
        log_path.write_text(
            f"# easy: {folder}\n"
            f"$ {AGENT} <easy prompt> --model {MODEL} ...\n\n"
            f"--- stdout ---\n{r.stdout}\n\n"
            f"--- stderr ---\n{r.stderr}\n"
        )
        m = LIMIT_RE.search(r.stderr or "") or LIMIT_RE.search(r.stdout or "")
        if m:
            LIMIT_HIT.set()
            return folder, False, f"PLAN LIMIT HIT: {m.group(0)}"
        ok = r.returncode == 0 and out_path.exists()
        action = "overwritten" if force else "written"
        msg = f"rc={r.returncode}, easy_{action}={'yes' if out_path.exists() else 'no'}"
        return folder, ok, msg
    except subprocess.TimeoutExpired:
        log_path.write_text(f"# easy: {folder}\nTIMEOUT after {PER_PAPER_TIMEOUT_SEC}s\n")
        return folder, False, "timeout"
    except FileNotFoundError:
        return folder, False, f"{AGENT} CLI not found in PATH"
    except Exception as e:  # noqa: BLE001
        return folder, False, f"error: {e}"


def generate_all(parallel: int, only: str | None, force: bool = False) -> int:
    LOGS_DIR.mkdir(exist_ok=True)
    NOTES_EASY_DIR.mkdir(exist_ok=True)

    if AGENT not in DEFAULT_MODELS:
        print(f"error: unsupported AGENT={AGENT!r} (expected 'claude' or 'codex')", file=sys.stderr)
        return 1

    all_papers = sorted(p.name for p in PAPERS_DIR.glob("arXiv-*") if p.is_dir())
    if only:
        if only not in all_papers:
            print(f"error: '{only}' is not under papers/", file=sys.stderr)
            return 1
        if (NOTES_EASY_DIR / f"{only}.md").exists() and not force:
            print(f"already exists: notes_easy/{only}.md (delete to regenerate)")
            return 0
        todo = [only]
    elif force:
        todo = all_papers
    else:
        todo = [f for f in all_papers if not (NOTES_EASY_DIR / f"{f}.md").exists()]

    print(f"Agent: {AGENT}")
    print(f"Model: {MODEL}")
    print(f"Parallel: {parallel}")
    print(f"Force: {force}")
    label = "targets" if force else "missing easy notes"
    print(f"Total papers: {len(all_papers)}, {label}: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    counts = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futures = {ex.submit(run_one, f, force): f for f in todo}
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


# ============================== HTML 公開 ==============================

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>初学者研究者向け 論文解説</title>
<style>
  :root {{
    --fg: #1f2328; --muted: #59636e; --bg: #ffffff;
    --border: #d1d9e0; --code-bg: #f6f8fa; --link: #0969da;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --fg: #e6edf3; --muted: #9198a1; --bg: #0d1117;
      --border: #30363d; --code-bg: #161b22; --link: #4493f8;
    }}
  }}
  html, body {{ background: var(--bg); color: var(--fg); }}
  body {{
    max-width: 880px; margin: 2em auto; padding: 0 1.2em;
    font-family: -apple-system, BlinkMacSystemFont, "Hiragino Sans",
                 "Yu Gothic", "Segoe UI", sans-serif;
    line-height: 1.75; font-size: 16px;
  }}
  h1 {{ border-bottom: 1px solid var(--border); padding-bottom: .3em; }}
  a {{ color: var(--link); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: .8em 0; border-bottom: 1px solid var(--border); }}
  .id {{
    color: var(--muted);
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12.5px; margin-right: .8em;
  }}
  .summary {{ color: var(--muted); font-size: 14px; margin-top: .3em; }}
  .meta {{ color: var(--muted); font-size: 13px; margin-top: -.3em; }}
  .lead {{
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 1em 1.2em; margin: 1.5em 0;
    font-size: 14.5px; line-height: 1.75;
  }}
  .lead p {{ margin: .4em 0; }}
</style>
</head>
<body>
<h1>初学者研究者向け 論文解説</h1>
<div class="lead">
<p>機械学習の論文を、初学者の研究者が原論文と正規ノートへ進める粒度で整理したページ集です。</p>
<p>論文中の定義・式・数値・評価設定を保ちつつ、問題設定、仮定、手法、妥当性、限界を読み解けるように補足しています。</p>
<p>研究者向けの正規ノートは <a href="../">こちら</a> から見られます。</p>
</div>
<p class="meta">{n} 本 · 更新 {ts}</p>
<ul>
{items}
</ul>
</body>
</html>
"""


def _title_of(note: Path) -> str:
    try:
        first = note.read_text().splitlines()[0]
    except Exception:
        return note.name
    return first.lstrip("# ").strip() or note.name


def _oneliner_of(note: Path) -> str:
    """`## 一言で言うと` セクションの最初の非空行を返す。"""
    try:
        lines = note.read_text().splitlines()
    except Exception:
        return ""
    in_section = False
    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            in_section = (s == "## 一言で言うと")
            continue
        if in_section and s:
            return s
    return ""


def _render_one_html(note: Path) -> tuple[str, str, int]:
    """1 本の easy note を HTML 文字列にする。返り値は (html, title, n_macros)。"""
    pdir = paper_dir_for(note)
    macros = extract_macros_from_tex(pdir)
    raw = preprocess_latex(note.read_text())
    title = raw.splitlines()[0].lstrip("# ").strip() if raw else note.name
    html_txt = HTML_TEMPLATE.format(
        title=html.escape(title),
        src=html.escape(str(note.relative_to(ROOT))),
        n_macros=len(macros),
        watch_meta="",
        watch_label="",
        raw_json=json.dumps(raw, ensure_ascii=False),
        macros_json=json.dumps(macros, ensure_ascii=False),
    )
    return html_txt, title, len(macros)


def publish_all(out_dir: Path) -> int:
    if not NOTES_EASY_DIR.is_dir():
        print(f"error: {NOTES_EASY_DIR} does not exist. generate first.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str, str]] = []
    for note in sorted(NOTES_EASY_DIR.glob("*.md")):
        if note.stem.startswith("_"):
            continue  # _template.md など
        html_txt, _title, n_macros = _render_one_html(note)
        (out_dir / f"{note.stem}.html").write_text(html_txt)
        entries.append((note.stem, _title_of(note), _oneliner_of(note)))
        print(f"[publish] easy/{note.stem}.html  (macros: {n_macros})")

    # arXiv ID は概ね時系列なので新しい順に
    rows = sorted(entries, key=lambda x: x[0], reverse=True)
    items = "\n".join(
        f'<li><div><span class="id">{html.escape(stem)}</span>'
        f'<a href="{html.escape(stem)}.html">{html.escape(title)}</a></div>'
        + (f'<div class="summary">{html.escape(summary)}</div>' if summary else "")
        + "</li>"
        for stem, title, summary in rows
    )
    (out_dir / "index.html").write_text(INDEX_TEMPLATE.format(
        n=len(rows),
        ts=time.strftime("%Y-%m-%d %H:%M"),
        items=items,
    ))
    print(f"[publish] easy/index.html ({len(entries)} notes) → {out_dir}")
    return 0


# ============================== main ==============================

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("parallel", type=int, nargs="?", default=DEFAULT_PARALLEL,
                   help=f"並列度 (default {DEFAULT_PARALLEL})")
    p.add_argument("--only", type=str, default=None,
                   help="特定 1 本だけ生成 (例: --only arXiv-1312.6114v11)")
    p.add_argument("--force", action="store_true",
                   help="既存 notes_easy/*.md も上書き再生成する")
    p.add_argument("--publish", action="store_true",
                   help="notes_easy/*.md → docs/easy/*.html を一括レンダリング")
    p.add_argument("--publish-dir", type=Path, default=DOCS_EASY_DIR,
                   help=f"--publish の出力先 (default {DOCS_EASY_DIR.relative_to(ROOT)})")
    p.add_argument("--all", action="store_true",
                   help="生成 → publish を続けて実行")
    args = p.parse_args(argv[1:])

    if args.publish:
        return publish_all(args.publish_dir.resolve())

    rc = generate_all(args.parallel, args.only, args.force)

    if args.all:
        # 生成側で一部失敗しても publish はとりあえず実行する
        # （成功した分だけでも HTML 化したいため）。
        publish_rc = publish_all(args.publish_dir.resolve())
        return rc if rc != 0 else publish_rc

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
