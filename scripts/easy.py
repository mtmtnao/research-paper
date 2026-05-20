#!/usr/bin/env python3
"""papers/arXiv-*/ から「中学生向け」の平易な解説 Markdown を生成し、HTML として公開する。

一次ソース: papers/{folder}/   （TeX。最終的な真実）
補助ソース: notes/{folder}.md  （既にあれば方向性アンカーとして使う・無くても可）
出力:       notes_easy/{folder}.md, docs/easy/{folder}.html

Usage:
    python3 scripts/easy.py                            # 不足分を並列生成（default 3 並列）
    python3 scripts/easy.py 5                          # 並列度変更
    python3 scripts/easy.py --only arXiv-1312.6114v11  # 1 本だけ生成
    python3 scripts/easy.py --publish                  # notes_easy/*.md → docs/easy/*.html
    python3 scripts/easy.py --all                      # 生成 → publish を続けて実行
    MODEL=claude-opus-4-7 python3 scripts/easy.py

仕様:
- `claude -p` ヘッドレスを subprocess で並列起動（note.py と同じ機構を踏襲）
- 既存 notes_easy/*.md は skip（冪等）
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

# 最新の Opus。新版が出たらここを更新する。
MODEL = os.environ.get("MODEL", "claude-opus-4-7")
DEFAULT_PARALLEL = 3
PER_PAPER_TIMEOUT_SEC = 60 * 60  # 1 時間

TEMPLATE_REL = "notes_easy/_template.md"

# プラン使用制限の検出パターン（note.py と同じ）
LIMIT_RE = re.compile(r"You've hit your .*?limit", re.IGNORECASE)
LIMIT_HIT = threading.Event()


# ============================== プロンプト ==============================

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
あなたは ML 論文を「中学校 2 年生（14 歳）にもわかる日本語」で解説するアシスタントです。

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
4. notes_easy/{folder}.md を Write で新規作成する。

# 読者像（絶対遵守）
- 14 歳。九九・一次方程式・座標グラフは OK。
- 線形代数（ベクトル・行列・内積・固有値）は知らない。
- 確率（期待値・確率分布・条件付き確率）は知らない。
- 微積分（微分・積分・勾配・偏微分）は知らない。
- 対数 log・指数 exp・自然対数 ln は知らない。
- ニューラルネットワーク・機械学習用語は何も知らない前提で書く。
- これらが本文に出るたび、その場で 1〜2 文で噛み砕いて補足する。
  「キーワード辞典」セクションに丸投げしない（読者は前後にスクロールできない前提）。
  例: 「ベクトル（数を順番に並べたもの。住所の数字みたいに、複数の数で 1 つの物を表す道具）」
  例: 「log（数が 1 より小さくなるほどマイナスに大きくなる関数。0 に近づくほど大きな罰）」

# 数式ルール（絶対遵守）
数式を出したら、直後に **必ず以下 3 ブロック** を書く。3 ブロック揃えていない裸の数式は禁止。

```
$$ ...数式... $$

**この式が言ってること**: 〜を計算しているよ。〜が大きいほど〜になる。（1〜3 文・自然言語）

**記号の意味**:
- $記号$ … 〜のこと（中学生語で）
- $記号$ … 〜のこと
（出てくる記号すべて漏れなく）

**身近な例え**: テスト / サイコロ / 地図 / 料理のレシピ / 家の鍵 / 列に並ぶ人 / ジャンケン
  などから、その式の構造に似ているものを 1 つ選んで具体的に説明する。
```

- 数式は厳選すること（中核 2〜4 個）。論文の式を全部解説する必要はない。
- 数式が論文に無い survey 系などでは、無理に出さなくて良い。

# 専門用語ルール（絶対遵守）
- ML 用語（attention, transformer, embedding, fine-tuning, diffusion, gradient, loss, ...）
  が初出のとき、その場で「← これは〜のこと」と 1 行補足。
- 後の章で再登場した時、しばらく出ていなかったら軽く再補足。
- 略語（VAE, GAN, RL, LLM, ELBO, SGD, ...）は初出で必ず展開して平易に解説。

# 内容ルール
- **TeX に書かれていない事は書かない**。推測したくなったら「TeX 中には書かれていないが、
  おそらく〜」と明示する（嘘を断定しない）。
- 数値（精度・loss・パラメータ数）・データセット名・ベンチマーク名・baseline 名・指標名
  は TeX から正確に拾う。
- 投げ出さない。長くなって OK。中学生が「なるほど」と言える深さまで噛み砕く。
- 日本語で書く。

# 出力構造（テンプレと完全一致）
1 行目: `# {{正式タイトル}}（日本語パラフレーズ）`

## 一言で言うと
30〜80 字の 1 文。

## どんな問題を解こうとしてるの？
- 現実の困りごと（身近な例えで）
- 既存の方法の何が足りなかったか

## どうやって解いたの？
### コアアイデア
1〜2 段落で。必ず例え話を使う。
### 仕組み
箇条書きで step1, step2, ...。
### 主要な数式
中核 2〜4 個。各式に 3 ブロック解説。

## 何がすごいの？
- 結果の具体的な数値（TeX 由来）
- データセット / ベンチマーク（TeX 由来）
- baseline との差
- 著者が論文中で「貢献」「contribution」として挙げている事

## キーワード辞典
本文に出た専門用語を出現順または 50 音順で。1 用語 1 行で平易定義。

## ちょっと深掘り（中学生は飛ばして OK）
- もう少し正確な定義
- なぜそうなるかの直感
- TeX に書かれている重要な細部

## もとの論文・正規ノート
- 論文 TeX: `papers/{folder}/`
- 正規ノート: `notes/{folder}.md`（存在する場合のみ）

# 最後にセルフチェック
書き終えたら、自分でセクションを 1 つずつ見返して以下を点検してから完了する:
- 中学生が知らないであろう単語が、補足なしで残っていないか
- 出した数式すべてに「この式が言ってること / 記号の意味 / 身近な例え」3 ブロックが揃っているか
- TeX に書かれていない事を推測で断定していないか
"""


# ============================== 並列実行（生成） ==============================

def run_one(folder: str) -> tuple[str, bool, str]:
    if LIMIT_HIT.is_set():
        return folder, False, "skip (plan limit hit)"

    out_path = NOTES_EASY_DIR / f"{folder}.md"
    log_path = LOGS_DIR / f"easy-{folder}.log"

    if out_path.exists():
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
            f"# easy: {folder}\n"
            f"$ claude -p <easy prompt> --model {MODEL} ...\n\n"
            f"--- stdout ---\n{r.stdout}\n\n"
            f"--- stderr ---\n{r.stderr}\n"
        )
        m = LIMIT_RE.search(r.stderr or "") or LIMIT_RE.search(r.stdout or "")
        if m:
            LIMIT_HIT.set()
            return folder, False, f"PLAN LIMIT HIT: {m.group(0)}"
        ok = r.returncode == 0 and out_path.exists()
        msg = f"rc={r.returncode}, easy_written={'yes' if out_path.exists() else 'no'}"
        return folder, ok, msg
    except subprocess.TimeoutExpired:
        log_path.write_text(f"# easy: {folder}\nTIMEOUT after {PER_PAPER_TIMEOUT_SEC}s\n")
        return folder, False, "timeout"
    except FileNotFoundError:
        return folder, False, "claude CLI not found in PATH"
    except Exception as e:  # noqa: BLE001
        return folder, False, f"error: {e}"


def generate_all(parallel: int, only: str | None) -> int:
    LOGS_DIR.mkdir(exist_ok=True)
    NOTES_EASY_DIR.mkdir(exist_ok=True)

    all_papers = sorted(p.name for p in PAPERS_DIR.glob("arXiv-*") if p.is_dir())
    if only:
        if only not in all_papers:
            print(f"error: '{only}' is not under papers/", file=sys.stderr)
            return 1
        if (NOTES_EASY_DIR / f"{only}.md").exists():
            print(f"already exists: notes_easy/{only}.md (delete to regenerate)")
            return 0
        todo = [only]
    else:
        todo = [f for f in all_papers if not (NOTES_EASY_DIR / f"{f}.md").exists()]

    print(f"Model: {MODEL}")
    print(f"Parallel: {parallel}")
    print(f"Total papers: {len(all_papers)}, missing easy notes: {len(todo)}")
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


# ============================== HTML 公開 ==============================

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>中学生向け 論文解説</title>
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
<h1>中学生向け 論文解説</h1>
<div class="lead">
<p>機械学習の論文を「中学校 2 年生でも読める日本語」で解説したページ集です。</p>
<p>専門用語は出るたびに補足し、数式は「式が言ってること / 記号の意味 / 身近な例え」の 3 点でほどいています。</p>
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
    p.add_argument("--publish", action="store_true",
                   help="notes_easy/*.md → docs/easy/*.html を一括レンダリング")
    p.add_argument("--publish-dir", type=Path, default=DOCS_EASY_DIR,
                   help=f"--publish の出力先 (default {DOCS_EASY_DIR.relative_to(ROOT)})")
    p.add_argument("--all", action="store_true",
                   help="生成 → publish を続けて実行")
    args = p.parse_args(argv[1:])

    if args.publish:
        return publish_all(args.publish_dir.resolve())

    rc = generate_all(args.parallel, args.only)

    if args.all:
        # 生成側で一部失敗しても publish はとりあえず実行する
        # （成功した分だけでも HTML 化したいため）。
        publish_rc = publish_all(args.publish_dir.resolve())
        return rc if rc != 0 else publish_rc

    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
