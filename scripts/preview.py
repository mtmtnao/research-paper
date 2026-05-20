#!/usr/bin/env python3
"""notes/*.md を KaTeX 付き HTML に変換してブラウザで開く。

論文側 papers/<folder>/*.tex から \\newcommand / \\DeclareMathOperator 等の
独自マクロを自動抽出して KaTeX に渡すので、本文中に `\\bz` `\\pT` 等の
論文固有マクロがあっても破綻なくレンダリングされる。

Python 依存はゼロ（標準ライブラリのみ）。markdown→HTML / 数式 / マクロ展開は
全てブラウザ側で markdown-it + KaTeX (CDN) が担当する。

Usage:
    python3 scripts/preview.py notes/arXiv-1312.6114v11.md
    python3 scripts/preview.py notes/arXiv-1312.6114v11.md --watch
    python3 scripts/preview.py notes/arXiv-1312.6114v11.md --no-open
    python3 scripts/preview.py --publish        # docs/ に全ノートを一括レンダリング (+ index.html)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"
NOTES_DIR = ROOT / "notes"
DOCS_DIR = ROOT / "docs"
OUT_HTML = Path("/tmp/research-paper-preview.html")


# ----------------------------- TeX マクロ抽出 -----------------------------

CMD_RE = re.compile(
    r"\\(?:new|renew|provide)command\*?\s*\{?\s*\\([A-Za-z@]+)\s*\}?"
    r"(?:\s*\[(\d+)\])?"           # 引数の数
    r"(?:\s*\[[^\]]*\])?"          # デフォルト引数（無視）
    r"\s*\{"
)
DECL_OP_RE = re.compile(r"\\DeclareMathOperator\*?\s*\{\s*\\([A-Za-z@]+)\s*\}\s*\{")


def extract_balanced(s: str, open_idx: int) -> str | None:
    """s[open_idx] が '{' のとき、対応する '}' までの中身を返す。"""
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 1
    i = open_idx + 1
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : i]
        i += 1
    return None


def extract_macros_from_tex(tex_dir: Path) -> dict[str, str]:
    """papers/<folder>/ 配下の全 .tex を走査して KaTeX macros 辞書を作る。"""
    macros: dict[str, str] = {}
    if not tex_dir.is_dir():
        return macros
    for tex in sorted(tex_dir.rglob("*.tex")):
        try:
            content = tex.read_text(errors="replace")
        except Exception:
            continue

        for m in CMD_RE.finditer(content):
            name = m.group(1)
            body = extract_balanced(content, m.end() - 1)
            if body is None:
                continue
            # KaTeX は #1, #2 形式の引数をそのまま受ける
            macros.setdefault(f"\\{name}", body.strip())

        for m in DECL_OP_RE.finditer(content):
            name = m.group(1)
            body = extract_balanced(content, m.end() - 1)
            if body is None:
                continue
            macros.setdefault(f"\\{name}", f"\\operatorname{{{body.strip()}}}")

    return macros


def paper_dir_for(note_path: Path) -> Path:
    """notes/arXiv-XXXX.md → papers/arXiv-XXXX/"""
    return PAPERS_DIR / note_path.stem


# ----------------------------- text-mode LaTeX → Markdown 変換 -----------------------------

# `\cmd{...}` を変換するルール
# 値:
#   (left, right) → 中身を left+body+right に置換
#   None          → 命令ごと削除（中身も捨てる）
TEXT_CMD_RULES: dict[str, tuple[str, str] | None] = {
    # フォント装飾
    "textbf":     ("**", "**"),
    "textit":     ("*", "*"),
    "emph":       ("*", "*"),
    "texttt":     ("`", "`"),
    "textsc":     ("", ""),
    "textrm":     ("", ""),
    "textsf":     ("", ""),
    "textsl":     ("*", "*"),
    "underline":  ("", ""),
    "mbox":       ("", ""),
    # 参照系（label 名だけ残す）
    "ref":        ("", ""),
    "eqref":      ("(", ")"),
    "cref":       ("", ""),
    "Cref":       ("", ""),
    "autoref":    ("", ""),
    "pageref":    ("", ""),
    # 引用
    "cite":       ("[", "]"),
    "citep":      ("[", "]"),
    "citet":      ("", ""),
    "citeauthor": ("", ""),
    "citeyear":   ("(", ")"),
    # 脚注
    "footnote":   (" (", ")"),
    # 完全削除
    "label":      None,
    "input":      None,
    "include":    None,
    "vspace":     None,
    "hspace":     None,
    "noindent":   None,
}

# 2 引数のもの: \href{url}{text} → [text](url)
TWO_ARG_CMDS = {"href"}

CMD_NAME_RE = re.compile(r"\\([A-Za-z@]+)\*?")


def _consume_brace_arg(s: str, i: int) -> tuple[str, int] | None:
    """s[i:] の先頭の空白(同行のみ)をスキップしつつ '{...}' を読む。
    成功すれば (中身, 閉じ '}' の次の index) を返す。"""
    j = i
    while j < len(s) and s[j] in " \t":
        j += 1
    if j >= len(s) or s[j] != "{":
        return None
    body = extract_balanced(s, j)
    if body is None:
        return None
    return body, j + 1 + len(body) + 1


def _transform_text_once(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        m = CMD_NAME_RE.match(s, i)
        if not m:
            # \% \$ \& \# \_ などの単一記号エスケープを素のリテラルにする
            if i + 1 < n and s[i + 1] in "%$&#_":
                out.append(s[i + 1])
                i += 2
                continue
            out.append(s[i])
            i += 1
            continue
        cmd = m.group(1)
        after_cmd = m.end()
        # 2 引数命令
        if cmd in TWO_ARG_CMDS:
            r1 = _consume_brace_arg(s, after_cmd)
            if r1 is not None:
                body1, p1 = r1
                r2 = _consume_brace_arg(s, p1)
                if r2 is not None:
                    body2, p2 = r2
                    if cmd == "href":
                        out.append(f"[{body2}]({body1})")
                    i = p2
                    continue
        # 1 引数命令
        if cmd in TEXT_CMD_RULES:
            r = _consume_brace_arg(s, after_cmd)
            if r is not None:
                body, p = r
                rule = TEXT_CMD_RULES[cmd]
                if rule is not None:
                    out.append(rule[0] + body + rule[1])
                # rule is None なら完全削除
                i = p
                continue
        # 未知 or 引数を取らない命令はそのまま残す
        out.append(s[i:after_cmd])
        i = after_cmd
    return "".join(out)


def transform_text(s: str) -> str:
    """text 領域に LaTeX→Markdown 変換を fixed point まで適用。"""
    prev = ""
    while prev != s:
        prev = s
        s = _transform_text_once(s)
    # LaTeX の非破壊空白 ~ を半角空白に（"Table~3" / "Fig.~Hoge" など）
    s = re.sub(r"(\S)~(?=[A-Za-z0-9])", r"\1 ", s)
    return s


# math / code ブロックは触らずに保護する
PROTECT_RE = re.compile(
    r"(?P<fence>```[\s\S]*?```)"
    r"|(?P<inline_code>`[^`\n]+?`)"
    r"|(?P<display>\$\$[\s\S]+?\$\$)"
    r"|(?P<inline>\$[^\$\n]+?\$)"
    r"|(?P<dmath>\\\[[\s\S]+?\\\])"
    r"|(?P<imath>\\\([\s\S]+?\\\))"
)


def preprocess_latex(md: str) -> str:
    """ノート本文の text 領域だけ LaTeX→Markdown 変換する。
    math / code ブロックは無傷で通す（KaTeX や markdown-it に任せる）。"""
    out: list[str] = []
    last = 0
    for m in PROTECT_RE.finditer(md):
        out.append(transform_text(md[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(transform_text(md[last:]))
    return "".join(out)


# ----------------------------- HTML 生成 -----------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/markdown-it@14/dist/markdown-it.min.js"></script>
{watch_meta}
<style>
  :root {{
    --fg: #1f2328;
    --muted: #59636e;
    --bg: #ffffff;
    --border: #d1d9e0;
    --code-bg: #f6f8fa;
    --link: #0969da;
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
  h1, h2 {{ border-bottom: 1px solid var(--border); padding-bottom: .3em; margin-top: 1.8em; }}
  h3 {{ margin-top: 1.4em; }}
  a {{ color: var(--link); }}
  pre, code {{ background: var(--code-bg); border-radius: 6px; font-family: "SF Mono", Menlo, Consolas, monospace; }}
  pre {{ padding: 12px 14px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
  code {{ padding: .15em .4em; font-size: 87%; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 4px solid var(--border); color: var(--muted); padding: 0 1em; margin: 1em 0; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid var(--border); padding: 6px 12px; }}
  th {{ background: var(--code-bg); }}
  hr {{ border: none; border-top: 1px solid var(--border); margin: 2em 0; }}
  ul, ol {{ padding-left: 1.6em; }}
  li {{ margin: .25em 0; }}
  .katex {{ font-size: 1.02em; }}
  .katex-display {{ overflow-x: auto; overflow-y: hidden; padding: .2em 0; }}
  .meta-bar {{
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; font-size: 12px; color: var(--muted);
    margin-bottom: 1.5em;
  }}
  .meta-bar code {{ background: transparent; padding: 0; }}
</style>
</head>
<body>
<div class="meta-bar">
  source: <code>{src}</code>
  &nbsp;|&nbsp; macros: <code>{n_macros}</code> 件抽出
  {watch_label}
</div>
<div id="content"></div>
<script>
const RAW = {raw_json};
const MACROS = {macros_json};
window.addEventListener('load', () => {{
  const md = window.markdownit({{ html: false, linkify: true, typographer: false, breaks: false }});
  document.getElementById('content').innerHTML = md.render(RAW);
  renderMathInElement(document.body, {{
    delimiters: [
      {{ left: '$$', right: '$$', display: true }},
      {{ left: '$',  right: '$',  display: false }},
      {{ left: '\\\\[', right: '\\\\]', display: true }},
      {{ left: '\\\\(', right: '\\\\)', display: false }},
    ],
    macros: MACROS,
    throwOnError: false,
    errorColor: '#cc4444',
    strict: 'ignore',
  }});
  // スクロール位置を sessionStorage で保つ（--watch 時の auto refresh 対策）
  const key = 'scroll:' + location.pathname;
  const saved = sessionStorage.getItem(key);
  if (saved) window.scrollTo(0, parseInt(saved));
  window.addEventListener('scroll', () => sessionStorage.setItem(key, String(window.scrollY)));
}});
</script>
</body>
</html>
"""


def render_html(note_path: Path, macros: dict[str, str], watch_interval: int | None) -> str:
    raw = preprocess_latex(note_path.read_text())
    title = raw.splitlines()[0].lstrip("# ").strip() if raw else note_path.name
    watch_meta = (
        f'<meta http-equiv="refresh" content="{watch_interval}">'
        if watch_interval else ""
    )
    watch_label = (
        f"&nbsp;|&nbsp; auto-refresh: <code>{watch_interval}s</code>"
        if watch_interval else ""
    )
    return HTML_TEMPLATE.format(
        title=html.escape(title),
        src=html.escape(str(note_path.relative_to(ROOT))),
        n_macros=len(macros),
        watch_meta=watch_meta,
        watch_label=watch_label,
        raw_json=json.dumps(raw, ensure_ascii=False),
        macros_json=json.dumps(macros, ensure_ascii=False),
    )


# ----------------------------- 公開ビルド (docs/) -----------------------------

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Research Paper Notes</title>
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
  li {{ padding: .55em 0; border-bottom: 1px solid var(--border); }}
  .id {{
    color: var(--muted);
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 12.5px; margin-right: .8em;
  }}
  .meta {{ color: var(--muted); font-size: 13px; margin-top: -.3em; }}
</style>
</head>
<body>
<h1>Research Paper Notes</h1>
<p class="meta">{n} notes &middot; generated {ts}</p>
<ul>
{items}
</ul>
</body>
</html>
"""


def _read_title(note: Path) -> str:
    try:
        first = note.read_text().splitlines()[0]
    except Exception:
        return note.name
    return first.lstrip("# ").strip() or note.name


def _build_index(entries: list[tuple[str, str]]) -> str:
    # arXiv ID は概ね時系列なので新しい順に
    rows = sorted(entries, key=lambda x: x[0], reverse=True)
    items = "\n".join(
        f'<li><span class="id">{html.escape(stem)}</span>'
        f'<a href="{html.escape(stem)}.html">{html.escape(title)}</a></li>'
        for stem, title in rows
    )
    return INDEX_TEMPLATE.format(
        n=len(rows),
        ts=time.strftime("%Y-%m-%d %H:%M"),
        items=items,
    )


def publish_all(out_dir: Path) -> int:
    """notes/*.md を out_dir/<stem>.html に一括レンダリングし index.html を作る。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[str, str]] = []
    for note in sorted(NOTES_DIR.glob("*.md")):
        if note.stem.startswith("_"):
            continue  # _template.md など
        pdir = paper_dir_for(note)
        macros = extract_macros_from_tex(pdir)
        html_txt = render_html(note, macros, watch_interval=None)
        (out_dir / f"{note.stem}.html").write_text(html_txt)
        entries.append((note.stem, _read_title(note)))
        print(f"[publish] {note.stem}.html  (macros: {len(macros)})")
    (out_dir / "index.html").write_text(_build_index(entries))
    print(f"[publish] index.html ({len(entries)} notes) → {out_dir}")
    return 0


# ----------------------------- main -----------------------------

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("note", type=Path, nargs="?", help="notes/arXiv-*.md")
    p.add_argument("--watch", type=int, nargs="?", const=2, default=None,
                   help="ファイル変更を監視して再生成（既定 2 秒）")
    p.add_argument("--no-open", action="store_true", help="ブラウザを開かない")
    p.add_argument("--out", type=Path, default=OUT_HTML,
                   help=f"出力 HTML パス (default {OUT_HTML})")
    p.add_argument("--publish", action="store_true",
                   help="notes/ 全体を docs/ に一括レンダリング (+ index.html)")
    p.add_argument("--publish-dir", type=Path, default=DOCS_DIR,
                   help=f"--publish の出力先 (default {DOCS_DIR.relative_to(ROOT)})")
    args = p.parse_args(argv[1:])

    if args.publish:
        return publish_all(args.publish_dir.resolve())

    if args.note is None:
        p.error("note を指定するか --publish を付けてください")

    note_path: Path = args.note.resolve()
    if not note_path.is_file():
        print(f"error: not a file: {note_path}", file=sys.stderr)
        return 1

    pdir = paper_dir_for(note_path)
    macros = extract_macros_from_tex(pdir)
    print(f"[preview] note  : {note_path}")
    print(f"[preview] paper : {pdir}{'' if pdir.exists() else '  (NOT FOUND)'}")
    print(f"[preview] macros: {len(macros)} extracted")

    def render_once():
        html_txt = render_html(note_path, macros, args.watch)
        args.out.write_text(html_txt)

    render_once()
    print(f"[preview] wrote : {args.out}")

    if not args.no_open:
        subprocess.run(["open", str(args.out)], check=False)

    if args.watch:
        print(f"[preview] watching {note_path} (Ctrl-C to stop)")
        last = note_path.stat().st_mtime
        try:
            while True:
                time.sleep(0.5)
                m = note_path.stat().st_mtime
                if m != last:
                    last = m
                    render_once()
                    print(f"[preview] regenerated at {time.strftime('%H:%M:%S')}")
        except KeyboardInterrupt:
            print("\n[preview] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
