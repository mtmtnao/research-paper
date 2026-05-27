#!/usr/bin/env python3
"""既存の notes/*.md を papers/arXiv-*/ の TeX と突き合わせて検証し、
要約が不十分・不正確なら追記・修正する。

Usage:
    python3 scripts/verify_notes.py                  # デフォルト 3 並列、全ノート
    python3 scripts/verify_notes.py 5                # 並列度 5
    python3 scripts/verify_notes.py --force          # 既に verify 済みでも再実行
    python3 scripts/verify_notes.py --only arXiv-2305.14325v1 arXiv-2404.19756v5
    MODEL=claude-opus-4-6 python3 scripts/verify_notes.py
    AGENT=codex MODEL=gpt-5.5 python3 scripts/verify_notes.py

仕様:
- `claude -p` または `codex exec` ヘッドレスを subprocess で起動（note.py と同じ流儀）
- 各ワーカーは独立したエージェントセッション
- 冪等: logs/<folder>.verify.json に直近検証時のノート SHA256 を記録、
  ノートが変わっていなければ skip（--force で無効化）
- stdout/stderr は logs/<folder>.verify.log に保存
- プラン使用制限を検出したら新規ワーカー起動を停止
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

AGENT = os.environ.get("AGENT", "claude").strip().lower()
DEFAULT_MODELS = {
    "claude": "claude-opus-4-7",
    "codex": "gpt-5.5",
}
MODEL = os.environ.get("MODEL", DEFAULT_MODELS.get(AGENT, ""))
DEFAULT_PARALLEL = 3
PER_PAPER_TIMEOUT_SEC = 60 * 60  # 1 時間

TEMPLATE_REL = "notes/_template.md"
EXAMPLE_REL = "notes/arXiv-2305.14325v1.md"

LIMIT_RE = re.compile(
    r"^(?:error:\s*)?(?:You've hit your (?!\.\.\.)[^\n]*limit|usage limit[^\n]*|rate limit[^\n]*|quota exceeded[^\n]*)$",
    re.IGNORECASE | re.MULTILINE,
)
LIMIT_HIT = threading.Event()

# Markdown 装飾や説明文中の `VERDICT: UPDATED` も許容して拾う。
VERDICT_RE = re.compile(r"VERDICT:\s*[`*_]*(OK|UPDATED|ISSUES)\b", re.MULTILINE)


def build_cmd(prompt: str) -> list[str]:
    if AGENT == "claude":
        return [
            "claude", "-p", prompt,
            "--model", MODEL,
            "--allowed-tools", "Read,Edit,Write,Glob,Grep",
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


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_path(folder: str) -> Path:
    return LOGS_DIR / f"{folder}.verify.json"


def load_state(folder: str) -> dict:
    p = state_path(folder)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_state(folder: str, data: dict) -> None:
    state_path(folder).write_text(json.dumps(data, ensure_ascii=False, indent=2))


def build_prompt(folder: str) -> str:
    return f"""\
あなたは ML 論文の読書ノートをレビューする厳しい校閲者です。
**最優先事項は「嘘・憶測・記憶ベースの記述を絶対に混入させないこと」**。
内容を増やすより、根拠の無い記述を削るほうが価値が高いと心得てください。

【対象論文】 papers/{folder}/  （arXiv TeX ソース = 唯一の真実 (Source of Truth)）
【検証対象ノート】 notes/{folder}.md
【構造テンプレ】 {TEMPLATE_REL}
【スタイル手本】 {EXAMPLE_REL}

==================================================
◆ 反ハルシネーション原則（最重要・例外なし）
==================================================
1. **TeX に書かれている事だけが事実**。あなたの事前知識・記憶・"有名な論文だから多分こう" は一切信用しない。
   - 同名・類似テーマの別論文の記憶で補完しない。著者名・所属・年・data set の規模など全て TeX を見て確認する。
2. **既存ノートの記述も無罪推定しない**。前任者（過去の自分や他の Claude セッション）の記述も全て TeX で裏取りする。
   裏取りできない記述は「未検証」として扱う。
3. **数値・固有名詞は必ず TeX から原文コピー**して確認:
   - 数値（精度、loss、パラメータ数、学習ステップ、表中の値）
   - データセット名・ベンチマーク名・指標名・baseline 手法名
   - モデル名・著者名・所属・会議名・年
   - ハイパーパラメータ、アーキテクチャの層数・次元
   桁・単位（M/B/%, top-1/top-5）まで一致を確認する。違っていれば修正。確認できなければ削除。
4. **推測語の禁止**: 「おそらく」「と思われる」「一般に」「通常」「~ だろう」等で
   TeX 根拠の無い主張を埋めない。書きたいなら TeX の根拠を必ず添える。
5. **TeX に無い事は書かない**。どうしても文脈補足が必要な場合のみ
   `（TeX 中には明示されていない / 評者補足）` と必ず明示する。
6. **引用は \\citep / \\cite キーを main.bbl で引いて初めて書ける**。
   citation key だけで論文タイトルを書かない。bbl で確認できなければ書かない。
7. **図表番号・式番号は TeX のラベルで確認**してから書く。「Table 3 が ~」と書くなら
   実際にその表が TeX に存在し、その内容になっていることを Read で確認する。
8. **不確実なら書かない・削る・"VERDICT: ISSUES" を出す**ほうを常に選ぶ。
   "なんとなく埋める" は最大の罪。空欄のままにする勇気を持つ。

==================================================
◆ 手順
==================================================
1. notes/{folder}.md を Read で全文読む（既存記述を頭に入れるが、まだ信用しない）
2. README.md の「TeX ソースの読み方 > 読む順序」に従って papers/{folder}/ の TeX を読む
   （main.tex → text/abstract → introduction → method → experiments → discussion
    → tables → figText/figure 参照 → main.bbl）
   - 必ず Read で実物を見る。ファイル名から内容を推測しない。
3. ノートの各記述について、TeX の根拠箇所を**逐一特定**する:
   - 根拠ファイル＋該当文を必ず確認してから「正しい」と判定
   - 根拠が見つからない記述は「要修正」または「要削除」候補
4. 以下の観点で評価する（チェックリスト）:
   a. テンプレの全セクションが埋まっているか（空欄・プレースホルダ残しが無いか）
   b. **問題 / 手法 / 結果 / 貢献** が論文 abstract & introduction & conclusion と整合しているか
   c. 数値・データセット名・baseline 名・指標名が TeX（特に tables/ と experiments）と一致しているか
   d. 著者自身が論文中で limitations / future work として認めている点が Critical Thoughts に反映されているか
   e. 推測・憶測・他論文の記憶で書かれた箇所が無いか
   f. 1行目のタイトルが TeX の \\title{{...}} と一致しているか
   g. arXiv ID / authors / venue / year がメタ情報と一致しているか
5. 問題があれば notes/{folder}.md を Edit / Write で修正:
   - **問題が無ければ絶対に変更しない**。表現の好み・文体調整・網羅性向上だけを理由に編集しない。
   - **誤りは削除または訂正**。曖昧な記述で誤魔化さない。
   - **空欄は無理に埋めない**。TeX 根拠がある事だけを書く。
     根拠が薄いセクションは `- (TeX 中に明確な記述なし)` と書いて空欄を残す方が良い。
   - 既存の良い記述（TeX で裏取りできた記述）は壊さない。
   - 修正・追記した箇所はノート末尾の「Notes / Quotes」セクションに
     `- (verified YYYY-MM-DD) 何をどう直したか / 根拠 TeX ファイル名` の形で 1〜5 行残す。
     例: `- (verified 2026-05-20) 結果セクションの GLUE スコアを 88.4 → 89.3 に修正 (text/experiments.tex, Table 1)`
6. 最後に必ず下記いずれか 1 行だけを **標準出力の最終行** に出力する:
   - `VERDICT: OK`       … 全記述を TeX で裏取りでき、修正不要だった
   - `VERDICT: UPDATED`  … 不足・誤りを TeX 根拠付きで修正・追記した
   - `VERDICT: ISSUES`   … 自動修正しきれない問題（TeX 不足、判断が難しい等）があり人間判断が必要

==================================================
◆ 絶対要件
==================================================
- 1行目 `# {{論文タイトル}}` は TeX の \\title{{...}} と一致させる
- 日本語で書く
- Edit / Write を使うのはノート本体（notes/{folder}.md）のみ
- TeX ファイルや他のノートは絶対に書き換えない
- 必ず Read で TeX 実物を見てから判定する。ファイル名・既存ノートからの推測で判定しない
- 迷ったら「書かない」「VERDICT: ISSUES」を選ぶ。盛らない、埋めない、推測しない
"""


def parse_verdict(stdout: str) -> str | None:
    matches = VERDICT_RE.findall(stdout or "")
    return matches[-1] if matches else None


def run_one(folder: str, force: bool) -> tuple[str, bool, str]:
    if LIMIT_HIT.is_set():
        return folder, False, "skip (plan limit hit)"

    note_path = NOTES_DIR / f"{folder}.md"
    paper_dir = PAPERS_DIR / folder
    log_path = LOGS_DIR / f"{folder}.verify.log"

    if not note_path.exists():
        return folder, False, "skip (no note to verify)"
    if not paper_dir.is_dir():
        return folder, False, "skip (no paper dir)"

    before_sha = sha256_of(note_path)
    state = load_state(folder)
    if not force and state.get("note_sha256") == before_sha:
        return folder, True, f"skip (already verified, verdict={state.get('verdict')})"

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
            f"# verify {folder}\n"
            f"$ {AGENT} <prompt> --model {MODEL} ...\n\n"
            f"--- stdout ---\n{r.stdout}\n\n"
            f"--- stderr ---\n{r.stderr}\n"
        )
        m = LIMIT_RE.search(r.stderr or "") or LIMIT_RE.search(r.stdout or "")
        if m:
            LIMIT_HIT.set()
            return folder, False, f"PLAN LIMIT HIT: {m.group(0)}"
        if r.returncode != 0:
            return folder, False, f"rc={r.returncode}"

        verdict = parse_verdict(r.stdout)
        after_sha = sha256_of(note_path)
        changed = (after_sha != before_sha)

        # verdict と実際の差分の整合性チェック（軽め）
        if verdict is None:
            note_msg = "no VERDICT line"
            ok = False
        elif verdict == "UPDATED" and not changed:
            note_msg = "VERDICT=UPDATED but note unchanged"
            ok = False
        elif verdict == "OK" and changed:
            note_msg = "VERDICT=OK but note changed (suspicious)"
            ok = False
        else:
            note_msg = f"verdict={verdict}, changed={changed}"
            ok = True

        save_state(folder, {
            "note_sha256": after_sha,
            "verdict": verdict or "UNKNOWN",
            "changed": changed,
        })
        return folder, ok, note_msg
    except subprocess.TimeoutExpired:
        log_path.write_text(f"# verify {folder}\nTIMEOUT after {PER_PAPER_TIMEOUT_SEC}s\n")
        return folder, False, "timeout"
    except FileNotFoundError:
        return folder, False, f"{AGENT} CLI not found in PATH"
    except Exception as e:
        return folder, False, f"error: {e}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parallel", nargs="?", type=int, default=DEFAULT_PARALLEL,
                        help=f"parallel workers (default {DEFAULT_PARALLEL})")
    parser.add_argument("--force", action="store_true",
                        help="ノートが変わっていなくても再検証する")
    parser.add_argument("--only", nargs="+", default=None,
                        help="対象フォルダを明示指定（例: arXiv-2305.14325v1）")
    args = parser.parse_args(argv[1:])

    LOGS_DIR.mkdir(exist_ok=True)

    if AGENT not in DEFAULT_MODELS:
        print(f"error: unsupported AGENT={AGENT!r} (expected 'claude' or 'codex')", file=sys.stderr)
        return 1

    if args.only:
        todo = list(args.only)
    else:
        todo = sorted(
            p.stem for p in NOTES_DIR.glob("arXiv-*.md")
            if (PAPERS_DIR / p.stem).is_dir()
        )

    print(f"Agent:    {AGENT}")
    print(f"Model:    {MODEL}")
    print(f"Parallel: {args.parallel}")
    print(f"Force:    {args.force}")
    print(f"Targets:  {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    counts = {"ok": 0, "fail": 0}
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(run_one, f, args.force): f for f in todo}
        for fut in as_completed(futures):
            folder, ok, msg = fut.result()
            tag = "[ok]  " if ok else "[fail]"
            print(f"{tag} {folder}: {msg}")
            counts["ok" if ok else "fail"] += 1
            if LIMIT_HIT.is_set():
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
