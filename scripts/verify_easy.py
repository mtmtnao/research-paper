#!/usr/bin/env python3
"""既存の notes_easy/*.md を papers/arXiv-*/ の TeX と突き合わせて検証し、
初学者研究者向けの橋渡し解説として不正確・抽象化しすぎ・妥当性の説明不足があれば修正する。

Usage:
    python3 scripts/verify_easy.py                  # デフォルト 3 並列、全 easy ノート
    python3 scripts/verify_easy.py 5                # 並列度 5
    python3 scripts/verify_easy.py --force          # 既に verify 済みでも再実行
    python3 scripts/verify_easy.py --only arXiv-2305.14325v1 arXiv-2404.19756v5
    MODEL=claude-opus-4-7 python3 scripts/verify_easy.py
    AGENT=codex MODEL=gpt-5.5 python3 scripts/verify_easy.py

仕様:
- `claude -p` または `codex exec` ヘッドレスを subprocess で起動（verify_notes.py と同じ流儀）
- 各ワーカーは独立したエージェントセッション
- 冪等: logs/easy-<folder>.verify.json に直近検証時のノート SHA256 を記録、
  ノートが変わっていなければ skip（--force で無効化）
- stdout/stderr は logs/easy-<folder>.verify.log に保存
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
NOTES_EASY_DIR = ROOT / "notes_easy"
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
    return LOGS_DIR / f"easy-{folder}.verify.json"


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
    notes_path = NOTES_DIR / f"{folder}.md"
    if notes_path.exists():
        aux = (
            f"【補助ソース】notes/{folder}.md  ← 研究者向けの正規ノート。"
            "方向性のアンカーとして Read してよいが、内容は鵜呑みにせず、"
            "数値・名称・主張は必ず TeX で裏取りすること。"
        )
        final_note_line = f"- 正規ノート: `notes/{folder}.md`"
    else:
        aux = f"【補助ソース】notes/{folder}.md は存在しない。TeX のみが情報源。"
        final_note_line = "- 正規ノート: なし"

    return f"""\
あなたは ML 論文の「初学者研究者向けブリッジノート」をレビューする厳しい校閲者です。
**最優先事項は、TeX に無い断定・数値ミス・別論文の記憶による混入を絶対に残さないこと**。
そのうえで、説明が簡単すぎて論点が消えていないか、研究者が読むべき問題設定・仮定・妥当性が残っているかを確認してください。

【対象論文】 papers/{folder}/  （arXiv TeX ソース = 唯一の真実 (Source of Truth)）
【検証対象】 notes_easy/{folder}.md
{aux}
【構造テンプレ】 {TEMPLATE_REL}

==================================================
◆ 反ハルシネーション原則（最重要・例外なし）
==================================================
1. **TeX に書かれている事だけが事実**。あなたの事前知識・記憶・"有名な論文だから多分こう" は一切信用しない。
   - 同名・類似テーマの別論文の記憶で補完しない。著者名・所属・年・data set の規模など全て TeX を見て確認する。
2. **既存の notes_easy と notes も無罪推定しない**。前任者の記述も全て TeX で裏取りする。
   裏取りできない記述は「未検証」として扱う。
3. **数値・固有名詞は必ず TeX から原文コピー**して確認:
   - 数値（精度、loss、パラメータ数、学習ステップ、表中の値）
   - データセット名・ベンチマーク名・指標名・baseline 手法名
   - モデル名・著者名・所属・会議名・年
   - ハイパーパラメータ、アーキテクチャの層数・次元
   桁・単位（M/B/%, top-1/top-5）まで一致を確認する。違っていれば修正。確認できなければ削除。
4. **推測語で穴埋めしない**: 「おそらく」「と思われる」「一般に」「通常」「~ だろう」等で
   TeX 根拠の無い主張を埋めない。必要なら削る。
5. **論文中の表記を尊重する**。定義・手法名・モデル名・データセット名・評価指標・主要な式は、なるべく TeX の表記を保つ。
6. **引用は \\citep / \\cite キーを main.bbl で引いて初めて書ける**。
   citation key だけで論文タイトルを書かない。bbl で確認できなければ書かない。
7. **図表番号・式番号は TeX のラベルで確認**してから書く。「Table 3 が ~」と書くなら
   実際にその表が TeX に存在し、その内容になっていることを Read で確認する。
8. **不確実なら書かない・削る・"VERDICT: ISSUES" を出す**ほうを常に選ぶ。
   "なんとなく埋める" は最大の罪。盛らない。

==================================================
◆ 初学者研究者向け品質基準
==================================================
読者は ML / CS / 数学の基礎を学び始めた研究者・大学院生。
線形代数・確率・微積分・最適化の基本語は見たことがあるが、この論文のサブフィールドや固有の記法には慣れていない。

以下を必ず確認してください:
1. 過度な比喩や抽象化で、論文が実際に議論している問題がぼやけていないか。
2. 問題設定、対象範囲、仮定、既存研究との差分、この論文で答えたい問いが明示されているか。
3. 専門用語は避けず、初出時に「この論文では何を指すか」が分かる説明になっているか。
4. 提案手法のコアアイデアが、論文中の定義・記法・手法名と接続しているか。
5. 実験・結果では、データセット、baseline、指標、主な数値、著者の contribution が TeX と一致しているか。
6. 妥当性と限界では、著者の limitations / future work と読者としての疑問が混ざらず整理されているか。
7. 重要な主張や定義に、TeX 由来の表記・短い原文フレーズ・式番号・表番号・ファイル名など、根拠が追える手がかりがあるか。

==================================================
◆ 数式チェック（絶対要件）
==================================================
表示数式（例: `$$...$$`, `\\[...\\]`）を出している場合、その直後に必ず下記 3 ブロックが揃っていること:

1. `**式の意味**`
2. `**記号の定義**`
3. `**この論文での役割**`

チェック方針:
- 裸の数式があれば、3 ブロックを追加するか、解説上不要なら数式ごと削除する。
- 式は TeX の原式をできるだけ保つ。記号を勝手に改名しない。
- 「記号の定義」は式中の主要な記号を漏れなく、論文中の意味で説明する。
- 目的関数・更新式・制約・評価式などが、手法や主張のどこに効いているかを書いているか確認する。
- 直感説明はよいが、比喩で技術的説明を置き換えていないか確認する。

==================================================
◆ 手順
==================================================
1. notes_easy/{folder}.md を Read で全文読む（既存記述を頭に入れるが、まだ信用しない）
2. {TEMPLATE_REL} を Read して、期待するセクション構造を確認する
3. notes/{folder}.md が存在すれば Read する（補助ソース。事実確認には使わない）
4. README.md の「TeX ソースの読み方 > 読む順序」に従って papers/{folder}/ の TeX を読む
   （main.tex → text/abstract → introduction → method → experiments → discussion
    → tables → figText/figure 参照 → main.bbl）
   - 必ず Read で実物を見る。ファイル名から内容を推測しない。
5. notes_easy の各記述について、TeX の根拠箇所を逐一特定する:
   - 根拠ファイル＋該当文を確認してから「正しい」と判定
   - 根拠が見つからない断定は、削除または弱めるのではなく、原則として削除
6. 以下の観点で評価する（チェックリスト）:
   a. テンプレの全セクションがあるか（プレースホルダ残しが無いか）
   b. 1行目が `# 正式タイトル（研究上の位置づけ）` 形式で、正式タイトル部分が TeX の \\title{{...}} と整合しているか
   c. 問題設定・仮定・コアアイデア・手法・結果が abstract / introduction / method / experiments / conclusion と整合しているか
   d. 数値・データセット名・baseline 名・指標名が TeX（特に tables/ と experiments）と一致しているか
   e. 専門用語がこの論文での使われ方として説明されているか
   f. 表示数式すべてに「式の意味 / 記号の定義 / この論文での役割」が揃っているか
   g. 妥当性・限界・追加で確認したい疑問が、TeX の主張と読者の評価に分けて書かれているか
   h. 「もとの論文・正規ノート」のリンクが下記と一致しているか:
      - 論文 TeX: `papers/{folder}/`
      {final_note_line}
7. 問題があれば notes_easy/{folder}.md を Edit / Write で修正:
   - **問題が無ければ絶対に変更しない**。表現の好み・文体調整だけを理由に編集しない。
   - **誤りは削除または訂正**。曖昧な記述で誤魔化さない。
   - **抽象化しすぎた説明は、TeX の定義・表記・数値・図表と接続し直す**。
   - **数式説明不足は必ず直す**。直せない場合は `VERDICT: ISSUES`。
   - 既存の良い記述（TeX で裏取りでき、初学者研究者向けに十分な具体性がある記述）は壊さない。
   - レビュー履歴セクションは追加しない。公開用の読み物として自然な本文を保つ。
8. 最後に必ず下記いずれか 1 行だけを **標準出力の最終行** に出力する:
   - `VERDICT: OK`       … 全記述を TeX で裏取りでき、初学者研究者向け品質も満たし、修正不要だった
   - `VERDICT: UPDATED`  … 不足・誤り・抽象化しすぎた説明を修正した
   - `VERDICT: ISSUES`   … 自動修正しきれない問題（TeX 不足、判断が難しい等）があり人間判断が必要

==================================================
◆ 絶対要件
==================================================
- 日本語で書く
- Edit / Write を使うのは notes_easy/{folder}.md のみ
- papers/, notes/, docs/ は絶対に書き換えない
- 必ず Read で TeX 実物を見てから判定する。ファイル名・既存ノートからの推測で判定しない
- 迷ったら「書かない」「VERDICT: ISSUES」を選ぶ。盛らない、埋めない、推測しない
"""


def parse_verdict(stdout: str) -> str | None:
    matches = VERDICT_RE.findall(stdout or "")
    return matches[-1] if matches else None


def run_one(folder: str, force: bool) -> tuple[str, bool, str]:
    if LIMIT_HIT.is_set():
        return folder, False, "skip (plan limit hit)"

    note_path = NOTES_EASY_DIR / f"{folder}.md"
    paper_dir = PAPERS_DIR / folder
    log_path = LOGS_DIR / f"easy-{folder}.verify.log"

    if not note_path.exists():
        return folder, False, "skip (no easy note to verify)"
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
            f"# verify easy {folder}\n"
            f"$ {AGENT} <easy verify prompt> --model {MODEL} ...\n\n"
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
            note_msg = "VERDICT=UPDATED but easy note unchanged"
            ok = False
        elif verdict == "OK" and changed:
            note_msg = "VERDICT=OK but easy note changed (suspicious)"
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
        log_path.write_text(f"# verify easy {folder}\nTIMEOUT after {PER_PAPER_TIMEOUT_SEC}s\n")
        return folder, False, "timeout"
    except FileNotFoundError:
        return folder, False, f"{AGENT} CLI not found in PATH"
    except Exception as e:
        return folder, False, f"error: {e}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("parallel", nargs="?", type=int, default=DEFAULT_PARALLEL,
                        help=f"parallel workers (default {DEFAULT_PARALLEL})")
    parser.add_argument("--force", action="store_true",
                        help="easy ノートが変わっていなくても再検証する")
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
            p.stem for p in NOTES_EASY_DIR.glob("arXiv-*.md")
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
