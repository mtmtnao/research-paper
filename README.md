# Research Paper Notes

機械学習の論文を読み、**Summary / Takeaway / Critical Thoughts** の3点でノートとして蓄積するリポジトリ。

## フォルダ構成

```
research-paper/
├── README.md
├── papers/                          # 落としてきた論文をそのまま入れる（リネーム禁止）
│   └── arXiv-2305.14325v1/
├── notes/                           # 自分の読書ノート
│   ├── _template.md                 # ひな形（コピーして使う）
│   └── arXiv-2305.14325v1.md        # 対応する論文フォルダと同名
└── notes_prelude/                   # notes を読む前の橋渡しノート
    ├── _template.md                 # ひな形（コピーして使う）
    └── arXiv-2305.14325v1.md        # 対応する論文フォルダと同名
```

## 運用ルール

### 1. 論文フォルダはリネームしない
arXiv から落としたまま（例: `arXiv-2305.14325v1/`）で `papers/` に入れる。
arXiv ID がそのままフォルダ名になっているので原典追跡もしやすい。

### 2. ノートは「論文フォルダ名 + .md」で作る
- `papers/arXiv-2305.14325v1/` ↔ `notes/arXiv-2305.14325v1.md`
- 1対1で機械的に対応。リネーム作業はゼロ。
- 将来スクリプトで突き合わせるのも簡単。

### 3. ノートの1行目は必ず `# {論文タイトル}` にする
ファイル名が arXiv ID だけだと中身が分からないので、人間可読なタイトルを1行目に書く。
エディタのプレビュー・Grep 検索で一発判別できる。

### 4. ノートは `notes/_template.md` をコピーして書く
構成:
- **Summary**: 著者の主張（問題・手法・結果・貢献）
- **Takeaway**: 自分にとっての要点（使い道・新しい視点）
- **Critical Thoughts**: 評価・疑問・次に試したいこと

### 5. Prelude は `notes_prelude/_template.md` をコピーして書く
`notes_prelude/` は `notes_easy/` と `notes/` の中間に置く橋渡しノート。
抽象的な説明ではなく、**対応する `notes/<folder>.md` を読めるようになるための前提**を書く。

構成:
- **この論文が扱う問題**: 何を解きたい論文なのか
- **先に知っておくとよい前提**: note を読むために必要な概念
- **重要な用語**: 一般的な意味ではなく、この論文での使われ方
- **最低限わかればよい数式**: 導出ではなく、式が何を言っているか
- **note の読み順**: Summary / Takeaway / Critical Thoughts のどこから読むか

---

## 新しい論文を追加する手順

1. arXiv から TeX ソース zip を落として `papers/` 配下に展開
2. `notes/_template.md` をコピー → `notes/{論文フォルダ名}.md` にリネーム
3. 必要なら `notes_prelude/_template.md` をコピー → `notes_prelude/{論文フォルダ名}.md` にリネーム
4. ノート冒頭にタイトル・URL・タグを記入
5. 読みながら Prelude → Summary / Takeaway / Critical Thoughts を埋める

---

## ノート生成スクリプト (`scripts/note.py`)

`papers/arXiv-*/` のうち対応する `notes/*.md` が無いものに対し、Claude Code CLI (`claude -p`) または Codex CLI (`codex exec`) をワーカーごとに subprocess で起動して並列に下書きを生成する。

```bash
python3 scripts/note.py            # デフォルト 3 並列
python3 scripts/note.py 5          # 並列度を変更
MODEL=claude-opus-4-6 python3 scripts/note.py
AGENT=codex MODEL=gpt-5.5 python3 scripts/note.py
```

- 既存ノートは冪等 skip。途中で止めて再実行しても同じ論文を二重に処理しない。
- stdout/stderr は `logs/<folder>.log` に保存。
- デフォルトは `AGENT=claude`。Codex CLI を使う場合は `AGENT=codex` を指定する。
- **プラン使用制限の検出時は新規ワーカー起動を停止する**（追加課金回避）。
  - CLI 出力に `You've hit your ... limit` / `usage limit` / `rate limit` / `quota exceeded` が現れたら停止フラグを立て、未起動の future を cancel する。
  - 既に走行中の subprocess は最後まで待つ（途中 kill はしない）。
  - 終了コード `2` で抜けるので、リセット後にもう一度同じコマンドを叩けば残りから続行できる。

---

## ノートプレビュー (`scripts/preview.py`)

`notes/*.md` を KaTeX 付き HTML に変換してブラウザで開く。Python 依存ゼロ（標準ライブラリのみ）。

```bash
python3 scripts/preview.py notes/arXiv-1312.6114v11.md
python3 scripts/preview.py notes/arXiv-1312.6114v11.md --watch    # ファイル変更を監視して再生成
python3 scripts/preview.py notes/arXiv-1312.6114v11.md --no-open  # ブラウザを開かない
python3 scripts/preview.py --publish                              # docs/ に全ノートを一括レンダリング (+ index.html)
```

- 個別プレビューは `/tmp/research-paper-preview.html` に書き出してブラウザで開く（下書き確認用）。
- `--publish` は `docs/<stem>.html` を一括生成し、目次 `docs/index.html` を作る（公開用、GitHub Pages から配信）。`docs/` 配下のファイル名は `notes/<stem>.md` と 1 対 1 でリネーム不要。
- 公開 URL: **<https://mtmtnao.github.io/research-paper/>**（`main` branch / `/docs` 配信）。`docs/` を push すると 1〜2 分で自動再ビルドされる。
- 更新フロー: `python3 scripts/preview.py --publish && git add docs/ && git commit && git push`。

- 対応する `papers/<folder>/*.tex` から `\newcommand` / `\DeclareMathOperator` 等の独自マクロを自動抽出して KaTeX に渡すので、`\bz` `\pT` のような論文固有マクロを含む数式も正しくレンダリングされる。
- 本文（text mode）に残った LaTeX 制御命令も Markdown 等価物に自動変換する: `\textbf{X}`→`**X**`、`\emph{X}`→`*X*`、`\texttt{X}`→`` `X` ``、`\ref{tab:foo}`→`tab:foo`、`\citep{key}`→`[key]`、`\href{url}{text}`→`[text](url)`、`\%` `\$` `\&` `\#` `\_` → 素のリテラル、`\label{...}` `\input{...}` は削除、`WORD~\ref{X}` の非破壊空白 `~` は半角空白に。数式 (`$...$`, `$$...$$`, `\(..\)`, `\[..\]`) と code は無傷で通す。未対応の命令は触らず残るので、必要なら `TEXT_CMD_RULES` に追加する。

---

## ノート検証スクリプト (`scripts/verify_notes.py`)

既存の `notes/*.md` を対応する `papers/arXiv-*/` の TeX と突き合わせ、誤り・根拠不足・不足情報があれば修正する。

```bash
python3 scripts/verify_notes.py                          # デフォルト 3 並列
python3 scripts/verify_notes.py 5                        # 並列度を変更
python3 scripts/verify_notes.py --force                  # 検証済みも再実行
python3 scripts/verify_notes.py --only arXiv-2305.14325v1
AGENT=codex MODEL=gpt-5.5 python3 scripts/verify_notes.py
```

- デフォルトは `AGENT=claude`。Codex CLI を使う場合は `AGENT=codex` を指定する。
- レビューは TeX を唯一の真実として行い、問題が無ければ表現調整だけの編集はしない。
- `VERDICT: OK / UPDATED / ISSUES` を標準出力から拾い、`logs/<folder>.verify.json` にノート SHA256 と一緒に保存する。
- stdout/stderr は `logs/<folder>.verify.log` に保存。

---

## 初学者研究者向けブリッジノート (`scripts/easy.py`)

研究者向けノート (`notes/`) は原論文の要点を短くまとめるため、初学者には問題設定・仮定・式の役割・評価の妥当性が見えにくいことがある。`scripts/easy.py` は同じ論文を「初学者の研究者が原論文と正規ノートを読めるようになる」粒度で整理し、`docs/easy/` に公開する独立系統。

```bash
python3 scripts/easy.py                            # 不足分を並列生成（default 3 並列）
python3 scripts/easy.py 5                          # 並列度変更
python3 scripts/easy.py 5 --force                  # 既存 notes_easy/*.md も上書き再生成
python3 scripts/easy.py --only arXiv-1312.6114v11  # 1 本だけ生成（試運転に）
python3 scripts/easy.py --publish                  # notes_easy/*.md → docs/easy/*.html を一括レンダリング
python3 scripts/easy.py --all                      # 生成 → publish を続けて実行
MODEL=claude-opus-4-7 python3 scripts/easy.py
AGENT=codex MODEL=gpt-5.5 python3 scripts/easy.py
```

- **一次ソース**: `papers/<folder>/`（TeX。最終的な真実）
- **補助ソース**: `notes/<folder>.md`（あれば方向性アンカーとして使う・無ければ TeX のみ）
- **出力**: `notes_easy/<folder>.md` ↔ `docs/easy/<folder>.html`（既存 `notes/` `docs/` と同じ stem。リネーム不要）
- 論文中の定義・式・数値・評価設定をできるだけ保ち、問題設定 / 仮定 / 手法 / 実験 / 妥当性 / 限界が追えるように補足する。
- 数式は **「式の意味 / 記号の定義 / この論文での役割」** の 3 点で整理し、TeX の原式と論文中の記法をなるべく崩さない。
- `note.py` と同じ機構: `claude -p` / `codex exec` 並列、既存 skip で冪等（`--force` で上書き再生成）、プラン制限検知で新規ワーカー停止（終了コード 2）、ログは `logs/easy-<folder>.log`。
- デフォルトは `AGENT=claude`。Codex CLI を使う場合は `AGENT=codex` を指定する。
- HTML レンダリングは `preview.py` の関数を import で再利用（`preview.py` 本体は改変しない）。論文固有マクロも自動抽出される。
- 公開 URL: **<https://mtmtnao.github.io/research-paper/easy/>**（`docs/easy/` を push すると自動配信）。

### easy ノート検証スクリプト (`scripts/verify_easy.py`)

既存の `notes_easy/*.md` を対応する `papers/arXiv-*/` の TeX と突き合わせ、誤り・根拠不足・抽象化しすぎた説明・妥当性の説明不足があれば修正する。

```bash
python3 scripts/verify_easy.py
python3 scripts/verify_easy.py 5
python3 scripts/verify_easy.py --force
python3 scripts/verify_easy.py --only arXiv-2305.14325v1
AGENT=codex MODEL=gpt-5.5 python3 scripts/verify_easy.py
```

- レビューは TeX を唯一の真実として行い、問題が無ければ表現調整だけの編集はしない。
- `VERDICT: OK / UPDATED / ISSUES` を標準出力から拾い、`logs/easy-<folder>.verify.json` に easy ノート SHA256 と一緒に保存する。
- stdout/stderr は `logs/easy-<folder>.verify.log` に保存。

---

## arXiv TeX ソースの読み方（参考）

論文フォルダの中身は概ね以下のような構造になっている:

```
arXiv-XXXX.XXXXX/
├── main.tex          # \input で各章を読み込む親ファイル
├── text/             # 本文（abstract / introduction / method / experiments / ...）
├── tables/           # 表（tabular）
├── figText/          # 図のキャプション + ファイル名対応
├── fig/              # 図の実体（1ページPDF）
└── main.bbl          # 参考文献
```

### 読む順序
1. `main.tex` で章構成と `\input` 先を把握
2. `text/abstract.tex` → `introduction.tex` → `discussion.tex` で **Summary**
3. `tables/*.tex` と `text/experiments.tex` で定量結果 → **Takeaway**
4. 本文で言及されている図だけ `figText/<name>.tex` でファイル名を確認し、`fig/<name>.pdf` を参照
5. 主張と図表の整合を評価 → **Critical Thoughts**
6. 引用は `\citep{key}` を `main.bbl` で引いて確認

### 形式の優先順位
1. **TeX (ローカル zip)** ← 最優先。本文・数式・表・図 全て揃う
2. ar5iv HTML (`https://ar5iv.labs.arxiv.org/html/<id>`) — TeX が無いとき
3. PDF — 最終手段（数式が壊れることあり）

---

## 将来の拡張（必要になったときだけ）

- 同テーマの論文が3本以上溜まったら `notes/_topics_*.md` のような横断ノートを追加
- 用語の意味を残したくなったら `notes/_glossary.md` を追加
- 一覧が欲しくなったら `INDEX.md` を追加（arXiv ID → タイトルの対応表）

最初は作らなくてよい。運用が回ってから必要に応じて足す。
