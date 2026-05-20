# Research Paper Notes

機械学習の論文を読み、**Summary / Takeaway / Critical Thoughts** の3点でノートとして蓄積するリポジトリ。

## フォルダ構成

```
research-paper/
├── README.md
├── papers/                          # 落としてきた論文をそのまま入れる（リネーム禁止）
│   └── arXiv-2305.14325v1/
└── notes/                           # 自分の読書ノート
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

---

## 新しい論文を追加する手順

1. arXiv から TeX ソース zip を落として `papers/` 配下に展開
2. `notes/_template.md` をコピー → `notes/{論文フォルダ名}.md` にリネーム
3. ノート冒頭にタイトル・URL・タグを記入
4. 読みながら Summary / Takeaway / Critical Thoughts を埋める

---

## ノート生成スクリプト (`scripts/note.py`)

`papers/arXiv-*/` のうち対応する `notes/*.md` が無いものに対し、Claude Code CLI (`claude -p`) をワーカーごとに subprocess で起動して並列に下書きを生成する。

```bash
python3 scripts/note.py            # デフォルト 3 並列
python3 scripts/note.py 5          # 並列度を変更
MODEL=claude-opus-4-6 python3 scripts/note.py
```

- 既存ノートは冪等 skip。途中で止めて再実行しても同じ論文を二重に処理しない。
- stdout/stderr は `logs/<folder>.log` に保存。
- **プラン使用制限の検出時は新規ワーカー起動を停止する**（追加課金回避）。
  - `claude` CLI の出力に `You've hit your ... limit` が現れたら停止フラグを立て、未起動の future を cancel する。
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
- GitHub Pages の有効化: リポジトリの **Settings → Pages → Source: `main` branch / `/docs`** を選ぶ。公開 URL は `https://<user>.github.io/research-paper/`（`index.html` がトップ、各論文ノートはそこからリンク）。

- 対応する `papers/<folder>/*.tex` から `\newcommand` / `\DeclareMathOperator` 等の独自マクロを自動抽出して KaTeX に渡すので、`\bz` `\pT` のような論文固有マクロを含む数式も正しくレンダリングされる。
- 本文（text mode）に残った LaTeX 制御命令も Markdown 等価物に自動変換する: `\textbf{X}`→`**X**`、`\emph{X}`→`*X*`、`\texttt{X}`→`` `X` ``、`\ref{tab:foo}`→`tab:foo`、`\citep{key}`→`[key]`、`\href{url}{text}`→`[text](url)`、`\%` `\$` `\&` `\#` `\_` → 素のリテラル、`\label{...}` `\input{...}` は削除、`WORD~\ref{X}` の非破壊空白 `~` は半角空白に。数式 (`$...$`, `$$...$$`, `\(..\)`, `\[..\]`) と code は無傷で通す。未対応の命令は触らず残るので、必要なら `TEXT_CMD_RULES` に追加する。

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
