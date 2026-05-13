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
