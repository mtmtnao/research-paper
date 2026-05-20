# Editing Models with Task Arithmetic

- arXiv: https://arxiv.org/abs/2212.04089
- source: ../papers/arXiv-2212.04089v3/
- authors: Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, Ali Farhadi
- venue / year: ICLR 2023
- tags: [model-editing, weight-arithmetic, fine-tuning, multi-task, unlearning, CLIP, T5, GPT-2]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: 事前学習済みモデルの振る舞いを後から「編集」したい場面（下流タスクの精度向上、バイアス除去、特定タスクの忘却、新情報への更新）は多いが、既存の編集法（追加学習・patching・editing）は学習を伴うため重い。学習に頼らずモデルを編集する統一的な枠組みが欲しい。
- **手法**: タスク $t$ に対する **task vector** を $\tau_t = \theta_\textrm{ft}^t - \theta_\textrm{pre}$（fine-tune 後の重み − 事前学習重み、要素ごと差）として定義し、$\theta_\textrm{new} = \theta + \lambda \tau_\textrm{new}$ で適用する（$\lambda$ は held-out 検証集合で決定）。task vector に対し3つの算術演算を定義: (1) **negation** $-\tau$（忘却/挙動抑制）、(2) **addition** $\sum_i \tau_i$（マルチタスク化や単一タスク強化）、(3) **analogy** $\tau_C + (\tau_B - \tau_A)$（"A:B :: C:D" の関係でデータの無い D を改善）。要素ごとの重み演算のみで、追加学習も推論時オーバーヘッドも無い。
- **結果**:
  - *Negation – 画像*: CLIP ViT-L/14、8 タスク平均で Pre-trained 64.8% → Negative task vector 19.0%（−45.8pt の忘却。参考までに Fine-tuned は 94.0%）。ImageNet コントロールは Pre-trained 75.5 → Negative 72.9（−2.6pt）。Gradient ascent は target 3.93%/control 16.3% でコントロール崩壊、Random vector は forget できず 60.9%/72.9%（Table 1）。
  - *Negation – 言語*: GPT-2 Large で toxic generation 比率を 4.8% → 0.8%（6 倍減）、avg toxicity 0.06→0.01、WikiText-103 perplexity は 16.4→16.9（+0.5）。Fine-tuned on non-toxic（1.8% / 17.2 ppl）より両指標で優れる。Gradient ascent は ppl > 10$^{10}$ で破綻（Table 2）。
  - *Addition – 画像*: CLIP の 8 タスクから 2 つを足すと、専門 fine-tuned 2 モデル使用時の **98.9%** 正規化精度に到達。$2^8$ 通りの部分集合を試行し、全 8 タスクを足した最良モデルは平均 91.2% を維持（Fig. 2-3, §4.1）。
  - *Addition – NLP*: T5-base を GLUE 4 タスクで fine-tune → さらに Hugging Face Hub 上の 427 候補 task vector を加算。Avg 78.1 → 78.6（MRPC +0.8, RTE +0.2, CoLA +0.7, SST-2 +0.2、Table 3）。
  - *Analogy – ドメイン汎化*: Yelp/Amazon の sentiment を、もう一方の sentiment ラベル + 両ドメインの教師なし LM の3 vectors で構成。T5-large で Yelp 95.0→95.1（target fine-tune 95.5）、Amazon 94.8→95.2（target 95.5）。auxiliary fine-tune を常に上回り target fine-tune に肉薄（Table 4）。
  - *Analogy – サブ集団*: ImageNet × 人手スケッチで「real/sketch × 2 クラス群」の 4 区分を作成、3 区分の vector から 4 つ目を合成 → pre-trained に対し平均 +3.4pt、これは新規アノテーション約 100 サンプル分の効果に相当（Fig. 4）。
- **貢献**: (1) 重み空間での「タスクベクトル算術」という統一的編集枠組み（negation/addition/analogy）の提案、(2) CLIP（vision）・T5（NLP）・GPT-2（LM）の複数モダリティ・モデル規模での効果実証、(3) 追加学習なし・推論コストなしでマルチタスク化・忘却・データなしドメイン汎化を実現する具体的レシピ、(4) 公開 Hugging Face Hub チェックポイントが事実上の "task vector のライブラリ" として利用できることの提示。

## Takeaway（自分にとっての要点）

- **task vector は同じ initialization から fine-tune された重みの差分**という、ほとんど工夫のない定義で「重みを足し引きするだけで挙動を編集できる」というのが本質。これだけ単純な操作で 6× toxic 削減や 8 タスク 91.2% マルチタスクが出る、というのが筋の良さ。
- 公開チェックポイントは「学習済みモデル」ではなく「公開された task vector の山」として再解釈できる。Hugging Face Hub の 3,000+ BERT-base / 800+ T5-small（discussion 限界節）はそのまま編集可能アセットになる。
- **Negation が忘却に効くがコントロール劣化が小さい**点は実用的に重要。gradient ascent はコントロールを壊すので unlearning には使いにくいが、negation は ImageNet を 2.6pt しか下げずに target を 45.8pt 落とせる。RLHF の代替/補完として toxic 行動の抑制に直接使える可能性がある（GPT-2 Large で 6× 削減、ppl +0.5）。
- **task vector 同士が概ね直交**（Fig. 5 cos sim）であることが addition で干渉が小さい理由として提示されている。意味的に近いタスク（MNIST/SVHN/GTSRB の digit 系、EuroSAT/RESISC45 の衛星系）だけ高 cos sim。これは「ベクトル算術」というメタファーが weight space で実際に成立する条件として重要。
- **学習率に敏感**: task vector を使う場合は fine-tune 単体より低 LR を推奨。NLP で他者のチェックポイントを混ぜると分散が大きい理由として LR を疑っている（discussion）。
- **中間 task vector が早期に方向収束**（Fig. 7 = fig:intermediate）→ 完全 fine-tune を待たずに加算でき、計算節約の余地がある。
- analogy で「教師なし LM 由来の差分」が sentiment ベクトルを別ドメインに転写できる、というのは weight space の差が "ドメイン差" を粗くエンコードしている示唆として面白い。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 提案がほぼ「引き算と足し算」だけで、再現が極めて容易（コード公開、HF Hub チェックポイントで追試可能）。
  - vision（CLIP, 3 サイズ）・NLP（T5, 3 サイズ）・LM（GPT-2）の3モダリティで一貫して効くことを示しており、特定モデル依存ではない。
  - negation を「gradient ascent」「random vector」「non-toxic fine-tune」と公平に比較しており、なぜ negation が選ばれるべきかの根拠が明示的（Table 1, 2）。
  - analogy 実験で T5-large の sentiment が "ターゲットの fine-tune とほぼ同じ" まで来るのは（95.1 vs 95.5）、ラベルなしドメインへの転写としてはかなり強い結果。
  - 「task vector は概ね直交」という観察と「addition で干渉が少ない」現象を結びつけた説明が後続研究の取っ掛かりとして良い。
- **弱み / 疑問**:
  - 著者自身が認めている通り、**同じ architecture かつ同じ pre-trained 初期化から fine-tune された vector しか足し引きできない**（discussion §Limitations）。git-rebasin 系で緩和は示唆されているが、現状の本論文の枠内では強い前提。
  - scaling 係数 $\lambda$ を held-out validation で決める必要があり、本当に「データ不要」なのは analogy のうちでもラベルなし target がアクセス可能な前提のとき。完全に target task のデータがゼロの場合の挙動が不明瞭。
  - 「task vector がほぼ直交」の議論は CLIP の 8 視覚タスクから観察された傾向であり、NLP や同系統タスク群（例: GLUE 内の類似タスク）では同じ直交性が成り立つか定量的に示されていない。GLUE で +0.5pt しか伸びないのも、そもそも task vector 同士が直交していないことの裏返しかもしれない。
  - Negation の安全用途（脱 toxic）で評価指標が Detoxify 一本依存。Detoxify 自体のバイアス（偽陽性/偽陰性）が結果に乗っており、人手評価や別の toxicity 分類器との突き合わせが無い。また「toxic を負化」しても本当に意味理解レベルで毒性が消えたのか、表層トリガーが避けられただけなのかが Table 2 だけでは切り分けられない。
  - Addition のマルチタスク結果は「正規化精度」表記で示され、絶対精度では fine-tuned 単独モデルに 1pt 以上負けている可能性がある（8 タスク平均 91.2% normalized = 単独 fine-tune 比 −8.8pt）。これを「マルチタスクモデル」として使うかは用途次第。
  - $\lambda$ 探索コスト、HF Hub から 427 チェックポイントを試すコストなどが「学習不要」と謳う割に表面化されていない。1 候補あたり数秒の評価でも 427 × 4 タスク × $\lambda$ グリッドで非自明。
  - analogy の subpopulation 実験は「125 overlapping classes」を 2 分割した人工設定で、自然な少数派サブグループでの再現性は本論文の範囲外。
- **次に試したいこと**:
  - **task vector の sparsification / 低ランク化**: 重み差分の上位成分だけ残せば干渉が更に減るのでは（後の TIES-Merging / DARE と直接つながる仮説）。
  - 同一プリトレからの BERT-base ファミリーで HF Hub から「ライブラリ」を構築し、unseen タスクに対し最も近い task vector を取り出すゼロショット転移として使えるか。
  - $\lambda$ をスカラーでなく層ごとに自動決定（Fisher 重み加重平均 etc.）したときの multi-task 加算の改善幅。
  - Negation を **policy 制約**として捉え直し、RLHF の差分（RLHF後 − SFT）を negate して RLHF 部分だけ取り消す/強める実験。
  - 中間 task vector（Fig. 6）が完全 vector と等価に近いなら、fine-tune を early stop して "cheap task vector" を量産し、addition でアンサンブル代替にできるか。
  - task vector の cos sim を「タスク類似度の代理」として task embedding として使い、未知タスクへの転移性予測と組み合わせる。

## Notes / Quotes

- 定義: $\tau_t = \theta_\textrm{ft}^t - \theta_\textrm{pre}$、適用: $\theta_\textrm{new} = \theta + \lambda \tau_\textrm{new}$、$\lambda$ は held-out で決定（§2）。
- "Negating a task vector decreases performance on the target task, with little change in model behavior on control tasks." (abstract)
- ViT-L/14 で negation → target Pre-trained 64.8 → Negative 19.0（−45.8pt、Fine-tuned 比較値は 94.0）、control（ImageNet）75.5 → 72.9（Table 1）。
- GPT-2 Large で toxic 4.8%→0.8%、avg toxicity 0.06→0.01、WikiText-103 ppl 16.4→16.9（Table 2）。Toxicity の task vector 学習は Civil Comments の toxicity > 0.8 で fine-tune したものを negate。
- CLIP 画像 8 タスクで 2 vector 加算 = 専門 fine-tune 2 個と比較し 98.9% 正規化精度、8 vector 全加算で 91.2%（§4.1）。
- T5-base + HF Hub 427 candidates: GLUE 平均 78.1→78.6（Table 3）。$\lambda$ と best checkpoint を held-out で選ぶ。
- Sentiment analogy: $\hat{\tau}_\textrm{yelp;sent} = \tau_\textrm{amazon;sent} + (\tau_\textrm{yelp;lm} - \tau_\textrm{amazon;lm})$、sentiment 側の $\lambda$ を LM 側より高く設定（§5）。
- "Task vectors are typically close to orthogonal" – CLIP の cos sim 観察、digit 系（MNIST/SVHN/GTSRB）と satellite 系（EuroSAT/RESISC45）だけが顕著に高い（Fig. 5, §6）。
- 学習率は低めを推奨、task arithmetic は LR 増に対し fine-tune より急に劣化する（Fig. lr, §6）。
- 中間 task vector は早期に最終 vector の方向に収束する（Fig. intermediate, §6）。
- 著者明記の限界: 同一 architecture かつ同一 pre-trained 初期化からの fine-tune に限定（§6 Limitations）。HF Hub に 3,000+ BERT-base, 800+ T5-small の同一初期化チェックポイントがあるため実用上は致命傷ではない、とも書いている。
- コード: https://github.com/mlfoundations/task_vectors
- (verified 2026-05-20) Summary §結果 / Notes ViT-L/14 行を訂正: 「Fine-tuned 94.0 → Negative 19.0 で −45.8pt」は誤り。Table 1 と 03_negation.tex の caption が示す通り −45.8pt は Pre-trained 64.8 → Negative 19.0 の差。Fine-tuned 94.0 は別の baseline。
- (verified 2026-05-20) Takeaway の「Fig. 6（intermediate task vector）」を Fig. 7 に修正。06_discussion.tex の図順では Fig 5=cossim, Fig 6=lr, Fig 7=intermediate（label: fig:intermediate）。

## Related Papers

- Ilharco+ 2022, *Patching open-vocabulary models by interpolating weights* — 同著者の直接の前身、単一タスクの重み補間。本論文は補間→外挿/加算/類推へ拡張。
- Matena & Raffel 2021, *Merging models with Fisher-weighted averaging* — マルチタスクの重み平均。本論文の addition の baseline 的位置。
- Wortsman+ 2022, *Model soups* — fine-tuned モデル群の平均化で精度向上。直交性議論と LR 議論が直接接続。
- Frankle+ 2020 (Linear mode connectivity), Izmailov+ 2018 (SWA) — 重み補間が機能する条件の理論的・実証的基盤。
- Ainsworth+ 2022, *Git Re-Basin* — 異初期化でも置換で繋げる、limitations 緩和の方向。
- Subramani+ 2022 — hidden state にベクトルを足して LM を steer。重み空間 vs 表現空間の対比相手。
- Radford+ 2021 *CLIP*, Raffel+ 2020 *T5*, Radford+ 2019 *GPT-2* — backbone モデル。
- Borkan+ 2019 *Civil Comments*, Detoxify — toxic generation 評価設定。
- 後続として TIES-Merging, DARE, Model Soups 系の "task vector を疎にする/正規化する" 研究群が直接の発展（本論文の Fig. 5 直交性観察が動機）。
