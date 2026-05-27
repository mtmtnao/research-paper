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

- **問題**: 事前学習済みモデルの振る舞いを後から「編集」したい場面（下流タスクの精度向上、バイアス除去、望ましくない挙動の緩和、人間選好との整合、新情報への更新）は多い。著者は、task vector による単純・高速・効果的なモデル編集の枠組みを提案する。
- **手法**: タスク $t$ に対する **task vector** を $\tau_t = \theta_\textrm{ft}^t - \theta_\textrm{pre}$（fine-tune 後の重み − 事前学習重み、要素ごと差）として定義し、$\theta_\textrm{new} = \theta + \lambda \tau_\textrm{new}$ で適用する（$\lambda$ は held-out 検証集合で決定）。task vector に対し3つの算術演算を定義: (1) **negation** $-\tau$（忘却/挙動抑制）、(2) **addition** $\sum_i \tau_i$（マルチタスク化や単一タスク強化）、(3) **analogy** $\tau_C + (\tau_B - \tau_A)$（"A:B :: C:D" の関係で D の training data を使わず改善）。要素ごとの重み演算のみで、追加学習も推論時オーバーヘッドも無い。
- **結果**:
  - *Negation – 画像*: CLIP ViT-L/14、8 タスク平均で Pre-trained 64.8% → Negative task vector 19.0%（−45.8pt の忘却。参考までに Fine-tuned は 94.0%）。ImageNet コントロールは Pre-trained 75.5 → Negative 72.9（−2.6pt）。Gradient ascent は target 3.93%/control 16.3% でコントロール崩壊、Random vector は forget できず 60.9%/72.9%（Table 1）。
  - *Negation – 言語*: GPT-2 Large で toxic generation 比率を 4.8% → 0.8%（6 倍減）、avg toxicity 0.06→0.01、WikiText-103 perplexity は 16.4→16.9（+0.5）。Fine-tuned on non-toxic（1.8% / 17.2 ppl）より両指標で優れる。Gradient ascent は ppl > 10$^{10}$ で破綻（Table 2）。
  - *Addition – 画像*: CLIP の 8 タスクから 2 つを足すと、専門 fine-tuned 2 モデル使用時の **98.9%** 正規化精度に到達。$2^8$ 通りの部分集合を試行し、全 8 タスクを足した最良モデルは平均 91.2% を維持（Fig. 2-3, §4.1）。
  - *Addition – NLP*: T5-base を GLUE 4 タスクで fine-tune → さらに Hugging Face Hub 上の 427 候補 task vector を加算。Avg 78.1 → 78.6（MRPC +0.8, RTE +0.2, CoLA +0.7, SST-2 +0.2、Table 3）。
  - *Analogy – ドメイン汎化*: Yelp/Amazon の sentiment を、もう一方の sentiment ラベル + 両ドメインの教師なし LM の3 vectors で構成。T5-large で Yelp 95.0→95.1（target fine-tune 95.5）、Amazon 94.8→95.2（target 95.5）。auxiliary fine-tune を常に上回り target fine-tune に肉薄（Table 4）。
  - *Analogy – サブ集団*: ImageNet × 人手スケッチで「real/sketch × 2 クラス群」の 4 区分を作成、3 区分の vector から 4 つ目を合成 → pre-trained に対し平均 +3.4pt、これは新規アノテーション約 100 サンプル分の効果に相当（Fig. 4）。
- **貢献**: (1) 重み空間での「タスクベクトル算術」という統一的編集枠組み（negation/addition/analogy）の提案、(2) CLIP（vision）・T5（NLP）・GPT-2（LM）の複数モダリティ・モデル規模での効果実証、(3) 追加学習なし・推論コストなしでマルチタスク化・忘却・ラベルなし target task へのドメイン汎化を実現する具体的レシピ、(4) 公開 Hugging Face Hub チェックポイントを task vector として利用できることの提示。

## Takeaway（自分にとっての要点）

- **task vector は同じ initialization から fine-tune された重みの差分**という定義で、「重みを足し引きするだけで挙動を編集できる」というのが本質。TeX では、この操作で 6× toxic 削減や 8 タスク 91.2% normalized accuracy が示されている。
- 公開チェックポイントは task vector として再利用できる候補になる。Hugging Face Hub には、同じ BERT-base initialization から fine-tune された 3,000 超のモデル、同じ T5-small initialization から fine-tune された 800 超のモデルがあると discussion の limitations に書かれている。
- **Negation が忘却に効くがコントロール劣化が小さい**点は実用的に重要。gradient ascent は Table 1/2 で control task を大きく悪化させる一方、negation は ImageNet を 2.6pt 下げて target を 45.8pt 落とし、GPT-2 Large では toxic generations を 6× 削減しつつ WikiText-103 perplexity 増加を +0.5 に抑えている。
- **task vector 同士が概ね直交**（Fig. 5 cos sim）であることが addition で干渉が小さい理由として提示されている。意味的に近いタスク（MNIST/SVHN/GTSRB の digit 系、EuroSAT/RESISC45 の衛星系）だけ高 cos sim。これは「ベクトル算術」というメタファーが weight space で実際に成立する条件として重要。
- **学習率に敏感**: task vector を使う場合は fine-tune 単体より低 LR を推奨。NLP で他者のチェックポイントを混ぜると分散が大きい理由として LR を疑っている（discussion）。
- **中間 task vector が早期に方向収束**（Fig. 7 = fig:intermediate）→ 完全 fine-tune を待たずに加算でき、計算節約の余地がある。
- analogy で「教師なし LM 由来の差分」を使うと、sentiment ベクトルを別ドメインに転写できる結果になっている（Table 4）。これを weight space の差が "ドメイン差" を持つという示唆として読むのは評者補足。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 提案操作は要素ごとの足し算・引き算で、追加の推論コストが無い。コード URL も TeX に明記されている。
  - vision（CLIP, 3 サイズ）・NLP（T5, 3 サイズ）・LM（GPT-2）の3モダリティで一貫して効くことを示しており、特定モデル依存ではない。
  - negation を「gradient ascent」「random vector」「non-toxic fine-tune」と比較しており、Table 1/2 で target と control の両方を示している。
  - analogy 実験で T5-large の sentiment が "ターゲットの fine-tune とほぼ同じ" まで来るのは（95.1 vs 95.5）、ラベルなしドメインへの転写としてはかなり強い結果。
  - 「task vector は概ね直交」という観察と「addition で干渉が少ない」現象を結びつけている（Fig. 5, §6）。
- **弱み / 疑問**:
  - 著者自身が認めている通り、**同じ architecture かつ同じ pre-trained 初期化から fine-tune された vector しか足し引きできない**（discussion §Limitations）。git-rebasin 系で緩和は示唆されているが、現状の本論文の枠内では強い前提。
  - scaling 係数 $\lambda$ を held-out validation で決める必要がある。Appendix では $\lambda \in \{0.0, 0.05, 0.1, \cdots, 1.0\}$ や $\lambda \in \{0, 0.1, \cdots, 1.0\}$ の探索が使われている。
  - 「task vector がほぼ直交」の議論は CLIP の視覚タスクで示された観察であり、NLP task vector について同じ cosine similarity 図は TeX 中に示されていない。
  - Toxicity 評価は、Table 2 では Detoxify による 1000 generations、Appendix では RealToxicityPrompts + Perspective API も示されている。一方で、人手評価は TeX 中に明確な記述なし。
  - Addition のマルチタスク結果は「正規化精度」表記で示される。Appendix の multi-task training 比較では、8 タスク jointly fine-tuned model が 0.994、task vector の best result が 0.912。
  - $\lambda$ 探索コストや HF Hub から 427 checkpoint を試す wall-clock cost は TeX 中に明確な数値なし。著者は evaluation が training より安い、scaling coefficient は追加学習なしに変えられる、とは述べている。
  - analogy の subpopulation 実験は「125 overlapping classes」を 2 分割した人工設定で、自然な少数派サブグループでの再現性は本論文の範囲外。
- **次に試したいこと**:
  - **task vector の sparsification / 低ランク化**: 重み差分の一部だけ残したときに干渉が減るかを調べる（評者補足）。
  - 同一プリトレからの BERT-base ファミリーで HF Hub から「ライブラリ」を構築し、unseen タスクに対し最も近い task vector を取り出すゼロショット転移として使えるか（評者補足）。
  - $\lambda$ をスカラーでなく層ごとに自動決定したときの multi-task 加算の改善幅（評者補足）。
  - 中間 task vector（Fig. 7）が完全 vector に近い方向へ早期に収束するなら、fine-tune を early stop して "cheap task vector" を量産できるか（評者補足）。
  - task vector の cos sim を「タスク類似度の代理」として task embedding として使い、未知タスクへの転移性予測と組み合わせる（評者補足。Related work §Task embeddings に近い方向）。

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
- (verified 2026-05-27) TeX 根拠より強い推測（未検証の応用先、後続研究名、toxicity 評価の Detoxify 一本依存など）を削除または評者補足として明示 (03_negation.tex, 04_addition.tex, 06_discussion.tex, 97_related_work.tex, 99_appendix.tex)
- (verified 2026-05-27) Critical Thoughts の $\lambda$ 探索、multi-task training 比較、toxicity 追加評価を Appendix の記述に合わせて修正 (99_appendix.tex)

## Related Papers

- Ilharco+ 2022, *Patching open-vocabulary models by interpolating weights* — 同著者の直接の前身、単一タスクの重み補間。本論文は補間→外挿/加算/類推へ拡張。
- Matena & Raffel 2021, *Merging models with fisher-weighted averaging* — マルチタスクの重み平均。本論文の addition の baseline 的位置。
- Wortsman+ 2022, *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time* — fine-tuned モデル群の平均化で精度向上。直交性議論と LR 議論が直接接続。
- Frankle+ 2020 *Linear mode connectivity and the lottery ticket hypothesis*, Izmailov+ 2018 *Averaging weights leads to wider optima and better generalization* — 重み補間・重み平均が機能する条件の関連研究。
- Ainsworth+ 2022, *Git re-basin: Merging models modulo permutation symmetries* — 異初期化でも置換で繋げる、limitations 緩和の方向。
- Subramani+ 2022, *Extracting latent steering vectors from pretrained language models* — hidden state にベクトルを足して LM を steer。重み空間 vs 表現空間の対比相手。
- Radford+ 2021 *Learning transferable visual models from natural language supervision*, Raffel+ 2020 *Exploring the limits of transfer learning with a unified text-to-text transformer*, Radford+ 2019 *Language Models are Unsupervised Multitask Learners* — backbone モデル。
- Borkan+ 2019 *Nuanced metrics for measuring unintended bias with real data for text classification*, Hanu & Unitary team 2020 *Detoxify* — toxic generation 評価設定。
