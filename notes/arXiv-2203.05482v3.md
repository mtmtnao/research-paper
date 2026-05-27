# Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time

- arXiv: https://arxiv.org/abs/2203.05482
- source: ../papers/arXiv-2203.05482v3/
- authors: Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, Ludwig Schmidt
- venue / year: ICML 2022
- tags: [weight-averaging, fine-tuning, CLIP, robustness, transfer-learning]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 大規模事前学習モデルを fine-tune する際の慣行は「複数の hp 構成で学習 → val 最良の1個を採用 → 残りは捨てる」。捨てた重みも情報を持っており、ensemble すれば精度は上がるが推論コストが線形に増える。さらに val 最良モデルが OOD で最良とは限らない（CLIP のような事前学習モデルでは fine-tuning が頑健性を毀損することが知られている）。
- **手法**: 同じ事前学習初期値 $\theta_0$ から異なる hp で fine-tune した複数モデルの **重みを平均** して 1 つのモデルとする＝ "model soup"。3 種類:
  - *uniform soup*: 全モデルを単純平均
  - *greedy soup* (Recipe 1, 本命): val 精度降順に並べ、加えると val 精度が落ちないモデルのみ逐次的に追加
  - *learned soup* (付録): val 上で混合係数を勾配法で学習（全モデルをメモリに載せる必要があり大規模には不向き）
  推論コストは単一モデルと同じ $\mathcal{O}(1)$。同じ basin にある fine-tuned 解は線形補間しても精度が落ちないという Neyshabur+ 2020 の観察が基礎。
- **結果**:
  - **ViT-G/14 (JFT-3B 事前学習) を ImageNet で fine-tune** (`tab:vit_g_finetuned_result`): greedy soup が **90.94% top-1**（58 個中 14 個を選択）。当時の SOTA CoAtNet-7 の 90.88% を超え、25% 少ない FLOPs。val 最良単体は 90.72% / Avg shifts 84.38、oracle (各 test set 別の単体最良) は 90.78 / 84.68、greedy soup は 90.94 / **85.02**。後発実験で **BASIC-L** でも 90.98% を達成 (`tab:basic_finetuned_result`; footnote では、BASIC-L 90.98% が CoCa の報告精度と同等と述べる)。
  - **CLIP ViT-B/32 を ImageNet で fine-tune** (72 モデル random search, `tab:results`): 単体最良 80.38 / 47.83 (ImageNet / Dist. shifts 平均) → uniform soup 79.97 / 51.45、greedy soup **81.03 / 50.75**。logit ensemble は 81.19 / 50.77、greedy ensemble は 81.90 / 49.44。本文は、ImageNet では ensemble が高い一方、distribution shifts では reverse true と述べる (`fig:ose_wse_compare`)。
  - **ALIGN EfficientNet-L2**: 12 モデルから greedy soup が単体最良比 +0.5pp（CLIP は +0.7pp）。
  - **NLP (GLUE, BERT/T5)** (`tab:nlp`): T5 は MRPC 91.8→92.4, RTE 78.3→79.1, CoLA 58.8→60.2, SST-2 94.6→94.7。BERT は MRPC 88.3→88.3, RTE 61.0→61.7, CoLA 59.1→59.1, SST-2 92.5→93.0。改善は画像ほどではないが多くで正方向。
  - **cross-dataset soup** (`sec:zsperf`): CLIP zero-shot + CIFAR-10/DTD/Food-101/SUN397/Cars/ImageNet の 6 fine-tuned を soup して **CIFAR-100 zero-shot で +6.4pp**（CLIP baseline 比）。
  - **理論** (§4): soup と logit ensemble の損失差を $\frac{\alpha(1-\alpha)}{2}(-\partial^2_\alpha \mathcal{L}^{\text{soup}} + \beta^2 \mathbb{E}_x\,\mathrm{Var}_{Y\sim p_\text{sftmx}}[\Delta f_Y])$ で近似。**(a) endpoint が近い**または **(b) 予測が confident** な場合に variance term が小さくなる、と説明。
- **貢献**: (1) 異なる hp で独立 fine-tune したモデル群を「重み平均」だけで使い切る簡潔な recipe（特に greedy soup）、(2) ImageNet 90.94% という当時 SOTA、(3) ID/OOD/転移/NLP/zero-shot 横断で有効性を実証、(4) soup と ensemble の差を flatness × confidence で説明する解析、(5) WiSE-FT と直交し組み合わせ可能であることの提示 (`sec:robustft`)。

## Takeaway（自分にとっての要点）

- **「捨てていた hp sweep の副産物を再利用する」** だけで SOTA を更新できる、というメッセージの強さ。TeX では、既にある hyperparameter sweep の結果から model soup を作るには追加学習が不要で、推論時の追加コストもないと述べている。
- *greedy soup* は val 精度降順に sort してから足すので「**held-out validation set 上では val 最良単体より悪くならない**」と本文で述べられている。uniform soup は high-lr の外れ値で崩れる場合があり、CLIP では 79.97 で単体最良 80.38 を下回る。
- soup が効く条件として、本文は同じ pre-trained initialization から独立 fine-tune されたモデルが同じ error basin にあるという Neyshabur+ 2020 の観察を挙げる。`fig:angles` (`angles.pdf`) は、interpolation advantage が角度 $\phi$ と相関し、varying learning rate, seed, or data augmentation がより直交な解を生むと述べる。
- soup と ensemble の理論的な見方が明示的: loss difference の近似では、endpoint models が近い場合、または soup が confident predictions を出す場合に variance term が小さいと説明される。
- **WiSE-FT との重畳**: soup と zero-shot 初期値を更に線形補間すると OOD がさらに改善 (`sec:robustft`, `robust_ft_small.pdf`)。つまり「hp で多様性 → 平均 → 初期値方向に少し引き戻す」の二段構え。
- **限界 (著者自身が `sec:lim` で明示)**:
  - 大規模・多様な事前学習が前提。ImageNet-22k 事前学習だと改善幅が "less substantial" になる（同じ hp sweep で `fig:in22k` / `in22k.pdf`）。
  - **calibration は改善しない**。"model soups do not have the same effect" on calibration（`fig:cal` / `varied_seed.pdf`、同 hp・seed 違い 20 モデルで ECE を比較）。
- 実装メモ: ViT-G では EMA 2 系統 (decay 0.999 / 0.9999999) を保存し、「単体最良は high-EMA、soup/ensemble は low-EMA」が良い、と細かい運用ノウハウが書かれている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 推論コスト不変・追加学習ゼロ・既存パイプラインに後付け可能、というデプロイ可能性の高さ。
  - ID と OOD の両方で改善するうえ、greedy soup は test を見ずに val だけで構成される。`tab:vit_g_finetuned_result` で oracle (各 test set 別の単体最良) すら Avg shifts 85.02 vs 84.68 で soup が上回る点は強い。
  - hp sweep を "もうやってある" 研究者にはサンクコスト回収の現実的な手段。72 / 58 / 32 モデルといった具体的なスケール感が示されている。
  - 理論パート (§4) が「soup が ensemble に追いつく条件」を flatness と confidence の 2 項で表現していて、なぜ高 lr で崩れるか・なぜ ensemble が OOD で勝つときがあるかの定性説明と整合する。
- **弱み / 疑問**:
  - greedy soup は validation set accuracy に基づいて候補を順に追加する。Table `tab:methods` は inference cost を比較するが、greedy soup 構築時の validation 評価コストは別途の論点としては詳述していない（TeX 中には明示されていない / 評者補足）。
  - hp sweep そのものは依然必要。Table `tab:methods` は all methods require the same training と述べるが、計算予算が限られた状況で「同じ予算を soup と 1 モデルにどう配分するか」の比較は TeX 中には明示されていない。
  - 異なる random seed のみのモデルでは、`fig:in22kanalysis` / `analysis0.pdf` が CLIP と ImageNet-22k pre-trained ViT-B/32 の seed 変化のみの結果を示す。本文は CLIP models are more amenable to both ensembling and souping than models pre-trained on ImageNet-22k と述べる。
  - NLP 実験は著者自身が "preliminary experiments" と述べ、"more investigation is warranted" としている。BERT MRPC/CoLA は 88.3→88.3 / 59.1→59.1 で改善しない。
  - calibration が改善しないという限界は、著者自身が `sec:lim` で明示している。
  - 理論近似は **logit が線形に近い領域** で導出されている。著者自身、`fig:theory-eval` の説明で「高 lr の $10^{-4}$ のモデルは初期値から weight space で遠く、近似がタイトでないと予想されるため除外する」と述べており、高 lr 領域で近似が外れることを明示している。
- **次に試したいこと**:
  - 同一計算予算で「greedy soup vs 1 モデルを長く学習 vs SWA vs SAM」の比較を行う（TeX 中には明示されていない / 評者補足）。論文は Appendix で SWA/SAM/EMA とは比較している。
  - hp diversity を意図的に設計する研究: Appendix `app:nlp_ft` では、NLP 実験について broader set of hyperparameter choices が more diverse models and better soups につながる可能性を著者が hypothesize している。
  - calibration が落ちない soup の派生を検討する（TeX 中には明示されていない / 評者補足）。
  - cross-dataset soup (`sec:zsperf`) の延長として、fine-tuned models の含め方が soup 精度に大きく影響する点をさらに調べる。

## Notes / Quotes

- "we average the weights of models fine-tuned independently, and refer to the result as a *model soup*." (Introduction, main.tex L211)
- "the greedy soup can be no worse than the best individual model on the held-out validation set." (Method §2, L304-305)
- "When fine-tuning large pre-trained models such as CLIP, ALIGN, and a ViT-G pre-trained on JFT, our soup recipe provides significant improvements over the best model in a hyperparameter sweep on ImageNet." (Abstract)
- ViT-G/14 結果: greedy soup 90.94 / Avg shifts 85.02, val 最良 90.72 / 84.38, oracle 90.78 / 84.68 (`tab:vit_g_finetuned_result`)
- CLIP ViT-B/32 (`tab:results`): best 80.38/47.83, uniform 79.97/51.45, greedy soup 81.03/50.75, ensemble 81.19/50.77, greedy ensemble 81.90/49.44
- ensemble との解析的関係: $\mathcal{L}^{\text{soup}}_\alpha - \mathcal{L}^{\text{ens}}_\alpha \approx \frac{\alpha(1-\alpha)}{2}(-\partial^2_\alpha \mathcal{L}^{\text{soup}} + \beta^2 \mathbb{E}_x \mathrm{Var}_{Y\sim p_\text{sftmx}}[\Delta f_Y])$ (Eq. 1)
- "model soups do not have the same effect" on calibration as ensembles (§5 Calibration)
- ImageNet-22k 事前学習では gain が縮む: "the improvement is less substantial" (`sec:lim` Applicability, `fig:in22k` / `in22k.pdf`)
- 角度 $\phi$ と interpolation advantage の相関 (`fig:angles` / `angles.pdf`): hp を散らすほど $\phi$ が大きくなり soup の利得も上がる。
- greedy soup が CLIP/ALIGN で選んだ ingredient 数: 各 5 個 / sweep 中で（CLIP は 72 から、ALIGN は 12 から）。ViT-G は 58 から 14 個。
- (verified 2026-05-20) val 最良 ViT-G を 90.78→90.72 に修正、oracle を 90.78 と分離 (main.tex `tab:vit_g_finetuned_result`)
- (verified 2026-05-20) "Table 1/2/3"・"Figure 4"・"Figure C2"・"付録 C/D" 等の番号参照を TeX label 参照に置換（番号は LaTeX 出力依存で実体と一致しない可能性があり、label の方が確実）
- (verified 2026-05-20) BERT MRPC/CoLA を "±0" から具体値 (88.3→88.3, 59.1→59.1) に明示 (main.tex `tab:nlp`)
- (verified 2026-05-20) cross-dataset soup の対象データセット 6 種 (CIFAR-10/DTD/Food-101/SUN397/Cars/ImageNet, eval = CIFAR-100) を明示 (main.tex `sec:zsperf`)
- (verified 2026-05-27) TeX にない「1行足すだけ」「実運用は基本 greedy 一択」「ensemble の方が安全」「LoRA/adapter」等の断定・外挿を削除または評者補足に修正 (main.tex `sec:method`, `sec:lim`, `app:nlp_ft`)
- (verified 2026-05-27) BASIC-L と CoCa の比較表現を footnote/table に合わせて「CoCa の報告精度と同等」に修正 (main.tex footnote, `tab:basic_finetuned_result`)
- (verified 2026-05-27) 理論要約の「両者が一致」という強い表現を、variance term が小さいという TeX の説明に合わせて弱めた (main.tex `sec:theory`)

## Related Papers

- Neyshabur et al. 2020 — fine-tuned models from a shared init live in the same loss basin（本論文の理論的支柱）
- Izmailov et al. 2018, SWA — 単一軌道上の重み平均。本論文は「独立 run 間の平均」に拡張。
- Frankle et al. 2020 (Linear Mode Connectivity) — 同じ初期値・異なるデータ順での補間は accuracy が落ちる、という対比対象。
- Wortsman et al. 2021, WiSE-FT — fine-tune 後モデルと zero-shot 初期値の補間。soup と組み合わせ可能 (`sec:robustft`)。
- Matena & Raffel 2021 — Fisher 情報で異タスク fine-tune モデルを merging（model merging 系の先行）。
- Gontijo-Lopes et al. 2022 — training methodology の divergence が ensemble 精度に効くという観察（soup 設計の根拠）。
- Radford et al. 2021 CLIP, Jia et al. 2021 ALIGN, Zhai et al. 2021 ViT-G/JFT, Pham et al. 2021 BASIC — fine-tune 対象の事前学習モデル。
- Dai et al. 2021 CoAtNet — 当時 SOTA、本論文が 90.94% で抜いた比較対象。
- Foret et al. 2021 SAM — Appendix で比較される baseline。
