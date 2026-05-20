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
  - **ViT-G/14 (JFT-3B 事前学習) を ImageNet で fine-tune** (`tab:vit_g_finetuned_result`): greedy soup が **90.94% top-1**（58 個中 14 個を選択）。当時の SOTA CoAtNet-7 の 90.88% を超え、25% 少ない FLOPs。val 最良単体は 90.72% / Avg shifts 84.38、oracle (各 test set 別の単体最良) は 90.78 / 84.68、greedy soup は 90.94 / **85.02**。後発実験で **BASIC-L** でも 90.98% を達成 (`tab:basic_finetuned_result`, footnote: CoCa 90.98% と同等)。
  - **CLIP ViT-B/32 を ImageNet で fine-tune** (72 モデル random search, `tab:results`): 単体最良 80.38 / 47.83 (ImageNet / Dist. shifts 平均) → uniform soup 79.97 / 51.45、greedy soup **81.03 / 50.75**。logit ensemble は 81.19 / 50.77、greedy ensemble は 81.90 / 49.44。soup は ID で ensemble に近く、OOD では ensemble を上回りやすい。
  - **ALIGN EfficientNet-L2**: 12 モデルから greedy soup が単体最良比 +0.5pp（CLIP は +0.7pp）。
  - **NLP (GLUE, BERT/T5)** (`tab:nlp`): T5 は MRPC 91.8→92.4, RTE 78.3→79.1, CoLA 58.8→60.2, SST-2 94.6→94.7。BERT は MRPC 88.3→88.3, RTE 61.0→61.7, CoLA 59.1→59.1, SST-2 92.5→93.0。改善は画像ほどではないが多くで正方向。
  - **cross-dataset soup** (`sec:zsperf`): CLIP zero-shot + CIFAR-10/DTD/Food-101/SUN397/Cars/ImageNet の 6 fine-tuned を soup して **CIFAR-100 zero-shot で +6.4pp**（CLIP baseline 比）。
  - **理論** (§4): soup と logit ensemble の損失差を $\frac{\alpha(1-\alpha)}{2}(-\partial^2_\alpha \mathcal{L}^{\text{soup}} + \beta^2 \mathbb{E}_x\,\mathrm{Var}_{Y\sim p_\text{sftmx}}[\Delta f_Y])$ で近似。**(a) endpoint が近い**または **(b) 予測が confident** な領域では両者が一致 → soup が ensemble にどれだけ迫れるかを説明。
- **貢献**: (1) 異なる hp で独立 fine-tune したモデル群を「重み平均」だけで使い切る簡潔な recipe（特に greedy soup）、(2) ImageNet 90.94% という当時 SOTA、(3) ID/OOD/転移/NLP/zero-shot 横断で有効性を実証、(4) soup と ensemble の差を flatness × confidence で説明する解析、(5) WiSE-FT と直交し組み合わせ可能であることの提示 (`sec:robustft`)。

## Takeaway（自分にとっての要点）

- **「捨てていた hp sweep の副産物を再利用する」** だけで SOTA を更新できる、というメッセージの強さ。学習側のコードは一切変えずに最終段に平均を 1 行足すだけ。
- *greedy soup* は val 精度降順に sort してから足すので「**最悪でも val 最良単体と同等**」が保証される。uniform soup は high-lr の外れ値で簡単に崩れる（CLIP では 79.97 で単体最良 80.38 を下回る）ので、実運用は基本 greedy 一択。
- soup が効く条件は intuitively **同じ basin**（同じ $\theta_0$、暴れすぎない lr）。`fig:angles` (`angles.pdf`) は「異なる hp ほど解がより直交 ($\phi$ が大) になり、interpolation advantage が大きい」を実証。論文文中では varying learning rate, seed, or data augmentation がより直交な解を生むと述べており、seed 変化だけでなく lr/aug も多様性源として効く。
- soup ≠ ensemble の理論的な見方が明示的: ensemble は常に **flat + confident** な領域では soup と一致する。逆に言えば「ensemble はしたいがコストが許さない」場面の代替として soup を選ぶ判断基準ができる。
- **WiSE-FT との重畳**: soup と zero-shot 初期値を更に線形補間すると OOD がさらに改善 (`sec:robustft`, `robust_ft_small.pdf`)。つまり「hp で多様性 → 平均 → 初期値方向に少し引き戻す」の二段構え。
- **限界 (著者自身が `sec:lim` で明示)**:
  - 大規模・多様な事前学習が前提。ImageNet-22k 事前学習だと改善幅が "less substantial" になる（同じ hp sweep で `fig:in22k` / `in22k.pdf`）。
  - **calibration は改善しない**。"model soups do not have the same effect" on calibration（`fig:cal` / `varied_seed.pdf`、同 hp・seed 違い 20 モデルで ECE を比較）。confidence が必要な応用では ensemble の方が安全。
- 実装メモ: ViT-G では EMA 2 系統 (decay 0.999 / 0.9999999) を保存し、「単体最良は high-EMA、soup/ensemble は low-EMA」が良い、と細かい運用ノウハウが書かれている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 推論コスト不変・追加学習ゼロ・既存パイプラインに後付け可能、というデプロイ可能性の高さ。
  - ID と OOD の両方で改善するうえ、greedy soup は test を見ずに val だけで構成される。`tab:vit_g_finetuned_result` で oracle (各 test set 別の単体最良) すら Avg shifts 85.02 vs 84.68 で soup が上回る点は強い。
  - hp sweep を "もうやってある" 研究者にはサンクコスト回収の現実的な手段。72 / 58 / 32 モデルといった具体的なスケール感が示されている。
  - 理論パート (§4) が「soup が ensemble に追いつく条件」を flatness と confidence の 2 項で表現していて、なぜ高 lr で崩れるか・なぜ ensemble が OOD で勝つときがあるかの定性説明と整合する。
- **弱み / 疑問**:
  - greedy soup は **val を sweep モデル数だけ評価する** ので、巨大モデル＋多数候補だと val コストが効いてくる（学習に比べれば小さいが）。論文は計算量の議論を簡略化している。
  - hp sweep そのものは依然必要。greedy soup が良いのは「sweep を回せる人」に限定された朗報で、計算予算が限られた状況では「同じ予算を soup と 1 モデルにどう配分するか」の比較がない。
  - 異なる random seed のみのモデルでは soup の伸びが小さい (`fig:in22kanalysis` / `analysis0.pdf`、CLIP/ImageNet-22k 双方で seed のみ変化させた fine-tune 5 本)。多様性の源を意図的に設計する必要があるが、論文の hp sweep は基本的に既存実験の流用で、「soup を効かせるための sweep 設計」は将来課題のまま。
  - NLP の改善幅が小さい (e.g. BERT MRPC/CoLA は ±0)。本人達も "preliminary" と認めているが、tokens 数の少ない GLUE タスクで basin が浅いのか、それとも事前学習スケールが画像ほど大きくない（BERT は CLIP 級でない）からなのか、要因が切り分けられていない。
  - calibration が改善しないという限界は応用上痛い。安全性が要る領域では soup よりは ensemble or soup + 温度補正 + α、という選択が必要。
  - 理論近似は **logit が線形に近い領域** で導出されている。著者自身、`fig:theory-eval` の説明で「高 lr の $10^{-4}$ のモデルは初期値から weight space で遠く、近似がタイトでないと予想されるため除外する」と述べており、高 lr 領域で近似が外れることを明示している。
- **次に試したいこと**:
  - 同一計算予算で「greedy soup vs 1 モデルを長く学習 vs SWA vs SAM」の pareto を引く。論文は Appendix で SWA/SAM/EMA とは比較しているが、**学習 FLOPs を揃えた比較ではない**。
  - hp diversity を意図的に設計する研究: 「どんな軸 (lr×aug×optimizer) を散らせば $\phi$ が最大化されるか」を方策化できれば、sweep そのもののコストを下げられそう。
  - calibration が落ちない soup の派生（例: temperature-scaled logits 空間で soup を構成、あるいは soup + 後段 ensemble の hybrid）。
  - LoRA や adapter の重みに対する soup。base model が共通なので原理的にも soup の前提が満たされやすい（model merging 系研究の発展形）。
  - cross-dataset soup (`sec:zsperf`) の延長として、「downstream task 毎に専門 fine-tune した重みを動的に重みづけて soup」する task-conditional soup。

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

## Related Papers

- Neyshabur et al. 2020 — fine-tuned models from a shared init live in the same loss basin（本論文の理論的支柱）
- Izmailov et al. 2018, SWA — 単一軌道上の重み平均。本論文は「独立 run 間の平均」に拡張。
- Frankle et al. 2020 (Linear Mode Connectivity) — 同じ初期値・異なるデータ順での補間は accuracy が落ちる、という対比対象。
- Wortsman et al. 2021, WiSE-FT — fine-tune 後モデルと zero-shot 初期値の補間。soup と組み合わせ可能 (`sec:robustft`)。
- Matena & Raffel 2021 — Fisher 情報で異タスク fine-tune モデルを merging（model merging 系の先行）。
- Gontijo-Lopes et al. 2021 — hp diversity が ensemble 精度に効くという観察（soup 設計の根拠）。
- Radford et al. 2021 CLIP, Jia et al. 2021 ALIGN, Zhai et al. 2021 ViT-G/JFT, Pham et al. 2021 BASIC — fine-tune 対象の事前学習モデル。
- Dai et al. 2021 CoAtNet — 当時 SOTA、本論文が 90.94% で抜いた比較対象。
- Foret et al. 2021 SAM — Appendix で比較される baseline。
