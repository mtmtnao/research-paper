# Training Compute-Optimal Large Language Models

- arXiv: https://arxiv.org/abs/2203.15556
- source: ../papers/arXiv-2203.15556v1/
- authors: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre
- venue / year: arXiv preprint 2022（TeX は deepmind.cls 同梱で投稿先の明示なし）
- tags: [scaling-laws, LLM, compute-optimal, pretraining, Chinchilla]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 固定の FLOPs 予算 $C$ の下で、transformer LM の最終 pre-training loss を最小化する最適なパラメータ数 $N$ と訓練トークン数 $D$ の配分は何か。Kaplan+ 2020 の予測 ($a=0.73, b=0.27$、すなわち model size を tokens より約 3 倍速く増やすべき) を前提に、近年の LLM（GPT-3 175B、Jurassic 178B、Gopher 280B、MT-NLG 530B）はいずれも約 300B tokens 前後で訓練されており、大幅に「under-trained」である可能性がある。
- **手法**: 70M〜16B+ パラメータ・5B〜400B+ tokens で 400+ モデルを訓練し、3 つの独立な方法で最適 $N_{opt}(C), D_{opt}(C)$ を推定:
  1. **Approach 1 (training-curve envelope)**: 各モデルサイズで 4 種のコサインサイクル長で訓練し、(FLOPs, loss) 軌跡の包絡から $N_{opt} \propto C^a$, $D_{opt} \propto C^b$ をフィット。$a=0.50, b=0.50$。
  2. **Approach 2 (IsoFLOP profiles)**: 9 種の固定 FLOPs 予算 ($6\times10^{18}$〜$3\times10^{21}$) でモデルサイズを振り、各予算で最終 loss を放物線フィット → 最適 $N$ を抽出。$a=0.49, b=0.51$。
  3. **Approach 3 (parametric loss)**: $\hat L(N,D) = E + A/N^\alpha + B/D^\beta$ を Huber loss + L-BFGS でフィット。$E=1.69, A=406.4, B=410.7, \alpha=0.34, \beta=0.28$ で $a=0.46, b=0.54$。
  この予測（model と data を等比でスケール）に従い、Gopher と同じ計算量 $5.76\times10^{23}$ FLOPs で **Chinchilla 70B**（Gopher の 1/4）を **1.4T tokens**（Gopher の 4 倍）で訓練して検証。アーキは Gopher と同じ Transformer だが、AdamW 採用、NFKC 無し SentencePiece、bfloat16 forward + float32 optimizer state。
- **結果**:
  - 3 手法とも $a \approx b \approx 0.5$ で一致し、Kaplan+ 2020 の $(0.73, 0.27)$ と明確に対立する（Table tab:comparison）。C4・GitHub データセットでも同様 ($a=0.50/0.53$, Table tab:comparison_c4_github)。
  - **MMLU 5-shot 67.6%**（Gopher 60.0%、+7.6%）、人間 forecaster の 2023 年予測 63.4% も上回る。57 タスク中 51 勝 / 2 引き分け / 4 敗 (college_mathematics, econometrics, moral_scenarios, formal_logic)。
  - **BIG-bench 65.1%**（Gopher 54.4%、+10.7%）、62 タスク中 4 敗のみ。
  - **Reading comp**: LAMBADA 77.4 vs Gopher 74.5 / MT-NLG 76.6、RACE-h 82.3 vs 71.6、RACE-m 86.8 vs 75.1。
  - **Common sense (0-shot)**: HellaSWAG 80.8、Winogrande 74.9、BoolQ 83.7、SIQA 51.3、PIQA 81.8（Gopher・GPT-3 をほぼ全勝、MT-NLG 530B にも PIQA 除き勝つ）。
  - **TruthfulQA**: 0-shot 43.6 → 10-shot 66.7（Gopher は 29.5 / 43.7）。
  - **Closed-book QA**: Natural Questions 5-shot 31.5%（SOTA）、TriviaQA filtered 5-shot 64.1。
  - **Pile**: 全 subset で Gopher を bpb で改善。WikiText103 perplexity 7.16 vs 7.75。
  - **Winogender**: 全体 78.3 vs 71.4。female gotcha で +10。male/female で改善幅に差。
  - **Toxicity**: 25,000 unprompted サンプルの PerspectiveAPI 平均 0.087（Gopher 0.081）でほぼ同等 → 「より良い LM ≠ より toxic」を Gopher 論文の知見と整合的に確認。
  - **外挿**: 本文では 175B モデルの最適予算を $4.41\times10^{24}$ FLOPs / 4.2T tokens、280B Gopher 級を約 $10^{25}$ FLOPs / 6.8T tokens と記述。Table tab:compute では 1T パラメータに $1.27\times10^{26}$ FLOPs（Gopher の 221.3 倍）と 21.2T tokens が必要と推定。
- **貢献**:
  1. 同じ結論に 3 つの独立手法で到達することで「$N$ と $D$ は等比で増やすべき」という主張をロバストに示した。
  2. Kaplan+ 2020 の指数が偏った原因を「全モデルで固定学習率スケジュールを使い、$D' \ll 130\text{B}$ の中間 loss を最終 loss として使ったため」と特定（cosine cycle 長は訓練トークン数と一致させるべき、Fig fig:cosine）。
  3. 実物の Chinchilla 70B を訓練し、Gopher 280B、GPT-3 175B、Jurassic-1 178B、MT-NLG 530B を ほぼ全タスクで上回ることを実証。
  4. 推論・fine-tuning コストが 4 倍下がるという実用的副産物。

## Takeaway（自分にとっての要点）

- **「model を大きくすれば良い」前提を著者らの実験範囲で定量的に否定した論文**。同じ FLOPs なら 70B × 1.4T tokens が 280B × 300B tokens に勝つ、という Chinchilla/Gopher 比較が中核。
- **3 手法独立で同じ指数に収束**という構成が決定打。1 手法だと「フィットの恣意性」を疑われるが、Approach 1 (envelope) / 2 (IsoFLOP) / 3 (parametric) は使うデータも仮定もずれていて、それでも $a \approx 0.5$ に揃ったのは強い。
- **Kaplan+ 2020 の指数のずれは LR スケジュール起因**という診断が重要。「短い実験で長い訓練を予測するなら、LR を実際の訓練 horizon に合わせて減衰させる」というメタ教訓は scaling 研究全般に効く。Fig fig:cosine の 25% 超過で性能崩れる、というのは実験設計の標準にすべき。
- パラメトリックフィットの $\alpha=0.34, \beta=0.28$ について、著者はどちらも $\frac{1}{2}$ より低く、future models and training approaches should endeavor to increase these coefficients と述べている。
- **「データを増やせ」の系として、データ品質と train-test contamination がボトルネックになる**ことを著者自身が明記。1T tokens 級ではプライバシー・有害性も増す。ただし leakage 警告は Pile/WikiText 等の **言語モデリングベンチマーク** に対するもので、著者は MMLU/BIG-bench/closed-book QA/common sense は「leakage が less of a concern」として強調軸に置いている（line 506-507）。
- **Chinchilla アーキ細部**: AdamW は language modelling loss と finetuning 後の downstream task performance を改善、SentencePiece の NFKC 無効化は数学・化学の表現に効くと著者は述べる（94.15% は Gopher トークナイザと共通）。forward/backward は bfloat16、optimizer state には float32 copy。
- 外挿表は実務的に有用: 「自社で $X$ FLOPs ある → どのサイズで何トークン回すべきか」が一発で引ける（ただし extrapolation の不確実性は著者自身が認めている）。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 3 つの methodologically 独立な手法が同じ指数に収束しており、結論の robustness が高い。
  - Kaplan+ 2020 の指数のズレを「LR スケジュールを fix した結果、短い run の loss を過大評価していた」と機序まで踏み込んで反論しており、単なる "我々の方が正しい" 主張になっていない。
  - 仮説（70B + 1.4T が最適）→ 実物（Chinchilla）の訓練 → MMLU 67.6 などで実証、というクローズドループの構成。
  - 推論・fine-tuning コストが 4× 軽くなる実用性が、scaling 議論を「アカデミック」から「実装すべき」に変えた。
  - Approach 3 の parametric form $E + A/N^\alpha + B/D^\beta$ は古典的 risk decomposition から自然に出てきて解釈しやすい。$E=1.69$ を「自然言語のエントロピー下限」と読める。
- **弱み / 疑問**:
  - **大規模スケールでは Chinchilla と Gopher の 2 点しか実物比較が無い**（著者自身が limitation で明記）。中間スケール（例えば 175B 級）での独立検証が無く、外挿表の 1T パラメータ予測は工学的には大きな飛躍。
  - $\log N_{opt}$ に concavity がある（Appendix curvature）と認めており、大規模では予測が optimal を過大評価する可能性がある。3 手法の中で Approach 3 だけ $a=0.46$ と低いが、この差の扱いは本文では限定的。
  - **train/test 汚染**について、著者は language modelling benchmarks では Chinchilla が 4× 多いデータで訓練されたため leakage が結果を人工的に押し上げうると注意している。一方、TeX 中には MMLU・BIG-bench 等の汚染チェック手順は示されていない。
  - スケーリング分析の training runs は 1 epoch 未満に限られており、multi-epoch 領域は未検証（著者明記）。ただし Chinchilla の 1.4T tokens 訓練では MassiveWeb と Wikipedia subset は 1 epoch を超える。
  - データ混合比 (MassiveWeb 45%, Books 30%, ...) を Gopher から微調整しているが、混合比の最適化と「等比スケール則」の交互作用は分析されていない。混合比次第で $a/b$ がブレる可能性。
  - LR・batch size・depth/width 比などのハイパラは「既存ヒューリスティクスを使う」で済ませており、これらが Kaplan+ と同じ罠を含んでいないか保証されていない。
  - Pile 上の bits-per-byte 改善（Fig fig:pile）は 4× データの直接効果か、compute-optimal フロンティアの効果か切り分けられていない。
  - "我々の指数の方が正しい" の主張は TeX 中では MassiveText / C4 / GitHub で検証されている。Model card では Chinchilla は English data で訓練されたとされており、多言語データでの追試は TeX 中には示されていない。
- **次に試したいこと**:
  - 異なる architecture（MoE、retrieval-augmented）で同じ 3 手法を回して $a, b$ がどう変わるか。著者は dense autoregressive transformer に方法を適用しており、retrieval は Borgeaud+ 2021 で実効的に training tokens を $\sim10$ 倍にする orthogonal approach として言及されている。
  - Multi-epoch 領域での scaling 則。Wikipedia は既に Chinchilla で 3.40 epoch 回されており、ここを起点に "data repetition vs new data" の効率曲線を引く。
  - Approach 3 の $\alpha=0.34, \beta=0.28$ を改善する future models and training approaches の比較。著者はこれらの係数を増やすべきと述べている。
  - MMLU 4 敗タスク（college_mathematics, econometrics, moral_scenarios, formal_logic）について、別 seed や追加の中間スケール実験で Chinchilla/Gopher 差の安定性を確認する（評者補足）。
  - Chinchilla で外挿された「175B モデル × 4.2T tokens」を実際に訓練したらどうなるか。
  - データ汚染を decontaminate した MMLU で +7.6 がどこまで残るか。

## Notes / Quotes

- 中心命題: "for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled." (abstract)
- Kaplan+ への反論のメカニズム: "setting the learning rate schedule to approximately match the number of training tokens results in the best final loss regardless of model size ... Using these intermediate losses results in underestimating the effectiveness of training models on less data than 130B tokens, and eventually contributes to the conclusion that model size should increase faster than training data size." (sec:related_work)
- Parametric fit の閉形式: $G = (\alpha A / \beta B)^{1/(\alpha+\beta)}, a = \beta/(\alpha+\beta), b = \alpha/(\alpha+\beta)$ → $\alpha=0.34, \beta=0.28$ から $a \approx 0.45, b \approx 0.55$。
- Limitations 自認: (1) 大規模で実物比較は Chinchilla/Gopher の 2 点のみ、(2) 高 FLOPs で $\log N_{opt}$ に concavity あり 最適サイズを過大評価の可能性、(3) すべて 1 epoch 未満、multi-epoch は未検証 (Discussion)。
- Toxicity: 25k unprompted サンプルで Perspective API 平均 0.087 (Chinchilla) vs 0.081 (Gopher)、95%ile 0.238 vs 0.230 → "negligible"。
- Chinchilla 70B vs Gopher 280B: 80 layers, 64 heads (vs Gopher 128 heads), d_model 8,192 (vs 16,384), max LR $1\times10^{-4}$ (vs $4\times10^{-5}$), batch size 1.5M→3M tokens。
- 外挿の極端例: 1T パラメータの compute-optimal training には 21.2T tokens / $1.27\times10^{26}$ FLOPs（Gopher の 221.3 倍）必要。
- Chinchilla の MMLU で 90% 超のタスク: high_school_gov_and_politics, international_law, sociology, us_foreign_policy。
- (verified 2026-05-20) 著者の leakage 警告は LM ベンチ向け (line 506-507)、と Critical Thoughts を訂正。venue 行を "NeurIPS 2022" 断定から TeX 不明記に変更。Related Papers の BIG-bench citation を bbl の "BIG-bench collaboration 2021" に合わせて訂正。根拠: main.tex L506-507, main.bbl L27-30, main.tex \title{...} と clsfile。
- (verified 2026-05-27) TeX に無い後続 OSS LLM 名・最適化器名・所属断定を削除し、外挿値の本文記述と Table tab:compute の値を区別 (main.tex, main.bbl)
- (verified 2026-05-27) multi-epoch と train/test leakage の記述を TeX の限定に合わせて修正し、MMLU 敗北タスク名を Table tab:mmlu_nums と本文に合わせて確認 (main.tex)

## Related Papers

- Kaplan+ 2020, "Scaling Laws for Neural Language Models" — 本論文が直接反論する scaling 指数 $(a=0.73, b=0.27)$ の出典。
- Rae+ 2021, Gopher — 比較対象の 280B モデル、データ MassiveText、評価プロトコルの基準。
- Brown+ 2020, GPT-3 — 175B、300B tokens の代表例。
- Lieber+ 2021, Jurassic-1; Smith+ 2022, MT-NLG 530B; Thoppilan+ 2022, LaMDA — under-trained の例として参照。
- Clark+ 2022, "Unified Scaling Laws for Routed LMs" — MoE の scaling、本論文の方法論的隣接。
- Hendrycks+ 2020, MMLU; BIG-bench collaboration 2021 — 主要評価ベンチマーク。
- Borgeaud+ 2021, "Improving language models by retrieving from trillions of tokens" — retrieval で実効データを増やすという orthogonal アプローチ。
- Loshchilov & Hutter 2018, AdamW — Chinchilla で採用。
- Huber 1964; Nocedal 1980 — Approach 3 のフィット（Huber loss + L-BFGS）。
