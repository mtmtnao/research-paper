# Training Compute-Optimal Large Language Models

- arXiv: https://arxiv.org/abs/2203.15556
- source: ../papers/arXiv-2203.15556v1/
- authors: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (DeepMind)
- venue / year: arXiv preprint 2022 (後に NeurIPS 2022)
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
  - **外挿表 (Table tab:compute)**: 175B モデルの最適予算は $4.41\times10^{24}$ FLOPs / 4.2T tokens、280B Gopher 級は $10^{25}$ FLOPs / 6.8T tokens、1T パラメータには $\sim10^{26}$ FLOPs（Gopher の 250 倍超）と 21.2T tokens が必要。
- **貢献**:
  1. 同じ結論に 3 つの独立手法で到達することで「$N$ と $D$ は等比で増やすべき」という主張をロバストに示した。
  2. Kaplan+ 2020 の指数が偏った原因を「全モデルで固定学習率スケジュールを使い、$D' \ll 130\text{B}$ の中間 loss を最終 loss として使ったため」と特定（cosine cycle 長は訓練トークン数と一致させるべき、Fig fig:cosine）。
  3. 実物の Chinchilla 70B を訓練し、Gopher 280B、GPT-3 175B、Jurassic-1 178B、MT-NLG 530B を ほぼ全タスクで上回ることを実証。
  4. 推論・fine-tuning コストが 4 倍下がるという実用的副産物。

## Takeaway（自分にとっての要点）

- **「model を大きくすれば良い」前提が定量的に否定された分水嶺の論文**。同じ FLOPs なら 70B × 1.4T tokens が 280B × 300B tokens に勝つ。以後の OSS LLM (LLaMA, Falcon 等) が「小さいモデル × 大量トークン」に舵を切った直接の根拠。
- **3 手法独立で同じ指数に収束**という構成が決定打。1 手法だと「フィットの恣意性」を疑われるが、Approach 1 (envelope) / 2 (IsoFLOP) / 3 (parametric) は使うデータも仮定もずれていて、それでも $a \approx 0.5$ に揃ったのは強い。
- **Kaplan+ 2020 の指数のずれは LR スケジュール起因**という診断が重要。「短い実験で長い訓練を予測するなら、LR を実際の訓練 horizon に合わせて減衰させる」というメタ教訓は scaling 研究全般に効く。Fig fig:cosine の 25% 超過で性能崩れる、というのは実験設計の標準にすべき。
- パラメトリックフィットの $\alpha=0.34, \beta=0.28$ はどちらも理論下限の 0.5 を下回る → 「現在の Transformer + AdamW はパラメータ効率・データ効率ともに最適でない」という診断にもなっており、アーキ・最適化器側の改善余地を示唆。
- **「データを増やせ」の系として、データ品質と train-test contamination がボトルネックになる**ことを著者自身が明記。1T tokens 級ではプライバシー・有害性も増す。Chinchilla の MMLU 圧勝の一部はデータ被りかもしれない、と自分でも認めている（"some caution is needed ... train/test set leakage may artificially enhance the results"）。
- **Chinchilla アーキ細部**: AdamW で +α、SentencePiece の NFKC 無効化で数学・化学の表現が改善（94.15% は Gopher トークナイザと共通）、bfloat16 fwd/bwd + float32 optimizer state。実用上の細かい教訓が transferable。
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
  - $\log N_{opt}$ に concavity がある（Appendix curvature）と認めており、大規模では予測が optimal を過大評価する可能性。3 手法の中で Approach 3 だけ $a=0.46$ と低い → 高 FLOPs では Approach 3 を信じるべき、と読むのが自然だが本文ではあまり強調されない。
  - **train/test 汚染**の影響を Chinchilla は 4× データで余計に受けるはずだが、MMLU・BIG-bench 等の汚染チェックは行われていない。MMLU の 4 敗タスク（college_mathematics, formal_logic 等）が「汚染で説明できない数学・記号推論」である点は示唆的。
  - 訓練は 1 epoch 未満に限られており、multi-epoch 領域は未検証（著者明記）。1.4T tokens を超えるデータ枯渇時の挙動が不明。
  - データ混合比 (MassiveWeb 45%, Books 30%, ...) を Gopher から微調整しているが、混合比の最適化と「等比スケール則」の交互作用は分析されていない。混合比次第で $a/b$ がブレる可能性。
  - LR・batch size・depth/width 比などのハイパラは「既存ヒューリスティクスを使う」で済ませており、これらが Kaplan+ と同じ罠を含んでいないか保証されていない。
  - Pile 上の bits-per-byte 改善（Fig fig:pile）は 4× データの直接効果か、compute-optimal フロンティアの効果か切り分けられていない。
  - "我々の指数の方が正しい" の主張は MassiveText / C4 / GitHub という相関の強い英語コーパスのみで検証されており、多言語・コード専用などでの追試は将来課題。
  - Approach 2 の "parabola fit" は IsoFLOP curve が単峰である前提だが、低 FLOPs では valley が浅く、フィット感度が不明（bootstrap で 10/90 percentile は出しているが）。
- **次に試したいこと**:
  - 異なる architecture（MoE、Mamba/SSM、retrieval-augmented）で同じ 3 手法を回して $a, b$ がどう変わるか。著者は dense transformer に限定しているが、retrieval は実効データを 10× 増やすので $b$ が大きく動くはず。
  - Multi-epoch 領域での scaling 則。Wikipedia は既に Chinchilla で 3.40 epoch 回されており、ここを起点に "data repetition vs new data" の効率曲線を引く。
  - Approach 3 の $\alpha=0.34, \beta=0.28$ を改善する最適化器・正則化（μP, Lion, AdEMAMix 等）の比較。lower-bound $1/2$ に近づけられれば scaling 自体が変わる。
  - MMLU 4 敗タスク（mathematics, formal_logic, moral_scenarios, econometrics）が「データを増やしても効かない」種類なのか、「Chinchilla で initialization が悪かった」だけなのかを seed 違いで切り分け。
  - Chinchilla で外挿された「175B モデル × 4.2T tokens」を実際に訓練したらどうなるか（→ 後の LLaMA-2/3 等が部分的に答えている）。
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

## Related Papers

- Kaplan+ 2020, "Scaling Laws for Neural Language Models" — 本論文が直接反論する scaling 指数 $(a=0.73, b=0.27)$ の出典。
- Rae+ 2021, Gopher — 比較対象の 280B モデル、データ MassiveText、評価プロトコルの基準。
- Brown+ 2020, GPT-3 — 175B、300B tokens の代表例。
- Lieber+ 2021, Jurassic-1; Smith+ 2022, MT-NLG 530B; Thoppilan+ 2022, LaMDA — under-trained の例として参照。
- Clark+ 2022, "Unified Scaling Laws for Routed LMs" — MoE の scaling、本論文の方法論的隣接。
- Hendrycks+ 2020, MMLU; Srivastava+ 2022, BIG-bench — 主要評価ベンチマーク。
- Borgeaud+ 2021, RETRO — retrieval で実効データを増やすという orthogonal アプローチ。
- Loshchilov & Hutter 2018, AdamW — Chinchilla で採用。
- Huber 1964; Nocedal 1980 — Approach 3 のフィット（Huber loss + L-BFGS）。
