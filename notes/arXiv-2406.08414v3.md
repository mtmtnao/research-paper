# Discovering Preference Optimization Algorithms with and for Large Language Models

- arXiv: https://arxiv.org/abs/2406.08414
- source: ../papers/arXiv-2406.08414v3/
- authors: Chris Lu, Samuel Holt, Claudio Fanconi, Alex J. Chan, Jakob Foerster, Mihaela van der Schaar, Robert Tjarko Lange
- venue / year: NeurIPS 2024 (Sakana AI / FLAIR / U. Cambridge / U. Oxford)
- tags: [preference-optimization, DPO, LLM-driven-discovery, meta-learning, RLHF, AutoML]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: オフライン preference optimization の loss は DPO・IPO・SLiC・KTO のように人手で convex な閉形式を一個ずつ設計しており、可能な loss 関数空間は探索されていない。人間の創造性・直観・専門知識がボトルネック。
- **手法**: LLM-driven objective discovery。GPT-4 に「これまで試した loss 関数（PyTorch コード）とその MT-Bench スコア」を in-context で渡し、次の候補 loss を JSON（thought / name / code）で生成させる。unit test（valid shape、NaN なし、勾配が出る）を通った候補だけを実機 fine-tuning に回す。base model は `zephyr-7b-gemma-sft`、訓練データは `Argilla DPO Mix 7K`、β=0.05 固定、learning rate 5e-7、2 epoch、batch 2×grad accum 8、AdamW、cosine scheduler、A100×8 で約30分/run。MT-Bench スコアを fitness として LLM にフィードバックし、約 100 個の objective を評価。 alignment-handbook 上に実装。
- **結果**:
  - MT-Bench で discovered loss が baseline 群（DPO 7.888 / SLiC 7.881 / KTO 7.603）を上回る。Top 7 は DBAQL 7.978, AQL 7.953, PADLL 7.941, AQFL 7.931, CELL 7.925, LRML 7.916, PFL 7.900。
  - Held-out: AlpacaEval 2.0 vs GPT-4 で DPO 11.23% → PADLL 14.07% / AQFL 13.63% / LRML(=DiscoPOP) 13.21%。LC win rate vs SFT では LRML がベスト 65.18%。
  - TL;DR 要約 (Reddit 10%, 694 sample): DPO 88.27%・PADLL 88.54%・LRML 87.63% vs Human、ほぼ拮抗。
  - IMDb (GPT-2 + 感情分類器を ground-truth reward): LRML は β ∈ {0.025, 0.05, 0.1} で DPO・SLiC を上回る reward/KL frontier。
  - DiscoPOP (= Log Ratio Modulated Loss, LRML) を主結果として推奨:
    $f_{lrml}(\beta\rho) = (1-\sigma(\beta\rho/\tau))\cdot\log(1+\exp(-\beta\rho)) + \sigma(\beta\rho/\tau)\cdot\exp(-\beta\rho)$, $\tau=0.05$。logistic loss と exponential loss の sigmoid-gated blend。**非凸**で ρ=0 付近で **負の勾配**を持つ。
- **貢献**: (1) LLM をコードレベルの mutation/recombination 演算子として使う preference loss 自動発見パイプライン、(2) MT-Bench で SOTA を取った 7 個の新規 loss、特に DiscoPOP の同定、(3) DiscoPOP が「logistic と exponential の動的混合 + 非凸」という人間が思いつきにくい性質を持つことの分析（IMDb 訓練集合の 1.35% が local optima 間に落ち、それらは reward 差が小さい "noisy" pair であるという仮説検証付き）。

## Takeaway（自分にとっての要点）

- **discovery が effective なのは「LLM がランダム探索でなく概念合成をする」から**：CIFAR-10 ケーススタディで label-smoothed CE → squared error variant → 両者の合成、という流れが観察されている (§3, fig 2)。preference loss でも logistic + exponential / logistic + hinge / quantile-gated blend のような **既存 loss の重み付け合成**が高スコアになっている。新規性 = 重み付けスケジュールの設計、と読める。
- **DiscoPOP が非凸で ρ=0 で負勾配というのは衝撃的**。「正解側にも負荷をかけて curriculum を作る」効果と著者は推測。これは margin loss 系で margin を増やすのと逆方向の発想で、現場感覚と合わない。
- **β=0.05 がメタ学習時に固定だったため、生成された loss が τ=0.05 のスケールに焼き込まれている**。β=0.01 や β≥2.5 では収束しない（IMDb で確認）。著者自身が "we should constrain the exploring LLM to uphold the β multiplication with the input" と認めており、再現実装でも τ=β=0.05 で割り戻している（§Appendix D）。**β-aware にしたいなら meta-discovery を multi-β でやり直す必要**がある。
- 評価信号 (MT-Bench) は 80 問・GPT-4 judge と非常に noisy。fitness 1 query = GPU 30分。約 100 評価で頭打ちというのは「探索効率良好」とも読めるし「fitness ノイズで真の改善が見えない」とも読める。
- TL;DR では DiscoPOP は DPO/PADLL とほぼ同点で、MT-Bench での順位は held-out で再現していない（DBAQL は AlpacaEval で逆に baseline 以下に落ちる）。**MT-Bench スコアへの過適合**は常に疑うべき。
- **discovered loss を一般 RLHF パイプラインに使うときは、「β=0.05 / Gemma-7B / Argilla 7K に最適化されたもの」と認識しておく**。違うサイズ・違うデータでは効果が剥落するリスクが高い。

## Critical Thoughts（評価・疑問）

- **強み**:
  - LLM をコード変異子に使う meta-learning の中で、reward function でも RL アルゴでもなく **preference loss** という閉形式関数を見つけにいった点が新しい。$f:\mathbb{R}\to\mathbb{R}$ の探索なので unit test と numerical sanity check で篩いやすく、約 100 評価で済む。
  - 3 つの held-out task（AlpacaEval / TL;DR / IMDb）で複数の discovered loss を一貫して評価しており、MT-Bench のみで止めなかったのは誠実。
  - DiscoPOP の非凸性・負勾配を post-hoc で「noisy preference pair が local optima に落ちる」と IMDb の golden reward で検証したのは（後付けではあるが）面白い。Table tab:abs_reward_differences で optima 間の |r_w - r_l| が 0.981 vs 全体 3.861 と有意に小さい。
- **弱み / 疑問**:
  - **MT-Bench スコアでの選別 → MT-Bench スコアでの報告**で circularity。せめて MT-Bench を fitness にせず、AlpacaEval / TL;DR を fitness にして再走しないと「discovery が一般 loss を作った」とは言えない。
  - **fitness 比較がノイズ込み**: MT-Bench は 80 問 × GPT-4 judge。DPO 7.888 と LRML 7.916 の差は標準誤差を出していない。Table 1 にも error bar 無し（コメント `% P2: Re-run results and add error bars` が残っている）。多くの "discovered" loss は MT-Bench 上では DPO と区別不能な可能性がある。
  - **β を loss の形に焼き込んだ**事は著者自身が limitation で認めている。これは「DPO の理論的優位（β を KL 正則化として明確に分離）」を捨てているということで、実用上は痛い。
  - **GPT-4 依存**で reproducibility が低く、loss generation のコストが見えない（API 課金額の記載なし）。
  - DiscoPOP がなぜ DPO より良いのかの理論的説明が薄い。「非凸が curriculum を作る」「sigmoid gate が adaptive」は qualitative。同じ振る舞いを sigmoid temperature ablation や別の凸 blend で再現できるかの ablation がない。
  - DBAQL が discovery task で 1 位なのに AlpacaEval では baseline 以下、PFL は全 held-out で baseline 以下と、discovery で測った fitness と汎化性能の相関は怪しい。**「DiscoPOP を選ぶ基準」が論文中で post-hoc に決まっている**（"consistently high across held-out evaluation tasks" §6）。
  - 7B Gemma 1 モデルでしか試していない。Llama 系・小モデル・別 dataset で再現しないと、loss の汎用性は保証できない。著者も Limitation でこれを認める。
- **次に試したいこと**:
  - 同じ pipeline を **fitness = AlpacaEval LC win rate** に差し替えて re-discover し、MT-Bench で測ったときに新 loss が同等以上に出るか。
  - DiscoPOP の sigmoid gate を $\sigma(\beta\rho/\tau)$ ではなく $\sigma(\rho)$（β/τ を切り離す）にリパラメタライズして β スイープでの安定性を取り戻せるか。
  - 「logistic + exponential を sigmoid で線形結合」だけを残した minimal DiscoPOP（学習可能 τ）と元の DiscoPOP の差。本質が gate か exp 項かを切り分け。
  - 候補 loss を **複数 seed × 複数 β** で fitness 評価し、ノイズ込みで top-K を選ぶよう pipeline を変える（現状 1-run/loss）。
  - 「discovered loss を distillation の教師信号にして、もう一度 GPT-4 にコード生成させる」self-improve loop。

## Notes / Quotes

- "The best performing of these we call \textit{Discovered Preference Optimization} (DiscoPOP), a novel algorithm that adaptively blends logistic and exponential losses." (abstract)
- "Surprisingly, the DiscoPOP function has a non-convex segment and negative gradients at the starting point $\rho = 0$. This is potentially helpful for introducing a curriculum or for stochasticity." (§6.1 Log Ratio Modulated Loss (DiscoPOP))
- "LRML struggles to converge when $\beta$ is too low ($\beta \leq 0.01$) or too high ($\beta \geq 2.5$), likely because $\beta \neq 0.05$ was never seen or used during the discovery process." (§6.2 Limitations of DiscoPOP)
- "we should constrain the exploring LLM to uphold the $\beta$ multiplication with the input before any other calculations are done with the difference of log-ratios $\rho$." (Appendix E Discovered Objective Functions 冒頭)
- "After evaluating approximately $100$ objective functions, we catalogued the best-performing ones in Table 1." (§4.2)
- システムプロンプトは "You are a machine learning researcher who is testing out different RLHF loss functions. ... You are deeply familiar with binary classification losses from the literature. Be creative and reference prior literature when possible." (Appendix A)
- Robustness ablation (Appendix D \"Discovery Robustness with respect to LLM Hyperparameters\", fig:vlm_discovery): sampling temperature {0.1, 0.5, 1.0}・top-K context・thought/error message の有無に対して CIFAR-10 discovery は概ね robust（3 seed 平均）。
- IMDb local optima 解析: $f_{lrml}(-2.3714)=0.785929$（local min）, $f_{lrml}(1.44012)=0.87829$（local max）、両 optima 間に落ちる訓練 sample は 1.35%、|r_w - r_l| が 0.981 (全体 3.861)。
- (verified 2026-05-20) Notes/Quotes 内のセクション参照を修正: §5.1→§6.1, §5.2→§6.2 (Analysis of DiscoPOP は §6), Appendix D→Appendix E (Discovered Objective Functions), robustness ablation の場所を Appendix D の subsection に訂正 (根拠: DiscoveredPreferenceOptimization.tex L528, L559, L873, L940, L956)。

## Related Papers

- Rafailov+ 2023, DPO — 出発点となる loss、$f=-\log\sigma$。
- Zhao+ 2023, SLiC — hinge loss `ReLU(1-βρ)`。比較対象。
- Azar+ 2023, IPO — squared loss `(x-1)^2`。
- Ethayarajh+ 2024, KTO — Kahneman-Tversky-inspired loss、比較対象。
- Tang+ 2024 (generalized preference optimization) — $f$ を任意のスカラー loss にする一般化、本論文の枠組みの基礎。
- Ma+ 2023 Eureka, Yu+ 2023 — LLM で **環境固有 reward** を書かせる先行研究。本論文は task-agnostic objective を狙う点が違う。
- Romera-Paredes+ 2024 (FunSearch) / Lehman+ 2023 / Lange+ 2024 LMX — LLM を evolution operator として使う系譜。
- Lu+ 2022 / Jackson+ 2024 / Metz+ 2022 (VeLO) — meta-learned objective / optimizer の neural parameterization 系。
- Zheng+ 2024 (MT-Bench) / Dubois+ 2024 (AlpacaEval 2.0 LC) / Völske+ 2017 (TL;DR) / Maas+ 2011 (IMDb) — 評価ベンチマーク。
