# Direct Preference Optimization: Your Language Model is Secretly a Reward Model

- arXiv: https://arxiv.org/abs/2305.18290
- source: ../papers/arXiv-2305.18290v3/
- authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
- venue / year: NeurIPS 2023
- tags: [RLHF, preference-learning, alignment, LLM, fine-tuning]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: RLHF は (1) 嗜好データから報酬モデル $r_\phi$ を学習し、(2) PPO 等の強化学習で $\max_\pi \mathbb{E}[r_\phi(x,y)] - \beta\mathrm{KL}[\pi\|\piref]$ を最適化、という多段パイプライン。reward model と policy の 2 モデルを保持し、学習中に policy からサンプリングする必要があり、複雑・不安定・計算コストが高い。
- **手法**: KL 制約付き報酬最大化の解析解 $\pi_r(y\mid x) = \frac{1}{Z(x)}\piref(y\mid x)\exp(\frac{1}{\beta}r(x,y))$ を逆に解いて $r(x,y) = \beta\log\frac{\pi(y\mid x)}{\piref(y\mid x)} + \beta\log Z(x)$ とし、Bradley-Terry 嗜好モデルに代入する。$Z(x)$ は差分でキャンセルし、嗜好確率が policy $\pi_\theta$ と $\piref$ だけで書ける。これにより最終損失は単純なロジスティック回帰：
  $$\mathcal{L}_\text{DPO} = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\bigl[\log\sigma\bigl(\beta\log\tfrac{\pi_\theta(y_w\mid x)}{\piref(y_w\mid x)} - \beta\log\tfrac{\pi_\theta(y_l\mid x)}{\piref(y_l\mid x)}\bigr)\bigr]$$
  勾配は「policy が誤って $y_l$ を高く評価しているサンプルほど重く」 $\nabla\log\pi(y_w) - \nabla\log\pi(y_l)$ を押す形になる。explicit/standalone な報酬モデル学習や RL ループを使わない。Theorem 1 で、Plackett-Luce/Bradley-Terry と整合する全ての報酬同値クラスがこの再パラメタライズで表現可能と保証している。
- **結果**:
  - **IMDb sentiment (GPT-2-large + siebert/sentiment-roberta-large-english を ground-truth 報酬として使用)**: 報酬–KL frontier で DPO が PPO, PPO-GT(真の報酬にアクセスできるオラクル), Preferred-FT, Unlikelihood を全 KL 領域で支配。
  - **Reddit TL;DR summarization (GPT-J SFT, $\beta=0.5$)**: GPT-4 (C) 評価で reference 要約に対する win rate が DPO ≈ 61% @ temp 0、PPO ≈ 57% @ temp 0。温度が上がるほど PPO は base GPT-J 並みまで崩れるが DPO は安定。Best of $N$ baseline より高い最大 win rate を達成。
  - **CNN/DailyMail への OOD 汎化（TL;DR で学習した policy をそのまま適用）**: DPO 0.36 / 0.31（temp 0 / 0.25）vs PPO 0.26 / 0.23。
  - **Anthropic HH 単ターン対話 (Pythia-2.8B + Preferred-FT を $\piref$ に)**: dataset の chosen 応答に対して勝ち越す唯一の手法（2-shot prompt / Preferred-FT / Best of 128 / 別ソースの PPO-HH と比較）。
  - **人間評価 (TL;DR)**: DPO @0.25 vs PPO @0 で human win rate 58% (GPT-4 (S) 47%, GPT-4 (C) 54%)。GPT-4 と人間の一致率 67–70% は人間同士の一致 (65%) と同等。
- **貢献**: (1) RLHF の KL 制約付き報酬最大化を、**explicit/standalone な報酬モデルを陽に学習せず・RL ループも使わず**、単純なバイナリ分類損失として最適化する DPO アルゴリズム。 (2) Theorem 1: $r(x,y) = \beta\log\frac{\pi(y\mid x)}{\piref(y\mid x)}$ という再パラメタライズが Plackett-Luce 系報酬の同値クラス全体を覆うことの証明。 (3) sentiment / TL;DR / Anthropic HH での実証で PPO と同等以上を、ハイパラチューニングほぼなしで達成。

## Takeaway（自分にとっての要点）

- DPO は「RL を使わない RLHF」というより、**KL 制約付き報酬最大化の最適 policy の閉形式から $r$ を消去するトリック**。policy 自身が implicit reward $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\piref(y\mid x)}$ を運ぶ。Theorem 1 によって、Plackett-Luce/Bradley-Terry の reward equivalence class をこの再パラメタライズで表せることを示している。
- 勾配の $\sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$ という重みが本質的。これを外す（単なる SFT on $y_w$ + unlikelihood on $y_l$）と degenerate な「when when when ...」を吐く（Appendix の `tab:unlikelihood_generations`）。Unlikelihood は単独だと壊れる。
- 実装が衝撃的に短い（Appendix B の PyTorch コードはコア処理が 1 行の `F.logsigmoid(beta * (pi_logratios - ref_logratios))` で済む短さ）。default は $\beta=0.1$, batch 64, RMSprop, lr 1e-6 を 150 step で 0→1e-6 に線形 warmup。TL;DR のみ $\beta=0.5$。逆に言えば**ほぼハイパラ探索なしでこの強さ**は再現性の意味で大きい。
- PPO が温度に脆い（高温で base GPT-J 並みに崩壊）一方 DPO はロバスト。Frontier の比較で **PPO-GT (ground-truth reward オラクル) より良い frontier** も報告している。
- OOD 汎化（TL;DR→CNN/DailyMail）でも PPO を上回る。著者は、DPO が additional unlabeled Reddit TL;DR prompts を使っていないにもかかわらず PPO-based models と同様に汎化できる initial evidence と位置づけている。
- Anthropic HH では SFT モデルが存在しないので Preferred-FT で $\piref$ を作る → distribution shift を避けるための実用的な工夫。public 嗜好データを使う多くの再現プロジェクトで効く所作。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 理論（Theorem 1 + Lemma 1/2）と実装例（Appendix B の PyTorch loss）が両方そろっている。Plackett-Luce 一般化が Appendix にあるので、$K>2$ ランキングデータにも目的関数を書ける。
  - reward model を陽に学習しないのに **PPO-GT より良い frontier** を報告しているのは強い結果。著者は actor-critic の不安定性について、soft value/normalization term の扱いが高分散や value function 最適化の難しさにつながると議論している。
  - 評価が単一タスクに頼っておらず sentiment（合成 ground-truth 報酬で frontier）/ TL;DR / Anthropic HH / OOD (CNN/DailyMail) / 人間評価 と層が厚い。
  - 著者自身が GPT-4 と人間の一致率を別途人間スタディで検証している点（GPT-4 (C) prompt の妥当化）が誠実。
- **弱み / 疑問**:
  - **モデルサイズが最大 6B (GPT-J)** で、state-of-the-art models orders of magnitude larger へのスケーリングは未検証（著者自身 Limitations で言及）。
  - **standalone reward model を陽に持たない副作用** — Best of $N$ baseline は learned reward function で rerank するが、DPO の implicit reward を standalone RM のように別用途へ転用する設計は論文中で深く議論されていない。
  - **reward over-optimization** が DPO 内でどう現れるかは未解明。`fig:dialogue-main` 右図で学習後半に win rate がわずかに下がる現象を「これが over-optimization かも」と Limitations で示唆するに留まる。
  - public preference datasets ではサンプル元の $\pisft$ が利用できない場合がある。著者は $\piref$ を preferred completions の likelihood 最大化で作って distribution shift を緩和すると述べるが、この処置の重要度を切り分ける sweep は示されていない。
  - **GPT-4 (S) と (C) で win rate が 7pt 違う** (DPO で 47 vs 54)。GPT-4 win rate が prompt に影響される点を著者も認めている。
  - PPO ベースラインの実装品質に疑念の余地。Anthropic HH で「別ソースの PPO-HH モデルが base Pythia-2.8B にも勝てなかった」と書いてあり、本論文での PPO チューニングが万全だったか比較表だけからは判別しにくい。
  - 6B での結果が state-of-the-art models orders of magnitude larger で保たれる保証はない。
- **次に試したいこと**:
  - 同じ $\piref$ から DPO で得た $\pi_\theta$ の implicit reward $\hat r_\theta = \beta\log\frac{\pi_\theta}{\piref}$ を取り出して、別途 RM として Best of $N$ rerank に使い、explicit RM と性能比較する（評者補足）。
  - $\beta$ を学習中に schedule した場合に、reward over-optimization の様相が変わるか可視化する（評者補足）。
  - 著者が Future Work に挙げる self-labeling from the DPO policy で unlabeled prompts を活用できるか検証する。
  - Plackett-Luce で $K>2$ のランキング嗜好データに対する DPO 拡張の有効性を、Bradley-Terry pairwise 化したケースと比較する（評者補足）。

## Notes / Quotes

- Eq. 7 (DPO loss): $\mathcal{L}_\text{DPO}(\pi_{\theta}; \piref) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\bigl[\log \sigma(\beta \log\frac{\pi_{\theta}(y_w\mid x)}{\piref(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\piref(y_l\mid x)})\bigr]$
- "the policy network represents both the language model and the (implicit) reward." (§DPO)
- Theorem 1: $r(x,y) = \beta\log\frac{\pi(y\mid x)}{\piref(y\mid x)}$ で Plackett-Luce 整合の報酬同値類すべてを覆う。Proposition 1 でその選び方の一意性も示す（同値類ごとに唯一）。
- Default ハイパラ: $\beta=0.1$, batch 64, RMSprop, lr 1e-6, 150 step linear warmup（TL;DR のみ $\beta=0.5$）。
- 勾配の重み $\sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$ を外すと degenerate 生成（Appendix Table on unlikelihood, 「when when ...」を吐く実例）。
- 著者明示の Limitations: (i) OOD 汎化の包括的検証が未了、(ii) self-labeling で unlabeled prompt を活用できるか未検証、(iii) reward over-optimization が DPO でも起きるかは `fig:dialogue-main` 右図の僅かな低下が示唆するに留まる、(iv) 6B より大きいモデルへのスケーリング未検証、(v) GPT-4 評価が prompt sensitive、(vi) 言語以外のモダリティ（生成モデル一般）への応用は future work。
- 人間スタディ参加者は Stanford 在学/卒業/訪問者の STEM (主に CS) フォーカス 25 名（1 名は遅延提出で最終解析から除外、Appendix C.4 にフルネーム掲載）。GPT-4 と人間の一致率は人間同士の一致と同水準。
- (verified 2026-05-20) "Appendix のコードは 15 行" を実 verbatim 行数と照合し、論文には明示的な行数記述が無いため、コア処理が 1 行で書ける短さである旨に書き直した（main.tex §B, lines 556–581）。
- (verified 2026-05-20) 人間スタディの母集団記述を App C.4 の原文（Stanford students/recent graduates/visitors, STEM (mainly CS), 25 名中 1 名は遅延で除外）に合わせて精緻化（main.tex, app:human-study）。
- (verified 2026-05-27) TL;DR / OOD / Critical Thoughts の解釈過多な表現を、TeX の結果記述と Limitations/Future Work の表現に合わせて弱めた（main.tex, Experiments, Discussion, Appendix B）。
- (verified 2026-05-27) TeX に明示されていない追試案・解釈は「評者補足」と明記し、state-of-the-art models orders of magnitude larger など TeX 上の表現へ修正（main.tex, Limitations & Future Work）。
- (verified 2026-05-27) `\rev{old}{new}` の最終本文に合わせ、「optimized exactly」「有意に」など final TeX より強い表現を削除・弱化（main.tex, abstract, DPO, experiments）。
- (verified 2026-05-27) unlikelihood degeneration の参照を未確認の表番号ではなく TeX label `tab:unlikelihood_generations` に変更（main.tex, Appendix unlikelihood baseline）。
- (verified 2026-05-27) reward over-optimization の参照を図番号ではなく TeX label `fig:dialogue-main` に変更（main.tex, Discussion）。

## Related Papers

- Christiano+ 2017 "Deep reinforcement learning from human preferences" — human preferences からの学習に関する先行研究。
- Ziegler+ 2020 "Fine-tuning language models from human preferences" / Stiennon+ 2022 "Learning to summarize from human feedback" — RLHF pipeline と TL;DR summarization の直接の先行研究。
- Ouyang+ 2022 "Training language models to follow instructions with human feedback", Bai+ 2022 "Training a helpful and harmless assistant with reinforcement learning from human feedback" — RLHF pipeline と Anthropic Helpful and Harmless dialogue dataset の出所。
- Schulman+ 2017 "Proximal policy optimization algorithms" — PPO baseline。
- Bradley-Terry 1952, Plackett 1975 / Luce 2012 — 嗜好確率モデル。
- Peters & Schaal 2007 "Reinforcement learning by reward-weighted regression for operational space control" / Peng+ 2019 "Advantage-weighted regression: Simple and scalable off-policy reinforcement learning" / Korbak+ 2022 / Go+ 2023 — KL 制約付き最適 policy の解析解の系譜。
- Welleck+ 2019 "Neural text generation with unlikelihood training" — Unlikelihood baseline。
- Levine 2018 (Control as Inference) — DPO の最適化目的とのつながりを §5.2 で議論。
- Bong & Rinaldo 2022 — Plackett-Luce 識別可能性の理論的背景（DPO の consistency 議論で参照）。
