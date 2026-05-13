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
  勾配は「policy が誤って $y_l$ を高く評価しているサンプルほど重く」 $\nabla\log\pi(y_w) - \nabla\log\pi(y_l)$ を押す形になる。報酬モデル学習も RL ループも不要。Theorem 1 で、Plackett-Luce/Bradley-Terry と整合する全ての報酬同値クラスがこの再パラメタライズで表現可能と保証している。
- **結果**:
  - **IMDb sentiment (GPT-2-large + siebert/sentiment-roberta-large-english を ground-truth 報酬として使用)**: 報酬–KL frontier で DPO が PPO, PPO-GT(真の報酬にアクセスできるオラクル), Preferred-FT, Unlikelihood を全 KL 領域で支配。
  - **Reddit TL;DR summarization (GPT-J SFT, $\beta=0.5$)**: GPT-4 (C) 評価で reference 要約に対する win rate が DPO ≈ 61% @ temp 0、PPO ≈ 57% @ temp 0。温度が上がるほど PPO は base GPT-J 並みまで崩れるが DPO は安定。Best of $N$（=実質的に PPO 上限）も超える最大 win rate を達成。
  - **CNN/DailyMail への OOD 汎化（TL;DR で学習した policy をそのまま適用）**: DPO 0.36 / 0.31（temp 0 / 0.25）vs PPO 0.26 / 0.23。
  - **Anthropic HH 単ターン対話 (Pythia-2.8B + Preferred-FT を $\piref$ に)**: dataset の chosen 応答に対して有意に勝ち越す唯一の手法（2-shot prompt / Preferred-FT / Best of 128 / 別ソースの PPO-HH と比較）。
  - **人間評価 (TL;DR)**: DPO @0.25 vs PPO @0 で human win rate 58% (GPT-4 (S) 47%, GPT-4 (C) 54%)。GPT-4 と人間の一致率 67–70% は人間同士の一致 (65%) と同等。
- **貢献**: (1) RLHF の KL 制約付き報酬最大化を、**報酬モデルを陽に学習せず・RL ループも使わず**、単純なバイナリ分類損失で厳密に解く DPO アルゴリズム。 (2) Theorem 1: $r(x,y) = \beta\log\frac{\pi(y\mid x)}{\piref(y\mid x)}$ という再パラメタライズが Plackett-Luce 系報酬の同値クラス全体を覆うことの証明。 (3) sentiment / TL;DR / Anthropic HH での実証で PPO と同等以上を、ハイパラチューニングほぼなしで達成。

## Takeaway（自分にとっての要点）

- DPO は「RL を使わない RLHF」というより、**RLHF の KL 制約最適解の閉形式から $r$ を消去するトリック**。policy 自身が implicit reward $\hat r_\theta(x,y) = \beta\log\frac{\pi_\theta(y\mid x)}{\piref(y\mid x)}$ を運ぶ。Theorem 1 によって表現力が落ちないことが保証されており、「PPO の近似ではなく等価変形」だと主張できる点が強い。
- 勾配の $\sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$ という重みが本質的。これを外す（単なる SFT on $y_w$ + unlikelihood on $y_l$）と degenerate な「when when when ...」を吐く（Appendix Table 6）。Unlikelihood は単独だと壊れる。
- 実装が衝撃的に短い（Appendix のコードは 15 行）。$\beta$, バッチ 64, RMSprop, lr 1e-6 線形 warmup 150 step が default。TL;DR のみ $\beta=0.5$。逆に言えば**ほぼハイパラ探索なしでこの強さ**は再現性の意味で大きい。
- PPO が温度に脆い（高温で base モデル並みに崩壊）一方 DPO はロバスト。Frontier の比較で **PPO-GT (ground-truth reward オラクル) すら超える** のは、PPO 側のサンプリング分散・value baseline・PPO クリップなどの実装ロスが効いている可能性を示唆している。
- OOD 汎化（TL;DR→CNN/DailyMail）でも勝つ。「DPO は報酬モデルを陽に持たないので汎化が弱いはず」という直感的批判への先回り反論として収録されている。
- Anthropic HH では SFT モデルが存在しないので Preferred-FT で $\piref$ を作る → distribution shift を避けるための実用的な工夫。public 嗜好データを使う多くの再現プロジェクトで効く所作。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 理論（Theorem 1 + Lemma 1/2）と実装（15 行）が両方しっかりしている稀なタイプ。Plackett-Luce 一般化が Appendix にあるので $K>2$ ランキングデータにも自然に拡張できる。
  - reward model を捨てたのに **PPO-GT すら frontier で支配** したのは強い結果。実用上、PPO の不安定さの大半は KL 正則化項を Monte-Carlo で扱うことに由来していたという解釈の傍証になる。
  - 評価が単一タスクに頼っておらず sentiment（合成 ground-truth 報酬で frontier）/ TL;DR / Anthropic HH / OOD (CNN/DailyMail) / 人間評価 と層が厚い。
  - 著者自身が GPT-4 と人間の一致率を別途人間スタディで検証している点（GPT-4 (C) prompt の妥当化）が誠実。
- **弱み / 疑問**:
  - **モデルサイズが最大 6B (GPT-J)** で、PPO-RLHF が実際に効く 70B+ スケールの主張ではない（著者自身 Limitations で言及）。
  - **報酬モデルを陽に持たない副作用** — best-of-$N$ rerank、reward model を別の用途（safety filter, distillation の教師）に転用、といった派生的な使い方が DPO だけだと閉ざされる。論文中で深く議論されていない。
  - **reward over-optimization** が DPO 内でどう現れるかは未解明。Fig 3 (dialogue right) で学習後半に win rate がわずかに下がる現象を「これが over-optimization かも」と Limitations で示唆するに留まる。
  - 嗜好データを生成した SFT モデル $\pisft$ と $\piref$ の乖離（distribution shift）に DPO は弱いはず。Anthropic HH では Preferred-FT で吸収しているが、これがどの程度クリティカルかの sweep がない。
  - **GPT-4 (S) と (C) で win rate が 7pt 違う** (DPO で 47 vs 54)。評価者プロンプトのチューニングで論文の結論が動きうる脆さを著者も認める。
  - PPO ベースラインの実装品質に疑念の余地。Anthropic HH で「別ソースの PPO-HH モデルが base Pythia-2.8B にも勝てなかった」と書いてあり、本論文での PPO チューニングが万全だったか比較表だけからは判別しにくい。
  - 6B での結果が 70B+ で保たれる保証はない（Llama-2/3, GPT-3.5 級では追試が必要）。
- **次に試したいこと**:
  - 同じ $\piref$ から DPO で得た $\pi_\theta$ の implicit reward $\hat r_\theta = \beta\log\frac{\pi_\theta}{\piref}$ を取り出して、別途 RM として best-of-$N$ rerank に使い、explicit RM と性能比較する（DPO で reward model "を" 取り出せるかの検証）。
  - $\beta$ を学習中に schedule する（最初は緩く、後で締める） → reward over-optimization の様相を可視化。
  - DPO で得た policy を教師にして iterative DPO (online DPO / 自己嗜好) を回したときの収束挙動（後続研究で実際に出ている方向だが、本論文では未検証）。
  - Plackett-Luce で $K>2$ のランキング嗜好（例えば 4-way ranking）データに対する DPO 拡張の有効性を、Bradley-Terry pairwise 化したケースと比較。
  - PPO の reward model を DPO の $\pi_\theta$ に置き換えた場合の hybrid（DPO で warmup → PPO で仕上げ）が overall optimum に近づくか。

## Notes / Quotes

- Eq. 7 (DPO loss): $\mathcal{L}_\text{DPO}(\pi_{\theta}; \piref) = -\mathbb{E}_{(x, y_w, y_l)\sim \mathcal{D}}\bigl[\log \sigma(\beta \log\frac{\pi_{\theta}(y_w\mid x)}{\piref(y_w\mid x)} - \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\piref(y_l\mid x)})\bigr]$
- "the policy network represents both the language model and the (implicit) reward." (§DPO)
- Theorem 1: $r(x,y) = \beta\log\frac{\pi(y\mid x)}{\piref(y\mid x)}$ で Plackett-Luce 整合の報酬同値類すべてを覆う。Proposition 1 でその選び方の一意性も示す（同値類ごとに唯一）。
- Default ハイパラ: $\beta=0.1$, batch 64, RMSprop, lr 1e-6, 150 step linear warmup（TL;DR のみ $\beta=0.5$）。
- 勾配の重み $\sigma(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w))$ を外すと degenerate 生成（Appendix Table on unlikelihood, 「when when ...」を吐く実例）。
- 著者明示の Limitations: (i) OOD 汎化の包括的検証が未了、(ii) self-labeling で unlabeled prompt を活用できるか未検証、(iii) reward over-optimization が DPO でも起きるかは Fig 3-right の僅かな低下が示唆するに留まる、(iv) 6B より大きいモデルへのスケーリング未検証、(v) GPT-4 評価が prompt sensitive、(vi) 言語以外のモダリティ（生成モデル一般）への応用は future work。
- 人間スタディ参加者は Stanford 在学・卒業の STEM 学生 25 名（Appendix にフルネーム掲載）。GPT-4 と人間の一致率は人間同士の一致と同水準。

## Related Papers

- Christiano+ 2017 "Deep RL from Human Preferences" — RLHF の原型 baseline。
- Ziegler+ 2019 / Stiennon+ 2020 (TL;DR summarization with human feedback) — 直接の比較対象、SFT モデル・データセットも継承。
- Ouyang+ 2022 InstructGPT, Bai+ 2022 (Anthropic HH) — RLHF パイプラインと嗜好データセットの出所。
- Schulman+ 2017 PPO — 置き換え対象。
- Bradley-Terry 1952, Plackett 1975 / Luce 2012 — 嗜好確率モデル。
- Peters & Schaal 2007 / Peng+ 2019 (Advantage-weighted regression) / Korbak+ 2022, Go+ 2023 — KL 制約付き最適 policy の解析解の系譜。
- Welleck+ 2019 Unlikelihood — naive baseline で degenerate を示すために使用。
- Levine 2018 (Control as Inference) — DPO の最適化目的とのつながりを §5.2 で議論。
- Bong & Rinaldo 2022 — Plackett-Luce 識別可能性の理論的背景（DPO の consistency 議論で参照）。
