# Evolution Strategies as a Scalable Alternative to Reinforcement Learning

- arXiv: https://arxiv.org/abs/1703.03864
- source: ../papers/arXiv-1703.03864v2/
- authors: Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever (OpenAI)
- venue / year: arXiv preprint, 2017 (v2)
- tags: [evolution-strategies, RL, black-box-optimization, distributed, neuroevolution]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: 深層 RL は Q-learning や Policy Gradient のような MDP ベース手法が主流だが、(a) バックプロップに依存するため低精度ハードや非微分モジュール（hard attention 等）と相性が悪く、(b) action 空間でノイズを入れるので長い episode で勾配推定の分散が時間ステップ T と共に線形に増える、(c) frameskip 等の action frequency に敏感で、(d) value function 推定が逐次的でスケール律速になる。NES 系の Evolution Strategies は古典的に neural-net RL に応用されてきたが、現代の深層 RL タスクで本当に張り合えるかは未検証だった。
- **手法**: パラメータ θ を多変量等方ガウス N(θ, σ²I) で摂動して F(θ+σε) を評価し、REINFORCE 型 score function estimator ∇_θ E[F] = (1/σ) E_ε[F(θ+σε)ε] で θ を直接 SGA 更新する（Algorithm 1）。並列化（Algorithm 2）では全 worker が**同じ random seed のセット**を持ち、各 worker は scalar return F_i だけを全 worker に broadcast、他 worker の摂動 ε_j を seed から再生成して更新を独立に再構築する。→ 通信量がスカラだけになり、1,440 並列でも線形スピードアップ。実装上は **antithetic (mirrored) sampling**、**rank-based fitness shaping**、weight decay を併用。σ は固定（学習中に適応しない）。Atari の DeepMind CNN では **virtual batch normalization** を使わないと摂動が「常に同じ action を返す policy」を作りやすく探索が壊れた。MuJoCo では一部 env で連続 action だと滑らかすぎたので action を 10 bin に離散化。
- **結果**:
  - **MuJoCo**（Table 1, 5M timesteps）: TRPO に到達するまでの ES timesteps / TRPO timesteps 比は、HalfCheetah 0.58、Hopper 6.94、InvertedDoublePendulum 1.23、InvertedPendulum 0.88、Swimmer 0.30、Walker2d 7.88（100% 到達点）。難しい env でも最大 ~10x のサンプル損で TRPO に追いつく。Swimmer/HalfCheetah では TRPO より良いサンプル効率。
  - **Atari 51 game**（Table 2, 1B frames ≈ 1 hour on 720 CPUs）: A3C FF（320M frames, 1 day）と比べて **23 game で勝ち、28 game で負け**。計算量はほぼ同じ（backprop と value func を持たないので 1 episode あたりの計算量が ~3x 軽い分、データ量を多く使える）。
  - **3D Humanoid（OpenAI Gym, score 6000 到達まで）**: 18 cores で 657 分 → 1,440 cores で 10 分。Figure 1 で CPU 数に対しほぼ線形スケール。
  - **frame-skip 不変性**: Pong で frame-skip ∈ {1,2,3,4} を変えても学習曲線がほぼ同じ（Figure 2）。
  - **探索の多様性**: MuJoCo Humanoid で TRPO では現れない歩き方（横歩き・後ろ歩き）を学習。
  - **頑健性**: Atari 全 env / MuJoCo 全 env で**同じハイパラ**で動く（MuJoCo は 1 binary param のみ env 依存）。
- **貢献**: (1) NES 系を「shared seeds + scalar broadcast」で本気で分散させ、深層 RL の最難 env でも policy-gradient と張り合えると実証、(2) action frequency 不変・長 horizon に強い・value func 不要・backprop 不要・低精度ハード親和という ES の構造的利点を整理（§3）、(3) ES が parameter space の Gaussian-smoothed objective に対する randomized finite difference 推定であるという解釈、および「parameter 数ではなく問題の intrinsic dimension が効く」という主張、(4) Atari/MuJoCo の Wallclock 時間を桁で短縮した分散実装の存在証明。

## Takeaway（自分にとっての要点）

- ES の本質は「**action 空間ではなく parameter 空間にノイズを置く**」こと。policy gradient の勾配分散は ∇log p(a;θ) が T 個の項の和なので Var が T に比例して悪化するが、ES の ∇log p(θ̃;θ) は episode 長と独立 → 長い horizon・遅延報酬・ skip 不変性が「タダで」付いてくる（§3.1）。これは PPO/A3C を改良するより先に検討すべき設計選択だった。
- **shared random seeds による分散** は神 trick。普通の data parallel SGD は gradient（=数百万次元ベクトル）を交換するが、ES は scalar F_i だけ。これだけで 1,440 並列が成立する。同じ発想は LLM の RL（GRPO 等）で worker 間 gradient sync を減らす設計に流用できそう。
- **virtual batch normalization** がないと Atari で ES が壊れる、は重要。「parameter perturbation が policy の出力分布を意味のある形で動かすかどうか」が ES の成否を決めるという観測で、policy parameterization の重要性に注意を促している。
- 「**parameter 数ではなく intrinsic dimension**」の主張（§3.2）：x→(x,x) で特徴量を倍にしても問題は難しくならない（ノイズ σ と学習率を半分にすればよい）、という説明は ES の高次元への適用根拠として強い。実際に Atari で**大きい A3C ネットの方が ES の結果が良くなった**のは、KAWAGUCHI 2016 の「大ネットの方が local minima が少ない」と合致。
- ES の「弱み」を素直に挙げているのが好印象：A3C より 3〜10x データを食う／51 game 中 28 game で負けた／Mujoco の一部 env では action 離散化が必要／σ 適応は効果なし／indirect encoding 未着手（future work）。
- **計算予算をデータ効率と引き換えにする**設計：「データ効率は下がるが、その分 1 episode のコストが軽いし、何より wallclock を桁で短縮できる」というトレードオフを正面から主張している。これは現代の LLM RL/RLHF が直面する「scaling = parallelism」の議論の原型。
- TRPO で出ない異常歩行（横歩き、後ろ歩き）を ES が見つけるという観察は、parameter-space exploration が action-space exploration と**質的に違う**ことを示唆。multimodal な policy 分布を陽に持たなくても多様性が出る。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 主張がシンプルで再現容易（Algorithm 1/2 は 6 行で書ける）。分散側の novelty が「seed を共有して scalar だけ broadcast」というワンアイディアに集中していて美しい。
  - 「ES vs PG はどちらが Monte Carlo gradient の分散が小さいか」を episode 長 T に対する依存性で議論した §3.1 が、後続の long-horizon RL の設計議論にそのまま使える理論的整理になっている。
  - サンプル効率を犠牲にしても wallclock が桁で速くなるという「軸の取り直し」を、Humanoid 657min→10min の具体数字で示している（Figure 1）。
  - ハイパラを env ごとに調整していない（Atari 全 env / MuJoCo はほぼ全 env で fix）という頑健性報告。
  - **論文自身が limitations を素直に書いている**点も評価できる：3-10x 多いデータが必要／51 game 中 28 game で A3C に負け／σ 適応は今回効かなかった／indirect encoding は future work。
- **弱み / 疑問**:
  - **データ効率の劣化が「3-10x」というのは env 依存で実際はもっと悪い場合もあり得る**。Walker2d で 7.88x、Hopper で 6.94x は「ぎりぎり許容」だが、もっと長い horizon・もっと sparse な報酬の env で同じ比が維持されるのかは未検証。
  - **Atari の 23 勝 28 敗は微妙**。Table 2 を見ると ES が大勝している env (Atlantis, Kangaroo, Frostbite) と完敗している env (Beam Rider, Demon Attack, Q*Bert) が極端で、なぜそうなるのかの分析が薄い。
  - **virtual batch normalization が「無いと壊れる」程度に効いているのに、その depend がアブレーションされていない**。ES の成功がアルゴリズムの本質なのか、policy parameterization の trick なのかが切り分けられない。
  - **σ を固定にしている**（CMA-ES や NES 系の本来の売りである共分散適応を捨てている）。"did not see benefit from adapting σ" と書かれているが、どの env で何を試したかの詳細が無い。
  - 「intrinsic dimension が支配的でパラメータ数に依存しない」という主張は §3.2 の x→(x,x) 例で directional には正しいが、**実問題で intrinsic dimension が小さいかどうか**は仮定でしか無い。一般の MDP では成り立たない可能性がある。
  - 比較対象が TRPO / A3C で、**より新しい / 強い baseline**（PPO はこの時点で出ていない、SAC/IMPALA も後）への評価が当然ながら無い。今読むなら IMPALA や DD-PPO との比較が欲しい。
  - **CPU 1,440 個という現実的でないリソース要件**（2017 年 OpenAI EC2）が、ES の良さを「scalable だ」と言い切る根拠の柱になっている。一般読者にとって 8 GPU の A3C の方が再現しやすく、wallclock 比較は不公平とも言える。
  - "communication" が seed のみで済むので帯域は確かに小さいが、**1,440 worker 全員が** scalar を全員に broadcast（all-to-all）するコストは worker 数の二乗オーダー。最大規模 (1,440) でこの all-to-all がどれだけスケールするかの計測は無い。
- **次に試したいこと**:
  - **同じ wallclock 予算で IMPALA / DD-PPO / GRPO** と並べた pareto curve（sample efficiency × wallclock）。
  - **大規模 LLM の RL 後段** で ES を使えるか：policy gradient より gradient 通信を抑えられるなら、推論サーバを worker にしてしまえる利点が大きい。問題は intrinsic dimension が LLM パラメータでは小さいと仮定しにくいこと。
  - **σ を per-coordinate / per-layer に分け、layer ごとの感度に応じてスケジュール**する（NES 本来の自動適応の軽量版）。固定 σ のままでも layer scale だけ調整する案。
  - virtual batch normalization の代替（GroupNorm / LayerNorm）でも Atari ES が動くかのアブレーション。「ES の本質 vs parameterization の trick」の切り分け。
  - **異常歩行（sideways/backwards walking）** が出る現象を、policy distribution の multimodality として定量化する（fitness landscape 上の basin 数 vs PG）。

## Notes / Quotes

- Algorithm 1 のコア update: θ_{t+1} ← θ_t + α · (1/(nσ)) Σ_i F_i ε_i with ε_i ~ N(0, I)（Algorithm 1, line 5）。
- "The only information obtained by each worker is the scalar return of an episode: if we synchronize random seeds between workers before optimization, each worker knows what perturbations the other workers used, so each worker only needs to communicate a single scalar to and from each other worker to agree on a parameter update."（§2.1）
- "ES is invariant to action frequency and delayed rewards, tolerant of extremely long horizons, and does not need temporal discounting or value function approximation."（Abstract）
- "Without these [virtual batch normalization] reparameterizations ES proved brittle in our experiments, but with these reparameterizations we achieved strong results over a wide variety of environments."（§1, finding 1）
- ES vs PG 分散比較（§3.1）: Var[∇F_PG] ≈ Var[R] · Var[∇log p(a;θ)] は T 個の項の和なので T に比例して増えるが、Var[∇F_ES] ≈ Var[R] · Var[∇log p(θ̃;θ)] は T と独立。
- "we did not see benefit from adapting σ during training, and we therefore treat it as a fixed hyperparameter instead."（§2.1, σ 適応について）
- Limitations 周辺の率直な記述: 51 Atari game 中「performed better on 23 games tested, and worse on 28」（§1）。データ効率は A3C 比 3-10x 悪化、「partly offset by a reduction in required computation of roughly 3x due to not performing backpropagation and not having a value function」(§1)。
- 1,440 CPU 並列で 3D Humanoid を 10 分（Figure 1）。18 cores だと 657 分（11 時間）。
- Future work（§6）: meta-learning（learning-to-learn）への応用、低精度 NN 実装との組み合わせ。

## Related Papers

- Wierstra et al., *Natural Evolution Strategies* (2008/2014) — 直接のアルゴリズム的源流。
- Sehnke et al., *Parameter-exploring Policy Gradients* (2010) — 同じ「parameter space に noise」発想で著者らが "closely related" と認める先行。
- Nesterov & Spokoiny, *Random Gradient-Free Minimization of Convex Functions* (2011) — ES の finite-difference 解釈の理論裏付け。
- Williams, *REINFORCE* (1992) — score function estimator の起源、ES の gradient 推定と数式上同型。
- Hansen & Ostermeier, *CMA-ES* (2001) — 共分散適応する ES の代表、本論文はあえて固定 σ で勝負。
- Schulman et al., *TRPO* (2015) — MuJoCo の主 baseline。
- Mnih et al., *DQN* (2015) / *A3C* (2016) — Atari baseline と CNN アーキの出所。
- Salimans et al., *Improved Techniques for Training GANs* (2016) — virtual batch normalization の出典。
- Koutník et al. (2013), Hausknecht et al. (2014) — Atari への neuroevolution の先行（HyperNEAT 等）。
- Brockman et al., *OpenAI Gym* (2016) — 評価環境。
- Todorov et al., *MuJoCo* (2012) — 物理シミュレータ。
- Stanley et al., *HyperNEAT* (2009) — indirect encoding の代表、本論文では future work 扱い。
