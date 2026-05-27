# Adam: A Method for Stochastic Optimization

- arXiv: https://arxiv.org/abs/1412.6980
- source: ../papers/arXiv-1412.6980v9/
- authors: Diederik P. Kingma, Jimmy Lei Ba
- venue / year: ICLR 2015
- tags: [optimization, stochastic-gradient, adaptive-learning-rate, deep-learning]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 高次元・大規模・ノイズあり・スパース勾配の確率的目的関数を効率よく最適化したい。AdaGrad はスパース勾配に強く、RMSProp はオンライン/非定常設定に強い。両者の利点を統合し、メモリ要求が小さく、ハイパーパラメータ調整が少ない1次法を提示する。
- **手法**: 勾配 $g_t$ の指数移動平均 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$（1次モーメント=平均推定）と二乗勾配の指数移動平均 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$（2次 raw モーメント=非中心分散推定）を保持し、ゼロ初期化バイアスを $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$ で補正、$\theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat m_t / (\sqrt{\hat v_t}+\epsilon)$ で更新（Algorithm 1）。デフォルトは $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$。派生として $L^\infty$ ノルムに置き換えた **AdaMax**（Algorithm 2, $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$, デフォルト $\alpha=0.002$）も提案。
- **結果**: (a) ステップサイズが $|\Delta_t| \lesssim \alpha$ の「trust region」で上から押さえられ、勾配スケールに不変、SNR ($\hat m_t/\sqrt{\hat v_t}$) が小さいほど step が縮む自動 annealing が起きると解析。(b) オンライン凸学習で regret $R(T) = O(\sqrt T)$、平均 regret $O(1/\sqrt T)$ を証明（Theorem 4.1, Corollary 4.2、ただし $\alpha_t = \alpha/\sqrt t$、$\beta_{1,t} = \beta_1 \lambda^{t-1}$ で減衰、$\beta_1^2/\sqrt{\beta_2} < 1$ を仮定）。(c) 実験: MNIST ロジスティック回帰では SGD-Nesterov と同等で AdaGrad より速く（Figure 1 左）、IMDB BoW + dropout のスパース問題では Adam+dropout が最速級（Figure 1 右）、MNIST 多層 NN（FC×2, 1000 ユニット, ReLU, minibatch 128, dropout あり/なし）で他手法を上回り SFO よりも反復・wall-clock とも速い（Figure 2）、CIFAR-10 CNN (c64-c64-c128-1000) では序盤 AdaGrad と互角だが 45 epoch では Adam と SGD-Nesterov が最終的に有利（Figure 3）、VAE で $\beta_2$ を 1 に近づけたときバイアス補正を切ると不安定（Figure 4, §6.4）。
- **貢献**: (1) 1次・2次モーメントの指数移動平均 + バイアス補正という単純で計算/メモリ効率のよい更新則を提案、(2) 凸オンライン設定での regret bound $O(\sqrt T)$ を証明、(3) AdaGrad/RMSProp との関係を明示し AdaGrad は Adam の $\beta_1=0,\ (1-\beta_2)\to 0$ の極限に対応すると示す（§5）、(4) AdaMax（$L^\infty$ 版、バイアス補正不要、$|\Delta_t|\le\alpha$）と temporal averaging（§7.2）を派生として提示。

## Takeaway（自分にとっての要点）

- 「Adam の実効ステップ幅は $\alpha$ で上から押さえられる」というのが直感の核：勾配の生スケールではなく **SNR ベース**で動くので、層ごとにスケールがバラつくモデルに強い。
- バイアス補正 $1/(1-\beta_2^t)$ は **特に $\beta_2$ が 1 に近いとき**に効く。Figure 4 がその唯一の直接実験で、§6.4 では「補正なしだと序盤 instability、勾配がスパースになる後半で特に顕著」と述べている。RMSProp との実質的差分はここ。
- AdaGrad の累積型更新に対して、Adam の $v_t$ は指数移動平均であり、著者は RMSProp の利点として online / non-stationary settings に強い点を挙げている。
- AdaMax は $|\Delta_t|\le\alpha$ がより clean に成立し、バイアス補正項が要らない。$L^p,\ p\to\infty$ の極限で導出されるが、本文の実験節では AdaMax と Adam の比較図表は示されていない。
- 収束証明は **凸**前提（オンライン凸最適化フレームワーク, Zinkevich 2003）。多層 NN の結果はすべて経験的で、「証明の射程外であることを明示」している点は誠実。
- $\epsilon=10^{-8}$ は小さい既定値だが、§6.3 で CNN の場合 $\hat v_t$ がほぼ 0 に消えて $\epsilon$ が支配的になることが報告されている。CNN では Adam の優位は marginal とも述べられている。
- temporal averaging（Polyak-Ruppert / EMA）を 1行追加で適用できると §7.2 で言及している。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 既定ハイパラほぼそのままで広い問題に効く実用性。論文中でも「the hyper-parameters have intuitive interpretations and typically require little tuning」と主張している。
  - RMSProp と AdaGrad のいいとこ取りという立て付けが明快で、§5 の Related work で両者の極限としての位置づけを数式で示しているのは説得力が高い。
  - 理論（regret bound）と実装（10 行弱の擬似コード）と実験（凸〜非凸まで4種）が一冊にまとまっていて、後続研究者にとって参照点として完成度が高い。
  - 「step size が trust region」という解釈は実務でハイパラを決める直感に直結する。
- **弱み / 疑問**:
  - **収束証明は凸前提**。著者も「does not apply to non-convex problems」と明記しており、多層 NN の実験は経験的結果にとどまる。
  - **CNN での優位が薄い**ことを §6.3 で認めている（"Adam shows marginal improvement over SGD with momentum"）。同じ箇所で、$\hat v_t$ が数 epoch 後に 0 に近づき $\epsilon$ 支配になるとも述べている。
  - $\beta_1^2/\sqrt{\beta_2} < 1$ という Theorem 4.1 の制約は、デフォルト $\beta_1=0.9, \beta_2=0.999$ では $0.81/0.9995 \approx 0.81 < 1$ で満たすが、$\beta_1$ を大きくしたいとき自由度が狭い。
  - §6.2 では "L2 weight decay" を用いたと書かれているが、その正則化項と Adam 更新則の相互作用は本文中で追加分析されていない。
  - 比較対象が SGD-Nesterov, AdaGrad, RMSProp, AdaDelta, SFO に限られている。特に画像系で、SGD の learning-rate schedule を追加で変えた比較は本文中に示されていない（評者補足）。
  - AdaMax は提案だけで実験が無い。$|\Delta_t|\le\alpha$ の利点を実証してほしかった。
  - VAE のバイアス補正実験（Figure 4）が唯一の ablation で、$\beta_2$ が 1 に近い特殊条件のみ。一般設定でバイアス補正の効き目が定量化されていない。
- **次に試したいこと**:
  - 同じ Adam で **$\epsilon$ を変えた** ablation（特に CNN で $\epsilon$ を大きくしたら SGD+momentum との差は埋まる/開く？）。本論文では $\epsilon$ の影響は議論されていない。
  - Theorem 4.1 で出てくる $\beta_{1,t} = \beta_1 \lambda^{t-1}$ の **$\beta_1$ 減衰**を実装した場合の実験的効果（証明では重要なのに実験では使われていない）。
  - AdaMax と Adam の同条件比較（MNIST FC, CIFAR-10 CNN）で「ノルム置き換え」の効果を切り分け。
  - temporal averaging（§7.2）を ON にしたときの汎化への影響。§7.2 では方法だけが述べられており、実験結果は示されていない（評者補足）。

## Notes / Quotes

- "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments." (Abstract)
- "the name Adam is derived from adaptive moment estimation." (§1)
- "Good default settings for the tested machine learning problems are $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$." (Algorithm 1 キャプション)
- "the effective magnitude of the steps taken in parameter space at each timestep are approximately bounded by the stepsize setting $\alpha$, i.e., $|\Delta_t| \lessapprox \alpha$. This can be understood as establishing a *trust region* around the current parameter value." (§2.1)
- "With a smaller SNR the effective stepsize $\Delta_t$ will be closer to zero ... a form of automatic annealing." (§2.1)
- "AdaGrad corresponds to a version of Adam with $\beta_1 = 0$, infinitesimal $(1-\beta_2)$ and a replacement of $\alpha$ by an annealed version $\alpha_t = \alpha \cdot t^{-1/2}$." (§5)
- "Although our convergence analysis does not apply to non-convex problems, we empirically found that Adam often outperforms other methods in such cases." (§6.2)
- "We notice the second moment estimate $\hat v_t$ vanishes to zeros after a few epochs and is dominated by the $\epsilon$ in algorithm 1." (§6.3, CNN 実験)
- AdaMax: $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$, $\theta_t \leftarrow \theta_{t-1} - (\alpha/(1-\beta_1^t)) \cdot m_t / u_t$, $|\Delta_t|\le\alpha$（§7.1, Algorithm 2）
- Acknowledgments で "Thanks to Kai Fan from Duke University for spotting an error in the original AdaMax derivation." — 投稿後修正された経緯がある。
- Theorem 4.1 の仮定: $\|\nabla f_t\|_2 \le G$, $\|\nabla f_t\|_\infty \le G_\infty$, $\|\theta_n - \theta_m\|_2 \le D$, $\|\theta_m - \theta_n\|_\infty \le D_\infty$, $\beta_1^2/\sqrt{\beta_2} < 1$, $\alpha_t=\alpha/\sqrt t$, $\beta_{1,t}=\beta_1 \lambda^{t-1}$。
- (verified 2026-05-20) §6.2 引用の冒頭を原文通り "Although our convergence analysis..." に修正（0_adam_main.pdf p.6, §6.2 本文）。他の記述（Algorithm 1/2、Theorem 4.1 仮定、§2.1 trust region、§5 AdaGrad-as-Adam-limit、§6.1〜6.4 実験条件、Acknowledgments）は PDF 本文と一致を確認。
- (verified 2026-05-26) 後続研究・現代慣行・「暗に示す」系の TeX/PDF 根拠外の評価を削除または評者補足として明示（arxiv.tex, 0_adam_main.pdf §5, §6.2, §6.3, §7.1, §7.2）。この arXiv ソースには main.bbl は無く、参考文献は `arxiv.tex` が include する PDF の References 節で確認。

## Related Papers

- Duchi+ 2011, **AdaGrad** — スパース勾配対応の関連手法。Adam は $\beta_1=0$ 極限で AdaGrad に一致（§5）。
- Tieleman & Hinton 2012, **RMSProp** — Adam と直接関係する最適化手法。Adam はバイアス補正と1次モーメント運動量を加えた拡張。
- Graves 2013 — RMSProp + momentum 版、比較対象。
- Zeiler 2012, **AdaDelta** — 比較対象（§6.2）。
- Schaul+ 2012, **vSGD** — 比較対象。
- Sohl-Dickstein+ 2014, **SFO**（Sum-of-Functions Optimizer） — 比較対象。準ニュートン、メモリ線形でGPU不利。
- Amari 1998, **Natural Gradient** / Pascanu & Bengio 2013 — 前処理行列としての位置づけ（Adam は対角 Fisher 情報の逆数の平方根近似）。
- Zinkevich 2003 — オンライン凸最適化フレームワーク（regret 解析の土台）。
- Polyak & Juditsky 1992 / Ruppert 1988 / Moulines & Bach 2011 — Polyak-Ruppert averaging（§7.2 temporal averaging の元）。
- Kingma & Welling 2013, **VAE** — §6.4 のバイアス補正実験対象モデル。
