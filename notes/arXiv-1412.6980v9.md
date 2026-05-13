# Adam: A Method for Stochastic Optimization

- arXiv: https://arxiv.org/abs/1412.6980
- source: ../papers/arXiv-1412.6980v9/
- authors: Diederik P. Kingma, Jimmy Lei Ba
- venue / year: ICLR 2015
- tags: [optimization, stochastic-gradient, adaptive-learning-rate, deep-learning]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 高次元・大規模・ノイズあり・スパース勾配の確率的目的関数を効率よく最適化したい。SGD/モーメンタムはハイパラ調整が要るし、AdaGrad はスパースに強いが学習率が単調減衰して非定常目的に弱く、RMSProp はオンライン/非定常に強いがバイアス補正がなくスパースで挙動が悪い。両者の利点を統合した「メモリも計算も軽い1次法」が欲しい。
- **手法**: 勾配 $g_t$ の指数移動平均 $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$（1次モーメント=平均推定）と二乗勾配の指数移動平均 $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$（2次 raw モーメント=非中心分散推定）を保持し、ゼロ初期化バイアスを $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$ で補正、$\theta_t \leftarrow \theta_{t-1} - \alpha \cdot \hat m_t / (\sqrt{\hat v_t}+\epsilon)$ で更新（Algorithm 1）。デフォルトは $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$。派生として $L^\infty$ ノルムに置き換えた **AdaMax**（Algorithm 2, $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$, デフォルト $\alpha=0.002$）も提案。
- **結果**: (a) ステップサイズが $|\Delta_t| \lesssim \alpha$ の「trust region」で上から押さえられ、勾配スケールに不変、SNR ($\hat m_t/\sqrt{\hat v_t}$) が小さいほど step が縮む自動 annealing が起きると解析。(b) オンライン凸学習で regret $R(T) = O(\sqrt T)$、平均 regret $O(1/\sqrt T)$ を証明（Theorem 4.1, Corollary 4.2、ただし $\alpha_t = \alpha/\sqrt t$、$\beta_{1,t} = \beta_1 \lambda^{t-1}$ で減衰、$\beta_1^2/\sqrt{\beta_2} < 1$ を仮定）。(c) 実験: MNIST ロジスティック回帰では SGD-Nesterov と同等で AdaGrad より速く（Figure 1 左）、IMDB BoW + dropout のスパース問題では Adam+dropout が最速級（Figure 1 右）、MNIST 多層 NN（FC×2, 1000 ユニット, ReLU, minibatch 128, dropout あり/なし）で他手法を上回り SFO よりも反復・wall-clock とも速い（Figure 2）、CIFAR-10 CNN (c64-c64-c128-1000) では序盤 AdaGrad と互角だが 45 epoch では Adam と SGD-Nesterov が最終的に有利（Figure 3）、VAE で $\beta_2$ を 1 に近づけたときバイアス補正を切ると不安定（Figure 4, §6.4）。
- **貢献**: (1) 1次・2次モーメントの指数移動平均 + バイアス補正という単純で計算/メモリ効率のよい更新則を提案、(2) 凸オンライン設定での regret bound $O(\sqrt T)$ を証明、(3) AdaGrad/RMSProp との関係を明示し AdaGrad は Adam の $\beta_1=0,\ (1-\beta_2)\to 0$ の極限に対応すると示す（§5）、(4) AdaMax（$L^\infty$ 版、バイアス補正不要、$|\Delta_t|\le\alpha$）と temporal averaging（§7.2）を派生として提示。

## Takeaway（自分にとっての要点）

- 「Adam の実効ステップ幅は $\alpha$ で上から押さえられる」というのが直感の核：勾配の生スケールではなく **SNR ベース**で動くので、層ごとにスケールがバラつくモデルに強い。
- バイアス補正 $1/(1-\beta_2^t)$ は **特に $\beta_2$ が 1 に近いとき**に効く。Figure 4 がその唯一の直接実験で、§6.4 では「補正なしだと序盤 instability、勾配がスパースになる後半で特に顕著」と述べている。RMSProp との実質的差分はここ。
- AdaGrad の累積はリセットされないが、Adam の $v_t$ は指数 forget するので **非定常目的（特に深層NNの learning dynamics）に追従できる**。逆に「無限に古い情報を残したい」場合は AdaGrad が良い、と暗に示している。
- AdaMax は $|\Delta_t|\le\alpha$ がより clean に成立し、バイアス補正項が要らない。$L^p,\ p\to\infty$ の極限で出てくるという導出は綺麗だが、本論文では AdaMax 自体の実験はなく、推奨される理論的派生にとどまる。
- 収束証明は **凸**前提（オンライン凸最適化フレームワーク, Zinkevich 2003）。多層 NN の結果はすべて経験的で、「証明の射程外であることを明示」している点は誠実。
- $\epsilon=10^{-8}$ は小さい既定値だが、§6.3 で CNN の場合 $\hat v_t$ がほぼ 0 に消えて $\epsilon$ が支配的になることが報告されている。CNN では Adam の優位は marginal とも自白している。
- temporal averaging（Polyak-Ruppert / EMA）を 1行追加で適用できると 7.2 で言及。最近の EMA 流行を考えると当時から指摘されていたのは興味深い。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 既定ハイパラほぼそのままで広い問題に効く実用性。論文中でも「the hyper-parameters have intuitive interpretations and typically require little tuning」と主張しており、その後の標準化を見るとこの主張は正当だった。
  - RMSProp と AdaGrad のいいとこ取りという立て付けが明快で、§5 の Related work で両者の極限としての位置づけを数式で示しているのは説得力が高い。
  - 理論（regret bound）と実装（10 行弱の擬似コード）と実験（凸〜非凸まで4種）が一冊にまとまっていて、後続研究者にとって参照点として完成度が高い。
  - 「step size が trust region」という解釈は実務でハイパラを決める直感に直結する。
- **弱み / 疑問**:
  - **収束証明は凸前提**。著者も「does not apply to non-convex problems」と明記。実際の主戦場は深層 NN なので、理論パートは保険程度。事実、後に Reddi et al. 2018 (AMSGrad) が反例を出して証明の穴を指摘した（本論文時点では未指摘）。
  - **CNN での優位が薄い**ことを §6.3 で自白している（"Adam shows marginal improvement over SGD with momentum"）。$\hat v_t \to 0$ で $\epsilon$ 支配になるという観察は、ConvNet/ViT で SGD+momentum が長く生き残った理由とも整合する。
  - $\beta_1^2/\sqrt{\beta_2} < 1$ という Theorem 4.1 の制約は、デフォルト $\beta_1=0.9, \beta_2=0.999$ では $0.81/0.9995 \approx 0.81 < 1$ で満たすが、$\beta_1$ を大きくしたいとき自由度が狭い。
  - **L2 正則化と weight decay の混同**（後の AdamW 論文で批判される点）が §6.2 で "L2 weight decay" として無造作に併用されており、当時は問題視されていない。
  - 比較対象が SGD-Nesterov, AdaGrad, RMSProp, AdaDelta, SFO に限られ、運動量 + 学習率スケジュールを丁寧に組んだ SGD との fair comparison（特に画像系）が薄い。
  - AdaMax は提案だけで実験が無い。$|\Delta_t|\le\alpha$ の利点を実証してほしかった。
  - VAE のバイアス補正実験（Figure 4）が唯一の ablation で、$\beta_2$ が 1 に近い特殊条件のみ。一般設定でバイアス補正の効き目が定量化されていない。
- **次に試したいこと**:
  - 同じ Adam で **$\epsilon$ を変えた** ablation（特に CNN で $\epsilon$ を大きくしたら SGD+momentum との差は埋まる/開く？）。本論文では $\epsilon$ の影響は議論されていない。
  - Theorem 4.1 で出てくる $\beta_{1,t} = \beta_1 \lambda^{t-1}$ の **$\beta_1$ 減衰**を実装した場合の実験的効果（証明では重要なのに実験では使われていない）。
  - AdaMax と Adam の同条件比較（MNIST FC, CIFAR-10 CNN）で「ノルム置き換え」の効果を切り分け。
  - temporal averaging（§7.2）を ON にしたときの汎化への影響。当時実験がないので、現代の EMA との橋渡しに使える。

## Notes / Quotes

- "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments." (Abstract)
- "the name Adam is derived from adaptive moment estimation." (§1)
- "Good default settings for the tested machine learning problems are $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$." (Algorithm 1 キャプション)
- "the effective magnitude of the steps taken in parameter space at each timestep are approximately bounded by the stepsize setting $\alpha$, i.e., $|\Delta_t| \lessapprox \alpha$. This can be understood as establishing a *trust region* around the current parameter value." (§2.1)
- "With a smaller SNR the effective stepsize $\Delta_t$ will be closer to zero ... a form of automatic annealing." (§2.1)
- "AdaGrad corresponds to a version of Adam with $\beta_1 = 0$, infinitesimal $(1-\beta_2)$ and a replacement of $\alpha$ by an annealed version $\alpha_t = \alpha \cdot t^{-1/2}$." (§5)
- "Our convergence analysis does not apply to non-convex problems, we empirically found that Adam often outperforms other methods in such cases." (§6.2)
- "We notice the second moment estimate $\hat v_t$ vanishes to zeros after a few epochs and is dominated by the $\epsilon$ in algorithm 1." (§6.3, CNN 実験での自白)
- AdaMax: $u_t = \max(\beta_2 \cdot u_{t-1}, |g_t|)$, $\theta_t \leftarrow \theta_{t-1} - (\alpha/(1-\beta_1^t)) \cdot m_t / u_t$, $|\Delta_t|\le\alpha$（§7.1, Algorithm 2）
- Acknowledgments で "Thanks to Kai Fan from Duke University for spotting an error in the original AdaMax derivation." — 投稿後修正された経緯がある。
- Theorem 4.1 の仮定: $\|\nabla f_t\|_2 \le G$, $\|\nabla f_t\|_\infty \le G_\infty$, $\|\theta_n - \theta_m\|_2 \le D$, $\|\theta_m - \theta_n\|_\infty \le D_\infty$, $\beta_1^2/\sqrt{\beta_2} < 1$, $\alpha_t=\alpha/\sqrt t$, $\beta_{1,t}=\beta_1 \lambda^{t-1}$。

## Related Papers

- Duchi+ 2011, **AdaGrad** — スパース勾配対応の祖。Adam は $\beta_1=0$ 極限で AdaGrad に一致（§5）。
- Tieleman & Hinton 2012, **RMSProp** — 2 次モーメント EMA の祖。Adam はバイアス補正と1次モーメント運動量を加えた拡張。
- Graves 2013 — RMSProp + momentum 版、比較対象。
- Zeiler 2012, **AdaDelta** — 比較対象（§6.2）。
- Schaul+ 2012, **vSGD** — 比較対象。
- Sohl-Dickstein+ 2014, **SFO**（Sum-of-Functions Optimizer） — 比較対象。準ニュートン、メモリ線形でGPU不利。
- Amari 1998, **Natural Gradient** / Pascanu & Bengio 2013 — 前処理行列としての位置づけ（Adam は対角 Fisher 情報の逆数の平方根近似）。
- Zinkevich 2003 — オンライン凸最適化フレームワーク（regret 解析の土台）。
- Polyak & Juditsky 1992 / Ruppert 1988 / Moulines & Bach 2011 — Polyak-Ruppert averaging（§7.2 temporal averaging の元）。
- Kingma & Welling 2013, **VAE** — §6.4 のバイアス補正実験対象モデル。
