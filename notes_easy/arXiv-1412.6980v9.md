# Adam: A Method for Stochastic Optimization（確率的目的関数に対する adaptive moment estimation）

- arXiv: https://arxiv.org/abs/1412.6980
- 一次ソース: ../papers/arXiv-1412.6980v9/
- 正規ノート: ../notes/arXiv-1412.6980v9.md

---

## 一言で言うと

Adam は、確率的目的関数の一次勾配だけを使い、勾配の 1 次モーメントと 2 次 raw モーメントの指数移動平均からパラメータごとの adaptive learning rate を作る最適化手法である。著者は、Adam が sparse gradients、non-stationary objectives、very noisy gradients に適し、オンライン凸最適化では regret $O(\sqrt{T})$ を持ち、実験でも複数の機械学習モデルで既存の stochastic optimization methods と有利に比較できると主張している。

## 何を議論する論文か

- **問題設定**: 確率的で微分可能な scalar objective function $f(\theta)$ の期待値 $E[f(\theta)]$ を、パラメータ $\theta$ について最小化する問題を扱う。各時刻 $t$ では、ランダムな minibatch や関数ノイズから生じる実現 $f_t(\theta)$ と勾配 $g_t=\nabla_\theta f_t(\theta)$ を観測する。
- **対象範囲 / 仮定**: 主対象は、大規模データまたは高次元パラメータを持つ機械学習問題である。本文 §1 は、この設定では higher-order optimization methods が不向きであり、議論を first-order methods に限ると述べる。理論解析 §4 はオンライン凸最適化の枠組みで、$f_t$ の凸性、勾配有界性、パラメータ間距離の有界性などを仮定する。§6.2 が non-convex objective functions と呼ぶ multi-layer neural network 実験は、この理論の射程外である。
- **既存研究との差分**: Adam は、AdaGrad の sparse gradients への強さと、RMSProp の online / non-stationary settings への強さを組み合わせることを狙う。RMSProp with momentum との違いとして、Adam は rescaled gradient への momentum ではなく、勾配そのものの 1 次・2 次モーメント推定を更新し、さらに bias-correction term を持つ。AdaGrad との関係は §5 で、$\beta_1=0$、$(1-\beta_2)$ を無限小に近づけ、$\alpha_t=\alpha t^{-1/2}$ とする極限として説明される。
- **この論文で答えたい問い**: ノイズがあり、スパースで、非定常でもありうる高次元の確率的目的関数に対して、計算・メモリ効率がよく、ハイパーパラメータの直感的な意味を保ち、既存の stochastic first-order methods と競争できる adaptive method を構成できるか、またその収束性と実験的挙動をどこまで説明できるか。

## 背景と前提

- この論文を読む前に必要な概念
  - **stochastic gradient-based optimization**: 全データの目的関数ではなく、minibatch やノイズ付き実現 $f_t$ の勾配で更新する最適化である。本文では SGD / ascent を代表例として置く。
  - **first-order methods**: 目的関数の一次導関数、つまり勾配だけを使う手法である。Adam も Hessian などの高次情報は使わない。
  - **exponential moving average**: 古い値を指数的に忘れながら平均を更新する。Adam では $m_t$ と $v_t$ がそれぞれ勾配と二乗勾配の指数移動平均である。
  - **moment**: 本文での 1st moment は勾配の mean、2nd raw moment は uncentered variance としての二乗勾配の平均である。中心化分散ではない。
  - **online convex optimization と regret**: 逐次的に未知の凸損失 $f_t$ に対して点 $\theta_t$ を選び、後から見た最良固定点 $\theta^*$ と比較する枠組みである。§4 の理論保証はこの枠組みに依存する。
- この論文での用語の使われ方
  - **Adam**: "adaptive moment estimation" から来る名称で、Algorithm 1 の更新則を指す。
  - **bias correction**: $m_0=v_0=0$ から始めるため、初期ステップでモーメント推定が 0 側に偏る問題を、$1-\beta_1^t$ と $1-\beta_2^t$ で割って補正する操作である。
  - **SNR**: §2.1 では、やや用語を緩く使うと断った上で、$\hat{m}_t/\sqrt{\hat{v}_t}$ を signal-to-noise ratio と呼ぶ。SNR が小さいほど $\Delta_t$ が 0 に近づく。
  - **trust region**: §2.1 では、更新量の有効な大きさが stepsize setting $\alpha$ でおおよそ抑えられることを、current parameter value の周りの trust region と解釈する。
- 先行研究や baseline との関係
  - **AdaGrad** は sparse gradients に強い手法として位置づけられる。本文 §5 では、Adam の特定極限が AdaGrad の基本更新に対応すると説明される。
  - **RMSProp** は Adam と直接関係する手法で、online and non-stationary settings に強いとされる。ただし RMSProp は bias-correction term を欠き、$\beta_2$ が 1 に近い sparse gradients の設定では大きな初期ステップや divergence につながると著者は述べる。
  - **SGD with Nesterov momentum, AdaGrad, RMSProp, AdaDelta, SFO** が主な実験比較対象である。SFO は quasi-Newton method だが、minibatch partition 数に線形なメモリ要求があり、GPU のようなメモリ制約環境では不利だと §5 と §6.2 で説明される。

## 提案手法

### コアアイデア

Adam は、各パラメータ次元ごとに、勾配 $g_t$ の指数移動平均 $m_t$ と、二乗勾配 $g_t^2$ の指数移動平均 $v_t$ を保持する。$m_t$ は更新方向の平均推定、$v_t$ は勾配の 2nd raw moment 推定であり、更新では $\hat{m}_t$ を $\sqrt{\hat{v}_t}+\epsilon$ で割る。これにより、勾配の大きさが大きい次元では有効ステップが抑えられ、勾配スケールを対角的に再スケールしても更新量が変わりにくい。

ただし、$m_t$ と $v_t$ は 0 初期化されるため、初期時刻では 0 側に偏る。Algorithm 1 はこの bias を $\hat{m}_t=m_t/(1-\beta_1^t)$、$\hat{v}_t=v_t/(1-\beta_2^t)$ で補正する。著者は、この補正が特に $\beta_2$ が 1 に近い場合、すなわち sparse gradients に対して長い平均窓が必要な場合に重要だと述べる。

既定値として Algorithm 1 は、テストされた機械学習問題に対し $\alpha=0.001,\ \beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8}$ を挙げる。すべてのベクトル演算は element-wise である。

### 重要な定義・数式

$$
\begin{aligned}
g_t &= \nabla_\theta f_t(\theta_{t-1}),\\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t,\\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t},\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^t},\\
\theta_t &= \theta_{t-1} - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\end{aligned}
$$

**式の意味**: Algorithm 1 の Adam 更新則である。勾配の 1 次モーメント推定 $m_t$ と 2 次 raw モーメント推定 $v_t$ を指数移動平均で更新し、それぞれを bias correction した上でパラメータを更新する。

**記号の定義**:
- $f_t(\theta)$ ... 時刻 $t$ における stochastic objective の実現
- $g_t$ ... $f_t$ の $\theta$ に関する勾配
- $m_t$ ... biased first moment estimate
- $v_t$ ... biased second raw moment estimate
- $\hat{m}_t,\hat{v}_t$ ... bias-corrected first moment estimate と second raw moment estimate
- $\alpha$ ... stepsize
- $\beta_1,\beta_2\in[0,1)$ ... moment estimates の exponential decay rates
- $\epsilon$ ... 分母の数値安定化項。Algorithm 1 の既定値は $10^{-8}$

**この論文での役割**: この式が提案手法そのものであり、§2 の説明、§3 の bias correction、§4 の regret 解析、§6 の全実験の基盤になる。比較対象の AdaGrad や RMSProp との差分も、この式の $m_t$、$v_t$、bias correction の有無から説明される。

$$
\Delta_t = \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}},\qquad
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}}\ \text{is called the signal-to-noise ratio (SNR) in §2.1}.
$$

**式の意味**: §2.1 が $\epsilon=0$ と仮定して考える、パラメータ空間での有効ステップである。著者は、この有効ステップが stepsize setting $\alpha$ でおおよそ上から抑えられ、SNR が小さいほど $\Delta_t$ が 0 に近づくと説明する。

**記号の定義**:
- $\Delta_t$ ... 時刻 $t$ での effective step
- $\hat{m}_t$ ... bias-corrected first moment estimate
- $\hat{v}_t$ ... bias-corrected second raw moment estimate
- $\alpha$ ... 更新量の大きさを支配する stepsize setting
- SNR ... §2.1 で $\hat{m}_t/\sqrt{\hat{v}_t}$ に与えられる呼び名

**この論文での役割**: Adam の実用上の直感を支える式である。著者は $|\Delta_t|\lessapprox\alpha$ を trust region のように解釈し、また勾配を定数倍しても $\hat{m}_t$ と $\sqrt{\hat{v}_t}$ が同じ倍率で変わるため、更新が gradient scale に不変だと説明する。

$$
\begin{aligned}
v_t &= (1-\beta_2)\sum_{i=1}^{t}\beta_2^{t-i}g_i^2
&&\text{(Eq. 1)}\\
E[v_t] &= E[g_t^2]\cdot(1-\beta_2^t)+\zeta
&&\text{(Eq. 4)}
\end{aligned}
$$

**式の意味**: §3 の initialization bias correction の導出である。$v_0=0$ から始める指数移動平均 $v_t$ は、真の second raw moment $E[g_t^2]$ そのものではなく、初期化由来の係数 $(1-\beta_2^t)$ を伴う。

**記号の定義**:
- $v_t$ ... 二乗勾配の指数移動平均
- $\beta_2$ ... second raw moment estimate の exponential decay rate
- $g_i^2$ ... 時刻 $i$ の勾配の element-wise square
- $E[v_t]$ ... $v_t$ の期待値
- $E[g_t^2]$ ... underlying gradient distribution に関する second raw moment
- $\zeta$ ... true second moment が stationary なら 0、非定常なら過去の重みによるずれを表す項

**この論文での役割**: $\hat{v}_t=v_t/(1-\beta_2^t)$ で割る理由を示す式である。§5 の RMSProp との差分、§6.4 の bias-correction term の実験、AdaGrad との極限対応の説明に直接つながる。

$$
R(T)=\sum_{t=1}^{T}\left[f_t(\theta_t)-f_t(\theta^*)\right],
\qquad
\theta^*=\arg\min_{\theta\in\mathcal{X}}\sum_{t=1}^{T}f_t(\theta).
\tag{5}
$$

**式の意味**: §4 で用いられる regret の定義である。逐次的に選んだ $\theta_t$ の損失累積と、後から見た feasible set $\mathcal{X}$ 内の最良固定点 $\theta^*$ の損失累積との差を測る。

**記号の定義**:
- $R(T)$ ... $T$ ステップまでの regret
- $f_t$ ... 時刻 $t$ で現れる convex cost function
- $\theta_t$ ... アルゴリズムが時刻 $t$ に選ぶパラメータ
- $\theta^*$ ... $\mathcal{X}$ 内で累積損失を最小にする固定パラメータ
- $\mathcal{X}$ ... feasible set

**この論文での役割**: Theorem 4.1 と Corollary 4.2 の評価量である。著者は、$\alpha_t=\alpha/\sqrt{t}$、$\beta_{1,t}=\beta_1\lambda^{t-1}$、$\beta_1^2/\sqrt{\beta_2}<1$ などの仮定のもとで Adam が $O(\sqrt{T})$ regret bound を持ち、平均 regret が $O(1/\sqrt{T})$ で 0 に向かうと主張する。

$$
u_t=\max(\beta_2 u_{t-1}, |g_t|),
\qquad
\theta_t\leftarrow \theta_{t-1}-\frac{\alpha}{1-\beta_1^t}\frac{m_t}{u_t}.
\tag{12 / Algorithm 2}
$$

**式の意味**: §7.1 の AdaMax 更新である。Adam の $L^2$ norm based update rule を $L^p$ に一般化し、$p\to\infty$ の極限を取ることで、exponentially weighted infinity norm $u_t$ を使う更新が得られる。

**記号の定義**:
- $u_t$ ... exponentially weighted infinity norm
- $\beta_2$ ... $u_t$ の decay rate
- $|g_t|$ ... 勾配の element-wise absolute value
- $m_t$ ... Adam と同じ biased first moment estimate
- $\alpha/(1-\beta_1^t)$ ... first moment の bias correction を含む learning rate

**この論文での役割**: Adam の派生手法 AdaMax を定義する式である。著者は AdaMax では second moment の initialization bias correction が不要で、更新量に $|\Delta_t|\le\alpha$ という Adam より単純な上限があると述べる。ただし §6 の実験では AdaMax の比較結果は示されていない。

### 実装 / アルゴリズム上の要点

- step1: $m_0=0,\ v_0=0,\ t=0$ で初期化する。$\theta_0$ は初期パラメータである。
- step2: 収束するまで、$t\leftarrow t+1$ として、現在の stochastic objective $f_t(\theta_{t-1})$ に対する勾配 $g_t=\nabla_\theta f_t(\theta_{t-1})$ を計算する。
- step3: $m_t$ と $v_t$ を指数移動平均で更新する。$g_t^2$ は element-wise square である。
- step4: $\hat{m}_t$ と $\hat{v}_t$ を $1-\beta_1^t$、$1-\beta_2^t$ で割って bias correction する。
- step5: $\theta_t=\theta_{t-1}-\alpha\hat{m}_t/(\sqrt{\hat{v}_t}+\epsilon)$ で更新する。
- 実装上は、Algorithm 1 の最後の 3 行を、$\alpha_t=\alpha\sqrt{1-\beta_2^t}/(1-\beta_1^t)$ と $\theta_t\leftarrow\theta_{t-1}-\alpha_t m_t/(\sqrt{v_t}+\hat{\epsilon})$ の形に並べ替えると、明瞭さを少し犠牲にして効率化できると §2 は述べる。
- 必要メモリは、パラメータと同じ形の $m_t$ と $v_t$ を保持する分であり、Abstract は little memory requirements と書く。

## 実験・結果

- **データセット / ベンチマーク**:
  - MNIST images: L2-regularized multi-class logistic regression。入力は 784 dimension image vectors。minibatch size は 128。Figure 1 左。
  - IMDB movie reviews: 最頻 10,000 語の bag-of-words (BoW) feature vectors。10,000 次元 BoW は highly sparse。50% dropout noise を BoW features に適用する設定を含む。Figure 1 右。
  - MNIST images: two fully connected hidden layers、各 1000 hidden units、ReLU activation、minibatch size 128 の multi-layer neural network。dropout stochastic regularization ありと deterministic cost function の両方を扱う。Figure 2。
  - CIFAR-10: 5x5 convolution filters と 3x3 max pooling stride 2 の alternating stages を 3 つ持ち、その後に 1000 ReLU hidden units の fully connected layer を置く CNN。caption は c64-c64-c128-1000 architecture と書く。入力画像は whitening され、input layer と fully connected layer に dropout noise が適用される。minibatch size は 128。Figure 3。
  - Variational Auto-Encoder (VAE): Kingma & Welling (2013) と同じ architecture、single hidden layer with 500 hidden units、softplus nonlinearities、50-dimensional spherical Gaussian latent variable。Figure 4。
- **比較対象 / baseline**:
  - Logistic regression on MNIST: Adam、accelerated SGD with Nesterov momentum、Adagrad。
  - IMDB BoW logistic regression: Adagrad+dropout、RMSProp+dropout、SGDNesterov+dropout、Adam+dropout が Figure 1 右に示される。
  - MNIST multi-layer neural networks: dropout 設定では AdaGrad、RMSProp、SGDNesterov、AdaDelta、Adam。deterministic cost function では SFO との比較も含む。
  - CIFAR-10 CNN: AdaGrad、AdaGrad+dropout、SGDNesterov、SGDNesterov+dropout、Adam、Adam+dropout。
  - Bias correction 実験: bias-correction terms ありの Adam と、補正なしの RMSProp with momentum 相当を比較する。Figure 4 は赤線が bias correction、緑線が no bias correction。
- **指標**:
  - Figure 1 は logistic regression training negative log likelihood を示す。横軸は iterations over entire dataset。
  - Figure 2 と Figure 3 は training cost を示す。Figure 2 は MNIST multi-layer neural networks、Figure 3 は CIFAR-10 CNN の training cost。
  - Figure 4 は VAE 学習時の loss を示す。横軸は $\log_{10}(\alpha)$、条件は $\beta_1\in[0,0.9]$、$\beta_2\in[0.99,0.999,0.9999]$、$\log_{10}(\alpha)\in[-5,\ldots,-1]$。
- **主な結果**:
  - MNIST logistic regression では、著者は Adam が SGD with momentum と similar convergence を示し、両者が Adagrad より速く収束すると述べる。learning rate は理論と合わせて $\alpha_t=\alpha/\sqrt{t}$。
  - IMDB BoW では、Adagrad が SGD with Nesterov momentum を大きく上回り、Adam は Adagrad と同じくらい速く収束する。著者は、Adam が Adagrad と同様に sparse features を利用し、normal SGD with momentum より速い convergence rate を得ると解釈する。
  - MNIST multi-layer NN の deterministic cost function では、Adam は SFO より iterations と wall-clock time の両方で速い進捗を示す。§6.2 は、SFO は curvature information の更新コストにより Adam より 5-10x slower per iteration で、minibatch 数に線形なメモリ要求があると述べる。
  - Dropout 付き multi-layer NN では、SFO は deterministic subfunctions を仮定するため stochastic regularization 付き cost functions では failed to converge と書かれている。Figure 2 では Adam が他の stochastic first order methods より良い convergence を示すと著者は述べる。
  - CIFAR-10 CNN では、初期 3 epoch では Adam と Adagrad が速く cost を下げるが、45 epoch では Adam と SGD が Adagrad よりかなり速く収束する。著者は、$\hat{v}_t$ が数 epoch 後にほぼ 0 になり Algorithm 1 の $\epsilon$ に支配されるため、CNN では second moment estimate が cost function の geometry の近似として poor だと述べる。その上で、Adam は SGD with momentum に対して marginal improvement だが、SGD のように層ごとの learning rate scale を手で選ぶ代わりに自動適応するとする。
  - Bias correction 実験では、$\beta_2$ が 1 に近いときに補正なしでは training instabilities が生じ、特に最初の数 epoch で顕著だと述べられる。著者は、best results は小さい $(1-\beta_2)$ と bias correction で得られ、Adam は hyper-parameter setting に関わらず RMSProp と同等以上だったとまとめる。
- **著者が主張する貢献**:
  - 1 次・2 次モーメントの adaptive estimates に基づく、straightforward to implement、computationally efficient、little memory requirements な最適化手法を提案した。
  - diagonal rescaling of the gradients に不変で、large data / parameters、non-stationary objectives、very noisy and/or sparse gradients に適する手法だと主張した。
  - オンライン凸最適化で best known results と comparable な regret bound を与えた。
  - AdaGrad、RMSProp との関係を整理し、AdaMax と temporal averaging を拡張として議論した。

## 妥当性と限界

- **この主張を支える根拠**:
  - 手法の妥当性は、Algorithm 1 の単純な更新則、§2.1 の effective stepsize / SNR / gradient scale invariance の議論、§3 の bias correction 導出、§4 の online convex optimization における Theorem 4.1 と Corollary 4.2 によって支えられる。
  - 実験の妥当性は、convex な logistic regression、§6.2 が non-convex と明示する multi-layer NN、CNN、VAE まで複数のモデルを評価している点にある。§6 は同じ parameter initialization を使い、learning rate や momentum などの hyper-parameters を dense grid で探索し、best hyper-parameter setting の結果を報告すると述べる。
  - Sparse features については、IMDB BoW 10,000 次元 feature vectors と 50% dropout noise の設定、さらに Figure 1 の結果が AdaGrad との理論的近さを実験的に補強する。
  - Bias correction については、§3 の導出に加え、Figure 4 で $\beta_2\in\{0.99,0.999,0.9999\}$ と stepsize の広い範囲を振り、補正なしの不安定性を示している。
- **著者が認めている limitations / future work**:
  - §6.2 は、convergence analysis が non-convex problems には適用されないと明示する。その上で、多層 NN では empirical に Adam がしばしば他手法を上回ると述べる。
  - §6.3 は、CNN では $\hat{v}_t$ が数 epoch 後に 0 に近づき、$\epsilon$ が支配的になるため、second moment estimate が fully connected network に比べて poor approximation だと認める。また Adam の SGD with momentum に対する改善は marginal と書く。
  - AdaMax は §7.1 で導出と Algorithm 2 が示されるが、本文の実験節では AdaMax と Adam の定量比較は示されていない。
  - Temporal averaging は §7.2 で 1 行追加の方法として述べられるが、この論文中には temporal averaging を有効にした実験結果は示されていない。
- **読者として注意すべき点**:
  - Theorem 4.1 の Adam は、一般に実務で使う固定 $\beta_1$ の Adam そのままではなく、$\alpha_t=\alpha/\sqrt{t}$ と $\beta_{1,t}=\beta_1\lambda^{t-1}$ を仮定する。実験で常にこの減衰版を使っているわけではない。
  - Theorem 4.1 は $f_t$ の凸性、勾配有界性、パラメータ間距離の有界性、$\beta_1^2/\sqrt{\beta_2}<1$ などを仮定する。深層 NN の非凸実験は理論保証の直接の帰結ではない。
  - 実験結果は主に training cost / training negative log likelihood の学習曲線で示される。汎化性能や test accuracy の数値表は TeX 参照先の本文には明示されていない。
  - CNN の結果では、Adam の優位は大きいとは書かれていない。特に $\hat{v}_t$ が $\epsilon$ 支配になるという観察は、Adam の 2 次モーメント推定が常に有効な geometry 近似になるわけではないことを示す。
  - §6 は hyper-parameters を dense grid で探索し best setting を報告したと述べるため、Algorithm 1 の default hyper-parameters だけで全図が生成されたと読むべきではない。
- **追加で確認したい実験 / 疑問**:
  - AdaMax と Adam の同一条件比較。本文では AdaMax の導出と既定値 $\alpha=0.002,\ \beta_1=0.9,\ \beta_2=0.999$ は示されるが、実験比較はない。
  - Theorem 4.1 で必要な $\beta_{1,t}=\beta_1\lambda^{t-1}$ の momentum decay が、非凸実験でどの程度効くか。TeX 参照先にはこの ablation は明示されていない。
  - CNN で $\epsilon$ が支配的になるという §6.3 の観察に対し、$\epsilon$ の値や層別 learning rate の扱いを振った比較。TeX 参照先にはこの追加実験はない。
  - Temporal averaging を追加した場合の convergence や generalization。§7.2 は手順だけで、結果は示していない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **stochastic objective function**: $f(\theta)$ の noisy realization として $f_t(\theta)$ が逐次現れる目的関数である。ノイズ源は random subsamples / minibatches や dropout などの inherent function noise と説明される。
- **first-order gradients**: Adam が必要とする唯一の微分情報である。高次導関数や Hessian は使わない。
- **adaptive learning rates**: パラメータごとに、勾配の 1 次・2 次モーメント推定から作られる学習率調整である。
- **first moment estimate $m_t$**: 勾配 $g_t$ の指数移動平均。Algorithm 1 では biased first moment estimate と呼ばれる。
- **second raw moment estimate $v_t$**: 二乗勾配 $g_t^2$ の指数移動平均。中心化分散ではなく raw / uncentered な 2 次モーメントである。
- **bias-corrected estimates $\hat{m}_t,\hat{v}_t$**: 0 初期化による初期バイアスを $1-\beta_1^t$、$1-\beta_2^t$ で割って補正した推定量である。
- **$\epsilon$**: 分母 $\sqrt{\hat{v}_t}+\epsilon$ の数値安定化項。Algorithm 1 の既定値は $10^{-8}$。CNN 実験では $\hat{v}_t$ が 0 近くになり、この $\epsilon$ に支配されると観察されている。
- **SNR**: §2.1 で $\hat{m}_t/\sqrt{\hat{v}_t}$ に付けられた呼び名。true gradient 方向への確信が弱い場合に小さくなり、effective stepsize を縮めると説明される。
- **automatic annealing**: SNR が optimum に近づくにつれて 0 に近づき、有効ステップが小さくなる性質を著者がそう説明している。
- **diagonal rescaling of the gradients**: 各次元の勾配スケールを対角的に変えても、$\hat{m}_t$ と $\sqrt{\hat{v}_t}$ の倍率が打ち消し合い、更新が不変になるという性質である。
- **AdaGrad**: Sparse gradients に強い関連手法。§5 では Adam の $\beta_1=0$、$(1-\beta_2)\to0$、$\alpha_t=\alpha t^{-1/2}$ の極限として対応づけられる。
- **RMSProp**: Adam と直接関係する手法。Adam との違いは、bias correction の欠如と、momentum のかけ方であると §5 は説明する。
- **SFO**: Sum-of-Functions Optimizer。minibatch に基づく quasi-Newton method で、§6.2 では Adam より 5-10x slower per iteration かつメモリ要求が minibatch 数に線形とされる。
- **regret $R(T)$**: オンライン凸最適化で、逐次選択 $\theta_t$ と最良固定点 $\theta^*$ の累積損失差を測る量である。
- **AdaMax**: Adam の $L^p$ norm 一般化で $p\to\infty$ を取った派生手法。$u_t=\max(\beta_2u_{t-1},|g_t|)$ を使い、second moment の bias correction が不要になる。
- **temporal averaging**: stochastic approximation による last iterate のノイズを抑えるため、パラメータ $\theta_t$ の平均を取る拡張である。§7.2 では $\bar{\theta}_t\leftarrow\beta_2\bar{\theta}_{t-1}+(1-\beta_2)\theta_t$ と書かれる。

## 読む順番の提案

- まず Abstract と §1 を読み、Adam が「first-order gradient-based optimization of stochastic objective functions」に対する adaptive moment estimation として位置づけられていることを確認する。正規ノートの Summary 冒頭と対応する。
- 次に Algorithm 1 と §2 を読む。$m_t$、$v_t$、$\hat{m}_t$、$\hat{v}_t$、$\theta_t$ の更新を、正規ノートの「手法」箇条書きと照合するとよい。
- その後 §2.1 を読む。$|\Delta_t|\lessapprox\alpha$、trust region、SNR、gradient scale invariance が、Adam の直感的説明の中心である。正規ノートの Takeaway の「実効ステップ幅」と対応する。
- §3 では Eq. (1) から Eq. (4) を追い、なぜ $1-\beta_2^t$ で割るのかを確認する。Figure 4 と正規ノートの bias correction の記述につながる。
- 理論を読む場合は §4 の regret 定義 Eq. (5)、Theorem 4.1、Corollary 4.2 を先に読み、詳細証明は Appendix §10.1 を後回しにする。正規ノートの Theorem 4.1 仮定リストと対応する。
- 関連研究の位置づけは §5 を読む。RMSProp との差分と AdaGrad との極限対応が、正規ノートの Related Papers と Takeaway の土台である。
- 実験は Figure 1 から Figure 4 と §6.1 から §6.4 を対応させて読む。特に CNN の §6.3 は、Adam が常に大きく優位とは限らないことを示す注意点として重要である。
- 最後に §7.1 AdaMax と §7.2 temporal averaging を読む。正規ノートの AdaMax と temporal averaging の記述につながるが、本文実験には AdaMax / temporal averaging の結果がない点を確認する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1412.6980v9/`
- 正規ノート: `notes/arXiv-1412.6980v9.md`
