# Auto-Encoding Variational Bayes（連続潜在変数をもつ有向確率モデルの変分推論を SGD で学習可能にする論文）

- arXiv: https://arxiv.org/abs/1312.6114
- 一次ソース: ../papers/arXiv-1312.6114v11/
- 正規ノート: ../notes/arXiv-1312.6114v11.md

---

## 一言で言うと

この論文は、連続潜在変数をもつ有向確率モデルで、周辺尤度や真の事後分布が解析不能な場合でも、変分下界を微分可能な Monte Carlo 推定量に変換して SGD/Adagrad で学習する SGVB estimator と AEVB algorithm を提案する。i.i.d. データセットの各データ点にある連続潜在変数について、共有された recognition model $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ を学習することで、データ点ごとの MCMC や反復推論なしに近似事後推論を行う、というのが著者の主張である。

## 何を議論する論文か

- **問題設定**: データ $\mathbf{X}=\{\mathbf{x}^{(i)}\}_{i=1}^N$ が i.i.d. に与えられ、未観測の連続潜在変数 $\mathbf{z}$ を経て $\mathbf{x}$ が生成されると仮定する。有向生成モデルは $p_{\boldsymbol{\theta}}(\mathbf{z})p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ で、近似事後分布として $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ を導入する。
- **対象範囲 / 仮定**: $\mathbf{x}$ は連続または離散変数でよいが、潜在変数 $\mathbf{z}$ は連続である。本文の基本設定では、グローバルパラメータ $\boldsymbol{\theta}$ は ML または MAP、潜在変数は変分推論で扱う。$p_{\boldsymbol{\theta}}(\mathbf{z})$ と $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ の PDF は $\boldsymbol{\theta}$ と $\mathbf{z}$ に関して almost everywhere differentiable と仮定される（`Problem scenario`）。
- **既存研究との差分**: 通常の mean-field VB は近似事後に関する期待値を解析的に解く必要があり、一般には難しい。naive Monte Carlo gradient estimator は高分散で実用上問題がある。wake-sleep は同じように recognition model を使うが、2 つの目的関数を同時に最適化し、それらは周辺尤度またはその下界の最適化に対応しない、と本文は述べる。
- **この論文で答えたい問い**: intractable posterior、intractable marginal likelihood、大規模データセットという条件下で、近似 ML/MAP 推定、潜在変数の近似事後推論、周辺推論を効率的に行えるか。

## 背景と前提

- 変分推論では、真の事後分布 $p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x})$ の代わりに扱いやすい $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ を最適化する。この論文では、$q_{\boldsymbol{\phi}}$ は mean-field VB のように閉形式の期待値から更新されるものではなく、生成モデルのパラメータ $\boldsymbol{\theta}$ と同時に学習される recognition model である。
- 論文中の encoder / decoder は確率的な意味で使われる。$q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ は、データ点 $\mathbf{x}$ から code $\mathbf{z}$ の分布を出す probabilistic encoder。$p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ は、code $\mathbf{z}$ から $\mathbf{x}$ の分布を出す probabilistic decoder である。
- SGVB estimator は、$q_{\boldsymbol{\phi}}$ からのサンプルを $\boldsymbol{\phi}$ に依存する確率ノードとして扱うのではなく、$\boldsymbol{\phi}$ に依存しない補助ノイズ $\boldsymbol{\epsilon}$ と微分可能な変換 $g_{\boldsymbol{\phi}}$ に書き換える。これにより Monte Carlo 推定量を $\boldsymbol{\phi}$ について微分できる。
- 主な比較対象は wake-sleep algorithm と Monte Carlo EM (MCEM) である。関連研究として、Stochastic Variational Inference、Blei et al. 2012 の stochastic search 系、Ranganath et al. 2013 の Black Box Variational Inference、Salimans & Knowles 2013、DARN、Rezende et al. 2014 などが `iclr14_sva.bbl` に挙がる。

## 提案手法

### コアアイデア

著者は、周辺尤度 $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ を直接評価・微分できない代わりに、変分下界 $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x})$ を最大化する。問題は、下界に含まれる $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ に関する期待値を、低分散かつ $\boldsymbol{\phi}$ について微分可能に推定することである。

そのために、サンプル $\tilde{\mathbf{z}}\sim q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ を
$\tilde{\mathbf{z}}=g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon},\mathbf{x})$、$\boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon})$
と再パラメータ化する。Gaussian の例では $\mathbf{z}=\boldsymbol{\mu}+\boldsymbol{\sigma}\odot\boldsymbol{\epsilon}$、$\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ である。

i.i.d. データセットのケースでは、この推定量を使って recognition model と generative model を同時最適化する AEVB algorithm を構成する。recognition model にニューラルネットワークを使う場合、著者はこれを variational auto-encoder と呼ぶ。

### 重要な定義・数式

$$
\begin{aligned}
\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})
&= D_{KL}\!\left(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}^{(i)}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x}^{(i)})\right)
+ \mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)}) \\
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})
&= -D_{KL}\!\left(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}^{(i)}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z})\right)
+ \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}^{(i)})}
\left[\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}\mid\mathbf{z})\right]
\end{aligned}
$$

**式の意味**: 周辺対数尤度を、真の事後と近似事後の KL divergence と、変分下界に分解している。KL は非負なので、$\mathcal{L}$ は datapoint $i$ の marginal likelihood の lower bound になる（`eq:lowerbound`, `eq:lowerbound2`）。

**記号の定義**:
- $\mathbf{x}^{(i)}$ ... $i$ 番目の観測データ
- $\mathbf{z}$ ... 未観測の連続潜在変数
- $p_{\boldsymbol{\theta}}(\mathbf{z})$ ... 生成モデルの prior
- $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ ... 生成モデルの likelihood / decoder
- $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ ... 真の事後分布を近似する recognition model / encoder
- $D_{KL}(\cdot\|\cdot)$ ... 2 つの分布の KL divergence

**この論文での役割**: AEVB はこの下界を最大化する。第 1 項は近似事後を prior に近づける正則化項として解釈され、第 2 項は decoder の expected reconstruction term として解釈される。

$$
\tilde{\mathbf{z}} = g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon},\mathbf{x}),\quad
\boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon}),\quad
\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}^{(i)})}[f(\mathbf{z})]
= \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon},\mathbf{x}^{(i)}))]
\simeq \frac{1}{L}\sum_{l=1}^L f(g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon}^{(l)},\mathbf{x}^{(i)}))
$$

**式の意味**: $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ から直接サンプリングする代わりに、$\boldsymbol{\phi}$ に依存しないノイズ $\boldsymbol{\epsilon}$ から微分可能な変換で $\mathbf{z}$ を作る。本文ではこれを reparameterization trick として導入する。

**記号の定義**:
- $\tilde{\mathbf{z}}$ ... $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ からのサンプルとして扱う変数
- $g_{\boldsymbol{\phi}}$ ... ノイズとデータから潜在変数サンプルを作る微分可能な変換
- $\boldsymbol{\epsilon}$ ... 補助ノイズ変数。Gaussian 例では $\mathcal{N}(\mathbf{0},\mathbf{I})$
- $f(\mathbf{z})$ ... 期待値を取りたい任意の関数
- $L$ ... datapoint あたりの Monte Carlo サンプル数

**この論文での役割**: 下界の Monte Carlo 推定量を $\boldsymbol{\phi}$ について微分可能にする中心技術である。本文は inverse CDF、location-scale family、composition を、適用可能な分布クラスとして列挙している。

$$
\widetilde{\mathcal{L}}^{B}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})
= -D_{KL}\!\left(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}^{(i)}) \,\|\, p_{\boldsymbol{\theta}}(\mathbf{z})\right)
+ \frac{1}{L}\sum_{l=1}^L \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}\mid\mathbf{z}^{(i,l)}),
\quad
\mathbf{z}^{(i,l)} = g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon}^{(i,l)},\mathbf{x}^{(i)})
$$

**式の意味**: SGVB estimator B である。KL 項が解析積分できる場合、サンプリングで推定するのは reconstruction 側だけになるため、generic estimator A より typically less variance と説明される（`eq:estimator2`）。

**記号の定義**:
- $\widetilde{\mathcal{L}}^{B}$ ... estimator B による下界の Monte Carlo 推定量
- $\mathbf{z}^{(i,l)}$ ... datapoint $i$ に対する $l$ 番目の潜在サンプル
- $\boldsymbol{\epsilon}^{(i,l)}$ ... そのサンプルを作る補助ノイズ
- $\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}\mid\mathbf{z}^{(i,l)})$ ... decoder がデータ点 $\mathbf{x}^{(i)}$ を説明する対数尤度

**この論文での役割**: 実験で使う Gaussian prior / Gaussian approximate posterior のケースでは、KL 項を閉形式で計算できるため、この estimator B が実用的な目的関数になる。

$$
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{X})
\simeq
\widetilde{\mathcal{L}}^{M}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{X}^{M})
= \frac{N}{M}\sum_{i=1}^{M}\widetilde{\mathcal{L}}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})
$$

**式の意味**: full dataset の lower bound を、ランダムに選んだ minibatch $\mathbf{X}^M$ で推定する式である（`eq:minibatchestimator`）。

**記号の定義**:
- $\mathbf{X}$ ... $N$ 個のデータ点からなる全データセット
- $\mathbf{X}^{M}$ ... $M$ 個の datapoint からなるランダム minibatch
- $N$ ... 全データ数
- $M$ ... minibatch size
- $\widetilde{\mathcal{L}}$ ... estimator A または B による datapoint ごとの下界推定量

**この論文での役割**: 大規模データセットで online / minibatch stochastic optimization を可能にする部分である。実験では $M=100$、$L=1$ が使われる。

$$
\begin{aligned}
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x}^{(i)})
&\simeq
\frac{1}{2}\sum_{j=1}^{J}
\left(1+\log((\sigma_j^{(i)})^2)-(\mu_j^{(i)})^2-(\sigma_j^{(i)})^2\right)
+ \frac{1}{L}\sum_{l=1}^{L}\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}\mid\mathbf{z}^{(i,l)}) \\
\mathbf{z}^{(i,l)}
&= \boldsymbol{\mu}^{(i)}+\boldsymbol{\sigma}^{(i)}\odot\boldsymbol{\epsilon}^{(l)},\quad
\boldsymbol{\epsilon}^{(l)}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
\end{aligned}
$$

**式の意味**: Variational Auto-Encoder 例で、prior $p_{\boldsymbol{\theta}}(\mathbf{z})=\mathcal{N}(\mathbf{0},\mathbf{I})$、近似事後 $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}^{(i)})=\mathcal{N}(\boldsymbol{\mu}^{(i)},(\boldsymbol{\sigma}^{(i)})^2\mathbf{I})$ のときの下界推定量である（`eq:gaussian_estimator`）。

**記号の定義**:
- $J$ ... 潜在変数 $\mathbf{z}$ の次元数
- $\mu_j^{(i)}$ ... encoder MLP が datapoint $i$ に対して出す平均ベクトルの $j$ 番目の要素
- $\sigma_j^{(i)}$ ... encoder MLP が datapoint $i$ に対して出す標準偏差ベクトルの $j$ 番目の要素
- $\odot$ ... 要素ごとの積
- $\mathcal{N}(\mathbf{0},\mathbf{I})$ ... centered isotropic multivariate Gaussian

**この論文での役割**: VAE として最も具体的に実装される目的関数である。appendix の Gaussian KL 解により、KL 側はサンプリングせず閉形式で計算できる。

### 実装 / アルゴリズム上の要点

- step1: $\boldsymbol{\theta},\boldsymbol{\phi}$ を初期化する。
- step2: full dataset から minibatch $\mathbf{X}^{M}$ をランダムに引く。
- step3: ノイズ分布 $p(\boldsymbol{\epsilon})$ からランダムサンプルを引く。
- step4: minibatch estimator $\widetilde{\mathcal{L}}^{M}$ の勾配 $\nabla_{\boldsymbol{\theta},\boldsymbol{\phi}}\widetilde{\mathcal{L}}^{M}$ を計算する。
- step5: SGD または Adagrad で $\boldsymbol{\theta},\boldsymbol{\phi}$ を更新し、収束まで繰り返す。
- 実験では minibatch size $M=100$、datapoint あたりのサンプル数 $L=1$。パラメータは $\mathcal{N}(0,0.01)$ から初期化し、Adagrad の global stepsize は $\{0.01,0.02,0.1\}$ から training set の初期 performance に基づいて選ぶ。小さな weight decay は prior $p(\boldsymbol{\theta})=\mathcal{N}(0,\mathbf{I})$ に対応すると本文にある。
- appendix の MLP 定義では、Bernoulli decoder は single hidden layer の fully-connected network で確率 $\mathbf{y}=f_\sigma(\mathbf{W}_2\tanh(\mathbf{W}_1\mathbf{z}+\mathbf{b}_1)+\mathbf{b}_2)$ を出す。Gaussian encoder/decoder は diagonal covariance Gaussian の $\boldsymbol{\mu}$ と $\log\boldsymbol{\sigma}^2$ を hidden layer $\mathbf{h}=\tanh(\mathbf{W}_3\mathbf{z}+\mathbf{b}_3)$ から出し、encoder として使う場合は $\mathbf{z}$ と $\mathbf{x}$ を入れ替える。

## 実験・結果

- **データセット / ベンチマーク**: MNIST と Frey Face。Frey Face は continuous data として扱われ、decoder は Gaussian outputs を使い、means は decoder output の sigmoidal activation で $(0,1)$ に制約される。
- **比較対象 / baseline**: wake-sleep algorithm。estimated marginal likelihood の比較では Monte Carlo EM (MCEM) with Hybrid Monte Carlo (HMC) sampler も使う。wake-sleep には AEVB と同じ encoder / recognition model を使う。
- **指標**: Figure 2 は estimated average variational lower bound per datapoint。Figure 3 は estimated marginal likelihood。Figure 3 の PDF には $N_{\text{train}}=1000$ と $N_{\text{train}}=50000$ のパネルがある。
- **主な結果**: Figure 2 の caption では、AEVB は wake-sleep より considerably faster に収束し、all experiments で better solution に到達したと著者が述べる。Figure 2 のパネル題名では、MNIST は $N_{\mathbf{z}}=\{3,5,10,20,200\}$、Frey Face は $N_{\mathbf{z}}=\{2,5,10,20\}$ が確認できる。caption は estimator variance が小さい、具体的には $<1$ だったため省略したとも述べる。
- **主な結果**: likelihood lower bound の実験では、MNIST に 500 hidden units、Frey Face に 200 hidden units を使う。Frey Face の hidden units が少ない理由は、データセットが considerably smaller で overfitting を防ぐためと説明される。著者は、superfluous latent variables did not result in overfitting と述べ、variational bound の regularizing nature で説明している。
- **主な結果**: marginal likelihood の比較では、encoder/decoder は 100 hidden units、latent variables は 3。本文は、より高次元の latent space では estimates became unreliable と述べる。appendix の estimator は sampled space が low、具体的には less than 5 dimensions で十分なサンプルがある場合に good estimates を出す、と制限を明記している。
- **主な結果**: visualization では、2D latent space の generative model について、Gaussian prior の inverse CDF で unit square 上の座標を $\mathbf{z}$ に変換し、それぞれの $\mathbf{z}$ に対する $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ を描画する。Figure 5 は MNIST の learned generative models からの random samples を、2-D、5-D、10-D、20-D latent space で示す。
- **著者が主張する貢献**: reparameterization によって variational lower bound の simple differentiable unbiased estimator を得ること、i.i.d. データセットの continuous latent variables per datapoint に対して recognition model を学習して posterior inference を効率化すること、そして標準的な stochastic gradient methods で最適化できること。

## 妥当性と限界

- **この主張を支える根拠**: 理論面では、KL 分解により lower bound を定義し、reparameterization により expectation の Monte Carlo 推定量を $\boldsymbol{\phi}$ について微分可能にする。実験面では、MNIST と Frey Face で lower bound を比較し、MNIST では estimated marginal likelihood も比較して、wake-sleep / MCEM との差を図で示す。
- **著者が認めている limitations / future work**: Full VB、つまり global parameters $\boldsymbol{\theta}$ への変分推論は appendix に導出だけあり、experiments are left to future work と本文にある。marginal likelihood estimator は sampled space が less than 5 dimensions 程度でないと信頼しにくく、本文でも高次元 latent space で estimates became unreliable と述べる。Future work は、hierarchical / convolutional encoder-decoder、time-series models、global parameters への SGVB、latent variables をもつ supervised models である。
- **読者として注意すべき点**: 論文の主要実験は MNIST と Frey Face の 2 データセットであり、encoder/decoder も比較的 simple な single hidden layer MLP である。Gaussian diagonal covariance の近似事後は footnote で simplifying choice であって method の limitation ではないとされるが、この TeX で示される実験はその設定に基づく。
- **読者として注意すべき点**: Figure 2 の高次元 latent space の比較は variational lower bound の比較であり、同じ高次元設定での marginal likelihood 推定は提示されていない。Figure 3 の marginal likelihood は 3 latent variables に限定される。
- **読者として注意すべき点**: wake-sleep は discrete latent variables にも適用できる点を本文が advantage として挙げる。一方、この論文の reparameterization trick と AEVB の中心設定は continuous latent variables である。
- **追加で確認したい実験 / 疑問**: Estimator A と Estimator B の分散差を同じモデルで定量比較する実験、潜在次元を増やしたときの有効次元や KL 項の内訳、deep / convolutional encoder-decoder で同じ比較をした場合の収束、より信頼できる高次元 marginal likelihood estimator による評価は、TeX 中には明示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **SGVB (Stochastic Gradient Variational Bayes)**: reparameterization した variational lower bound の Monte Carlo 推定量。標準的な stochastic gradient ascent で最適化できることが重視される。
- **AEVB (Auto-Encoding VB)**: i.i.d. データセットと datapoint ごとの continuous latent variables の設定で、SGVB estimator を使って recognition model と generative model を同時学習するアルゴリズム。
- **Variational Auto-Encoder**: recognition model に neural network を使う AEVB の具体例。本文では encoder $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ と decoder $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ の確率モデルとして説明される。
- **recognition model / probabilistic encoder**: $\mathbf{x}$ を入力として、可能な code $\mathbf{z}$ の分布を返す $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$。真の事後 $p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x})$ の近似である。
- **probabilistic decoder**: code $\mathbf{z}$ から観測 $\mathbf{x}$ の分布を返す $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$。Bernoulli または Gaussian MLP として実装される。
- **variational lower bound / lower bound / ELBO**: 周辺対数尤度の下界。本文の記号では $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x})$。
- **reconstruction error**: estimator B の $\log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}\mid\mathbf{z}^{(i,l)})$ に対応する項を、auto-encoder の言葉で negative reconstruction error と説明している。
- **reparameterization trick**: $\mathbf{z}\sim q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ を、$\mathbf{z}=g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon},\mathbf{x})$、$\boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon})$ に書き換える方法。
- **Estimator A / Estimator B**: Estimator A は $\log p_{\boldsymbol{\theta}}(\mathbf{x},\mathbf{z})-\log q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ をサンプル平均する generic estimator。Estimator B は KL を解析的に計算し、reconstruction 項だけをサンプル平均する低分散版。
- **MCEM / HMC**: marginal likelihood 比較の baseline。appendix では MCEM は encoder を使わず、posterior gradient で HMC サンプリングする。10 HMC leapfrog steps、acceptance rate 90%、その後 5 weight update steps と書かれている。

## 読む順番の提案

- まず `iclr14_sva.tex` の Abstract と Introduction を読み、問いが "directed probabilistic models", "continuous latent variables", "intractable posterior distributions", "large datasets" の組み合わせであることを押さえる。正規ノートでは `Summary（著者の主張）` の問題設定に対応する。
- 次に `Method` の `Problem scenario` と Figure 1 caption を読む。ここで $p_{\boldsymbol{\theta}}(\mathbf{z})p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ と $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ の役割を確認する。
- その後、`The variational bound` の `eq:lowerbound`, `eq:lowerbound2` を読む。正規ノートの Takeaway にある「KL 正則化 + reconstruction」の理解につながる。
- 次に `The SGVB estimator and AEVB algorithm` の `eq:fullestimator`, `eq:estimator2`, `eq:minibatchestimator` と Algorithm 1 を読む。ここが AEVB の学習手順そのもの。
- `The reparameterization trick` は Gaussian の例だけでなく、inverse CDF、location-scale、composition の 3 類型を確認する。正規ノートの reparameterization trick の説明に対応する。
- `Example: Variational Auto-Encoder` と appendix の `Solution of -D_KL`, `MLP's as probabilistic encoders and decoders` を読む。`eq:gaussian_estimator` が実装に最も近い式である。
- 実験は `Experiments`、Figure 2、Figure 3、appendix の `Marginal likelihood estimator` と `Monte Carlo EM` を合わせて読む。正規ノートの `Critical Thoughts` の、評価設計と限界の議論につながる。
- 最後に `Related work`, `Conclusion`, `Future work`, `iclr14_sva.bbl` を読み、wake-sleep、SVI、DARN、Rezende et al. 2014 との位置づけを確認する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1312.6114v11/`
- 正規ノート: `notes/arXiv-1312.6114v11.md`
