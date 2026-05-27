# Auto-Encoding Variational Bayes

- arXiv: https://arxiv.org/abs/1312.6114
- source: ../papers/arXiv-1312.6114v11/
- authors: Diederik P. Kingma, Max Welling (Universiteit van Amsterdam)
- venue / year: TeX 中には明示なし（本文は `nips13submit_e` style を使用）
- tags: [variational-inference, generative-model, VAE, reparameterization, deep-learning]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 連続潜在変数 $\bz$ を持つ有向確率モデル $\pT(\bz)\pT(\bx|\bz)$ で、(i) 周辺尤度 $\pT(\bx)=\int \pT(\bz)\pT(\bx|\bz)d\bz$ が解析不可能、(ii) 真の事後 $\pT(\bz|\bx)$ も解析不可能、(iii) 平均場 VB の期待値も解析的に解けない、という三重 intractable な状況を、大規模 i.i.d. データセットでも online で効率的に学習・推論したい。従来の Monte Carlo gradient（score-function 推定）は分散が大きすぎて使い物にならず（cf. \cite{blei2012variational}）、wake-sleep は2つの目的関数を別々に最適化するため周辺尤度の下界に対応しない。
- **手法**: 2つの貢献を組み合わせる。(1) **SGVB 推定量**: 変分下界 $\LB{}{\bxi} = -D_{KL}(\qPhi(\bz|\bxi)||\pT(\bz)) + \Exp{\qPhi}{\log \pT(\bxi|\bz)}$ を、**reparameterization trick** $\bz=\gPhi(\beps,\bx),\,\beps\sim p(\beps)$（例: Gaussian なら $\bz=\bmu+\bsigma\odot\beps,\,\beps\sim\mathcal{N}(\bzero,\bI)$）で書き換え、$\bphi$ について微分可能な不偏推定量に変換する。Estimator A（生形）と Estimator B（Gaussian–Gaussian で KL を解析積分した低分散版）を提示。(2) **AEVB アルゴリズム**: 認識モデル（probabilistic encoder）$\qPhi(\bz|\bx)$ を MLP でパラメタライズし、生成モデル $\pT(\bx|\bz)$（Bernoulli or Gaussian MLP decoder）と $(\bT,\bphi)$ を SGD/Adagrad で**同時最適化**。これにより MCMC や per-datapoint の反復推論なしで ancestral sampling による approximate posterior inference を行う。Encoder MLP を使う場合が **Variational Auto-Encoder**。
- **結果**: MNIST と Frey Face で実験。Encoder/decoder の隠れユニット数は MNIST 500、Frey Face 200。$M=100,\,L=1$。
  - **下界の比較 (Fig.2, `graph_lowbound`)**: MNIST は $N_\bz \in \{3,5,10,20,200\}$、Frey Face は $N_\bz \in \{2,5,10,20\}$ の各設定で、AEVB は wake-sleep より速く収束し最終下界も高い。次元を増やしても overfitting しない（下界の正則化効果）。
  - **周辺尤度の比較 (Fig.3, `graph_marglik`)**: 100 hidden units, 3 latent variables で、wake-sleep / MCEM (HMC sampler) と convergence speed を比較。MCEM は online algorithm ではなく、AEVB/wake-sleep と異なり full MNIST dataset には効率的に適用できない、とキャプションで述べる。
  - **2D 潜在多様体の可視化 (Fig.4, appendix A)**: MNIST/Frey Face とも、prior $\mathcal{N}(\bzero,\bI)$ の逆 CDF で作った $\bz$ の値に対する生成分布 $\pT(\bx|\bz)$ を可視化。
  - **サンプル品質 (Fig.5, appendix A)**: 潜在次元 2/5/10/20 で MNIST の random sample を提示。
- **貢献**: (1) reparameterization により変分下界の低分散・微分可能な MC 推定量 (SGVB) を提示、(2) 認識モデルを使って per-datapoint の expensive iterative inference schemes なしに approximate posterior inference を行う AEVB を提示、(3) これらを連続潜在変数を持つ広いクラスの有向モデル（mean-field でも intractable な場合を含む）に適用可能にし、auto-encoder と有向確率モデルを理論的に橋渡しした。

## Takeaway（自分にとっての要点）

- **reparameterization trick の本質**は「$\bphi$ に依存する分布からのサンプル」を「$\bphi$ に依存しない noise からの決定論的変換」に書き換えること。score-function 推定の分散爆発を回避できる根拠は「$f$ に $\nabla\log q$ を掛ける」のではなく「$f\circ g$ をそのまま微分する」点にある。
- Gaussian–Gaussian の場合、KL 項は閉形式 $\frac{1}{2}\sum_j(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)$ なので、サンプリングが必要なのは reconstruction $\log\pT(\bx|\bz)$ 側だけ。
- recognition model $\qPhi(\bz|\bx)$ を導入する旨味は「per-datapoint の MCMC や iterative inference を避け、学習済みの approximate posterior inference model を使える」こと。
- 「unregularized autoencoder = mutual information の下界最大化」「VAE = それ + KL 正則化」という対応関係が、関連研究の議論として明示されている。SGVB の KL 項は denoising/contractive autoencoder の「nuisance な正則化ハイパラ」を不要にする、と主張。
- 実装上の細部: $L=1$ で十分（minibatch $M=100$ で平均化されるため）、Adagrad の global stepsize は $\{0.01,0.02,0.1\}$ から train 数イタで選択、初期化は $\mathcal{N}(0,0.01)$、prior $p(\bT)=\mathcal{N}(0,\bI)$ に対応する弱い weight decay。
- 周辺尤度の estimator（appendix）は、posterior から HMC でサンプルし、そのサンプルに density estimator $q(\bz)$ をフィットし、別の posterior sample と $q(\bz)$ を式に代入する。**5 次元未満の sampled space でないと reliable ではない** と著者自身が述べている。Fig.3 の marginal likelihood 比較が 3 latent variables に限られているのはこの制約のため。
- Rezende+ 2014 が独立に、auto-encoder / directed probabilistic model / stochastic variational inference の接続を同じ reparameterization trick で示した、と Related work で明記。

## Critical Thoughts（評価・疑問）

- **強み**:
  - reparameterization trick により「変分下界の微分」「recognition model による approximate posterior inference」「auto-encoder との接続」を同じ枠組みで扱っている。
  - wake-sleep（2 目的関数）/ MCEM（per-datapoint sampling）に対して、**1 つの下界を 1 つの SGD で最適化**する設計。Fig.2 のキャプションでは、AEVB が全実験で wake-sleep より速く収束し良い解に到達したと述べる。
  - 潜在次元を増やしても overfitting しないという観察を、著者は variational bound の regularizing nature で説明している。
  - reparameterization が適用可能な分布クラス（inverse CDF / location-scale / composition）を3類型で列挙しており、Gaussian 以外への拡張余地を明示している。
- **弱み / 疑問**:
  - 実験が MNIST + Frey Face の2データセットのみ。画像複雑度・モダリティの広がりは TeX 中には示されていない。
  - 周辺尤度評価は 3 latent variables の設定で実施されており、著者は高次元 latent space では推定が unreliable になったと述べている。Fig.2 の高次元（MNIST 200 dim）では marginal likelihood ではなく下界の比較のみが提示されている。
  - 離散潜在変数への適用は TeX 中には提示されていない。Related work では、wake-sleep の利点として discrete latent variables にも適用できる点を挙げている。
  - Encoder/decoder は実験では single hidden layer の MLP。posterior が diagonal Gaussian 近似から外れる場合の実験は TeX 中には示されていない。著者は脚注で「これは simplifying choice であり method の limitation ではない」と述べている。
  - Wake-sleep 比較では同じ encoder を使う一方、MCEM の HMC 設定（leapfrog 10 step, acceptance 90%）の感度分析は TeX 中には示されていない。
  - 「KL が正則化として機能する」ことの定量的検証（KL の値、有効次元数の測定）はなく、観察レベル。
- **次に試したいこと**:
  - SGVB Estimator A（KL を解析せず両項とも MC）と Estimator B（解析 KL）の分散を、同じ MNIST 設定で実測してどれだけ差があるか（評者補足）。
  - Bernoulli decoder の入力を normalize MNIST にしたときと、Gaussian decoder にしたときの test log-likelihood の差（評者補足）。
  - 潜在次元を 200 まで増やしても下界が下がらない件について、「使われていない次元」を $\bsigma_j\approx 1, \bmu_j\approx 0$ で同定し、有効次元と test marginal likelihood の関係を可視化（評者補足）。
  - 同じ AEVB 骨格で encoder を deep / convolutional に置き換えたとき、Fig.2 の収束カーブがどう変わるか（著者 Future work の (i)）。
  - reparameterization が成立しないケース（Bernoulli latent）で score-function + control variate と比較し、どこまで variance を詰められるか（評者補足）。

## Notes / Quotes

- "We show how a reparameterization of the variational lower bound yields a simple differentiable unbiased estimator of the lower bound." (introduction)
- "we make inference and learning especially efficient by using the SGVB estimator to optimize a recognition model that allows us to perform very efficient approximate posterior inference using simple ancestral sampling." (introduction)
- Estimator B（解析 KL 版）: $\LBT{B}{\bxi} = -D_{KL}(\qPhi(\bz|\bxi)||\pT(\bz)) + \frac{1}{L}\sum_l \log \pT(\bxi|\bzil)$ (eq. estimator2)
- Gaussian–Gaussian の閉形式 KL（appendix B）: $-D_{KL} = \frac{1}{2}\sum_{j=1}^J(1+\log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$
- 実装デフォルト: $M=100,\,L=1$、Adagrad stepsize $\in\{0.01,0.02,0.1\}$、init $\mathcal{N}(0,0.01)$（experiments）
- "Interestingly, superfluous latent variables did not result in overfitting, which is explained by the regularizing nature of the variational bound." (experiments)
- 周辺尤度 estimator の制限: "produces good estimates ... as long as the dimensionality of the sampled space is low (less then 5 dimensions)"（appendix D 著者自身による limitation 認識）
- MCEM 設定: HMC 10 leapfrog steps, acceptance 90%, 続いて 5 weight update（appendix E）。周辺尤度は train/test 各 1000 datapoint × 50 サンプル × 4 leapfrog。
- (verified 2026-05-20) Fig 番号を TeX のレイアウト順 (Fig.1=graphical model, Fig.2=lowbound, Fig.3=marglik, Fig.4=2D manifolds, Fig.5=mnist samples) に合わせて全 Summary/Critical Thoughts でリナンバー (iclr14_sva.tex の各 figure 環境、iclr14_sva_appendix.tex)。
- (verified 2026-05-20) Fig.2 (lowbound) の潜在次元集合を MNIST $\{3,5,10,20,200\}$ と Frey Face $\{2,5,10,20\}$ に修正（figs/graph_lowbound.pdf のサブプロットタイトルから直接確認）。
- (verified 2026-05-20) Related Papers の Blei+ 2012 を "Black-Box VI" 表記から bbl 通り "Variational Bayesian inference with Stochastic Search" に修正、Ranganath+ 2013 を Black Box VI として明示 (main.bbl)。
- (verified 2026-05-20) Rezende+ 2014 のタイトルを bbl 記載 "Stochastic back-propagation and variational inference in deep latent gaussian models" (arXiv:1401.4082) に揃えた。
- (verified 2026-05-26) venue/year の ICLR 2014 表記を削除し、TeX 中に明示がない範囲へ修正。Fig.3 の結果解釈、後続研究名、現代実装への一般化など TeX 根拠外の記述を削除または「評者補足」と明示 (iclr14_sva.tex, iclr14_sva_appendix.tex, iclr14_sva.bbl)。
- (verified 2026-05-27) 「MLP の forward 1 回で posterior」という TeX に明示されない実装寄り表現を削除し、recognition model による approximate posterior inference の記述へ修正 (iclr14_sva.tex introduction/method)。
- 並行発見への言及: "Even more recently, ~\cite{rezende2014stochastic} also make the connection between auto-encoders, directed probabilistic models and stochastic variational inference using the reparameterization trick we describe in this paper. Their work was developed independently of ours." (related work)
- Future work: (i) hierarchical / convolutional encoder-decoder, (ii) 時系列 (DBN), (iii) global parameters への SGVB 適用 (full VB, appendix F), (iv) latent 付き supervised model。

## Related Papers

- Hinton+ 1995, wake-sleep algorithm — 主たる baseline。連続潜在変数 online 学習の唯一の既存手法だが2目的関数で下界に対応しない。
- Rezende+ 2014 "Stochastic back-propagation and variational inference in deep latent gaussian models" (arXiv:1401.4082) — **独立並行発見**、同じ reparameterization。
- Hoffman+ 2013, Stochastic Variational Inference — stochastic variational inference への関心が高まっている文脈として引用。
- Blei+ 2012 "Variational Bayesian inference with Stochastic Search" (ICML 2012), Ranganath+ 2013 "Black Box Variational Inference" — score-function 推定の control variate 系、SGVB の比較対照。
- Salimans & Knowles 2013 — exponential family 用に類似 reparameterization を独立に使用。
- Roweis 1998 — PCA = linear-Gaussian モデルの ML 解、線形ケースでの auto-encoder と確率モデルの古典的接続。
- Vincent+ 2010, Stacked Denoising Autoencoders / Bengio+ 2013, Representation Learning — autoencoder 系列の文脈、infomax との関係。
- Gregor+ 2013 (DARN) — 離散潜在変数の有向モデルに対する auto-encoding 法。
- Bengio+ 2013, Generative Stochastic Networks — noisy auto-encoder で Markov chain transition を学習。
- Salakhutdinov & Larochelle 2010 — Deep Boltzmann Machine への recognition model 適用、無向モデルでの先行例。
- Duchi+ 2010, Adagrad — 最適化器。
- Duane+ 1987, Hybrid Monte Carlo — MCEM baseline のサンプラー。
