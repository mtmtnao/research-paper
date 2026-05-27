# Generative Adversarial Nets（adversarial process による生成モデル推定フレームワーク）

- arXiv: https://arxiv.org/abs/1406.2661
- 一次ソース: ../papers/arXiv-1406.2661v1/
- 正規ノート: ../notes/arXiv-1406.2661v1.md

---

## 一言で言うと

この論文は、生成器 $G$ と判別器 $D$ を同時に学習する adversarial process によって、明示的な尤度計算や Markov chain を使わずに生成モデルを推定する枠組みを提案する。著者は、任意関数の非パラメトリック極限では $p_g=p_\text{data}$ かつ $D(\bm{x})=\frac{1}{2}$ が唯一の解になると示し、MNIST / TFD / CIFAR-10 の生成サンプルと、MNIST / TFD での Parzen window による定量評価を報告する（`adversarial.tex` abstract, §Theoretical Results, Table 1, Figure 2）。

## 何を議論する論文か

- **問題設定**: 深層生成モデルで、データ分布 $p_\text{data}$ をよく近似する生成分布 $p_g$ を学習したい。従来の最大尤度推定や関連手法では、partition function とその勾配、training 中の inference などの「intractable probabilistic computations」が問題になりやすい（§Introduction, §Related work, Table 2）。
- **対象範囲 / 仮定**: 本文で扱う特殊例は、ノイズ $\bm{z}\sim p_{\bm{z}}$ を多層パーセプトロン $G(\bm{z};\theta_g)$ に通してサンプルを作り、別の多層パーセプトロン $D(\bm{x};\theta_d)$ が「データ由来か、$G$ 由来か」を判定する場合である（§Adversarial nets）。理論結果は「$G$ と $D$ が十分な capacity を持つ」「non-parametric setting」「十分小さい更新」などの理想化を置く（§Theoretical Results）。
- **既存研究との差分**: RBM / DBM などの undirected graphical models は partition function とその勾配が難しく、MCMC の mixing が問題になる。DBN は undirected / directed の両方の困難を持つ。Score matching と NCE は正規化定数までの解析的な確率密度を要求する。GSN は sampling に Markov chain を使う。Adversarial nets は、生成手続きを微分可能関数として置き、判別器からの勾配で学習するため、学習・サンプリングのどちらにも Markov chain を必要としない、という位置づけである（§Related work, Table 2）。
- **この論文で答えたい問い**: 「本物データと生成サンプルを区別する判別器」を相手にして生成器を学習すれば、明示的な尤度や近似推論なしに $p_\text{data}$ を復元できるのか。さらに、その枠組みは実験上も既存の生成モデルと競争的なサンプルを出せるのか。

## 背景と前提

- 生成モデルは、データの確率分布を表し、そこから新しいサンプルを生成するモデルである。この論文では $G$ 自体が明示的な密度 $p_g(\bm{x})$ を返すのではなく、$\bm{z}\sim p_{\bm{z}}$ を $G(\bm{z})$ に写すことで暗黙に $p_g$ を定める。
- 判別モデルは、入力がどのクラスに属するかを返すモデルである。この論文の $D(\bm{x})$ は、$\bm{x}$ が $p_g$ ではなく training data から来た確率を表す単一スカラーである。
- minimax game は、一方が目的関数を大きくし、もう一方が小さくする最適化である。ここでは $D$ が本物と偽物を正しく分類する方向に $V(D,G)$ を最大化し、$G$ が $D$ を間違えさせる方向に最小化する。
- 非パラメトリック極限の理論では、$G$ や $D$ を特定の有限個のパラメータを持つ MLP としてではなく、十分自由な関数または確率密度関数の空間で考える。実際の実装は $\theta_g,\theta_d$ を持つ MLP なので、この理論保証がそのまま有限モデルに移るわけではない（§Theoretical Results 末尾）。
- 先行研究との関係では、NCE も discriminative training criterion で生成モデルを fit する点が近い。ただし NCE では生成モデル自身が固定 noise distribution からのサンプルを識別するのに対し、この論文では別の判別器 $D$ を学習し、生成器 $G$ はノイズからデータ空間への写像として定義される（§Related work）。

## 提案手法

### コアアイデア

Adversarial nets は、生成器 $G$ と判別器 $D$ を二人の minimax game として同時に訓練する。$G$ は input noise variables の prior $p_{\bm{z}}(\bm{z})$ からサンプルした $\bm{z}$ をデータ空間へ写す微分可能関数であり、$D$ は入力 $\bm{x}$ が training data から来た確率を返す微分可能関数である。

学習では、$D$ は training examples と $G$ のサンプルの両方に正しいラベルを付けるように更新され、$G$ は $D$ が間違える確率を大きくするように更新される。実装では $D$ を完全最適化すると計算上重く、有限データでは overfitting するため、$D$ を $k$ steps 更新してから $G$ を 1 step 更新する。Algorithm 1 のキャプションでは、実験では最も安い選択として $k=1$ を使ったと明記されている。

この設計で重要なのは、$G$ が本物データを直接入力として受け取らず、$D$ を通る勾配だけで更新される点である。著者はこれを、入力成分が generator のパラメータへ直接コピーされにくいという統計的利点の可能性として述べる（§Advantages and disadvantages）。

### 重要な定義・数式

$$
\min_G \max_D V(D, G) =
\mathbb{E}_{\bm{x} \sim p_{\text{data}}(\bm{x})}[\log D(\bm{x})]
+ \mathbb{E}_{\bm{z} \sim p_{\bm{z}}(\bm{z})}[\log (1 - D(G(\bm{z})))].
$$

**式の意味**: adversarial nets の中心となる minimax 目的関数である。$D$ は本物データに高い $D(\bm{x})$ を、生成サンプル $G(\bm{z})$ に低い $D(G(\bm{z}))$ を与えるように最大化し、$G$ はその逆に、生成サンプルを本物と誤認させるように最小化する（Eq. `minimaxgame-definition`）。

**記号の定義**:
- $G(\bm{z};\theta_g)$ ... noise $\bm{z}$ をデータ空間へ写す generator
- $D(\bm{x};\theta_d)$ ... $\bm{x}$ が data rather than $p_g$ から来た確率を返す discriminator
- $p_\text{data}$ ... training data を生成する分布
- $p_{\bm{z}}$ ... generator へ入れる input noise variables の prior
- $p_g$ ... $\bm{z}\sim p_{\bm{z}}$ のときの $G(\bm{z})$ が暗黙に定める分布
- $V(D,G)$ ... $D$ が最大化し、$G$ が最小化する value function

**この論文での役割**: 手法の訓練規準そのものであり、Proposition 1、Theorem 1、Algorithm 1 はこの式から展開される。Table 2 では、adversarial models を他の深層生成モデル系統と比べた操作上の性質が整理される。

$$
D^*_G(\bm{x}) =
\frac{p_\text{data}(\bm{x})}{p_\text{data}(\bm{x}) + p_g(\bm{x})}
$$

**式の意味**: $G$ を固定したとき、最適な discriminator は、その点 $\bm{x}$ が data distribution から来る相対的な割合を返す。これは Proposition 1 の式であり、$D$ の学習を「$Y=1$ が $p_\text{data}$、$Y=0$ が $p_g$ を表す conditional probability $P(Y=y|\bm{x})$ の log-likelihood 最大化」と見られることに対応する。

**記号の定義**:
- $D^*_G$ ... 固定された $G$ に対する最適 discriminator
- $p_\text{data}(\bm{x})$ ... 点 $\bm{x}$ における data distribution の密度
- $p_g(\bm{x})$ ... 点 $\bm{x}$ における generator distribution の密度
- $Supp(p_\text{data}) \cup Supp(p_g)$ ... discriminator が定義されればよい support の範囲

**この論文での役割**: $D$ が十分に最適化されたときに $G$ がどの規準を最小化しているかを導く入口である。Figure 1 の説明でも、$D$ がこの形へ収束してから $G$ の更新が data と分類されやすい領域へ誘導される、という直観が示される。

$$
\begin{aligned}
C(G)
&= \max_D V(G,D) \\
&= -\log(4)
+ KL \left(p_\text{data} \left \| \frac{p_\text{data} + p_g}{2} \right. \right)
+ KL \left(p_g \left \| \frac{p_\text{data} + p_g}{2} \right. \right) \\
&= - \log(4) + 2 \cdot JSD \left(p_\text{data} \left \| p_g \right. \right).
\end{aligned}
$$

**式の意味**: $D$ を常に最適にした仮想的な generator の訓練規準 $C(G)$ は、定数 $-\log 4$ と Jensen--Shannon divergence の和として書ける。$JSD$ は非負で、2 つの分布が等しいときだけ 0 なので、$C(G)$ の global minimum は $p_g=p_\text{data}$ で、値は $-\log 4$ になる（Theorem 1）。

**記号の定義**:
- $C(G)$ ... $D$ を最大化したあとの virtual training criterion
- $KL(\cdot\|\cdot)$ ... Kullback--Leibler divergence
- $JSD(p_\text{data}\|p_g)$ ... data distribution と model distribution の Jensen--Shannon divergence
- $-\log(4)$ ... $p_g=p_\text{data}$ かつ $D^*_G(\bm{x})=\frac{1}{2}$ のときに達する値

**この論文での役割**: 「adversarial training は理想化された条件下で data generating process を復元する」という中心的な理論主張の根拠である。ただし、この証明は有限の MLP パラメータ空間ではなく、確率密度関数の空間での議論である。

$$
\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m
\left[
\log D\left(\bm{x}^{(i)}\right)
+ \log \left(1-D\left(G\left(\bm{z}^{(i)}\right)\right)\right)
\right],
\qquad
\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m
\log \left(1-D\left(G\left(\bm{z}^{(i)}\right)\right)\right).
$$

**式の意味**: Algorithm 1 の minibatch stochastic gradient descent 更新である。左は discriminator を stochastic gradient ascent で更新する方向、右は generator を stochastic gradient descent で更新する方向を表す。

**記号の定義**:
- $m$ ... minibatch size
- $\{\bm{x}^{(1)},\dots,\bm{x}^{(m)}\}$ ... $p_\text{data}(\bm{x})$ からサンプルした minibatch
- $\{\bm{z}^{(1)},\dots,\bm{z}^{(m)}\}$ ... noise prior からサンプルした minibatch
- $\theta_d,\theta_g$ ... discriminator と generator のパラメータ
- $k$ ... $G$ を 1 step 更新する前に $D$ を更新する回数

**この論文での役割**: 理論的な minimax 目的を、実際にどのような交互更新で近似するかを示す。Algorithm 1 では任意の standard gradient-based learning rule が使えるとし、実験では momentum を使ったと書かれている。なお Algorithm 1 の noise prior は TeX 上で $p_g(\bm{z})$ と書かれる箇所があるが、本文の定義では input noise prior は $p_{\bm{z}}(\bm{z})$ であるため、読む際は表記ゆれに注意する。

### 実装 / アルゴリズム上の要点

- step1: noise prior $p_{\bm{z}}(\bm{z})$ を定め、generator $G(\bm{z};\theta_g)$ を MLP として用意する。$G$ は $\bm{z}$ をデータ空間のサンプルへ写す。
- step2: discriminator $D(\bm{x};\theta_d)$ を MLP として用意する。出力は、$\bm{x}$ が $p_g$ ではなく training data から来た確率を表す単一スカラーである。
- step3: 各 training iteration で $D$ を $k$ steps 更新する。各 step では noise minibatch と data minibatch をサンプルし、$D$ の stochastic gradient を ascent する。
- step4: $G$ を 1 step 更新する。Algorithm 1 の形式では $\log(1-D(G(\bm{z})))$ の stochastic gradient を descent する。
- step5: 実際には、学習初期に $D$ が生成サンプルを高信頼に拒否すると $\log(1-D(G(\bm{z})))$ が saturate し、$G$ への勾配が弱くなる。そのため著者は、$G$ については $\log(1-D(G(\bm{z})))$ を最小化する代わりに $\log D(G(\bm{z}))$ を最大化できると述べる。この目的は同じ fixed point を持ち、初期学習でより強い勾配を与える（§Adversarial nets）。
- step6: 実験では generator nets に rectifier linear activations と sigmoid activations の混合を使い、discriminator net には maxout activations を使う。dropout は discriminator の訓練に適用する。理論的枠組みは generator の中間層で dropout や他の noise を使うことを許すが、実験では noise を generator の bottommost layer への入力としてのみ使った（§Experiments）。
- step7: サンプリング時は $G$ の forward propagation のみでよい。論文は、training / generation のどちらにも Markov chains や unrolled approximate inference networks は不要だと強調する（abstract, §Introduction）。

## 実験・結果

- **データセット / ベンチマーク**: MNIST、Toronto Face Database (TFD)、CIFAR-10。CIFAR-10 では fully connected model と、convolutional discriminator + "deconvolutional" generator のサンプルが Figure 2 に示される。
- **比較対象 / baseline**: Table 1 の定量比較は DBN、Stacked CAE、Deep GSN、Adversarial nets。Table 2 では、Deep directed graphical models、Deep undirected graphical models、Generative autoencoders、Adversarial models の操作上の困難を比較する。
- **指標**: 生成器 $G$ の samples に Gaussian Parzen window を fit し、test set data の log-likelihood を報告する。Gaussian の $\sigma$ は validation set で cross validation する。MNIST では test set 上の mean log-likelihood と standard error of the mean、TFD では dataset folds across の standard error を報告する。MNIST は real-valued version の dataset で比較している（Table 1 caption）。
- **主な結果**: Table 1 の Parzen window-based log-likelihood estimates は次の通りである。
  - DBN: MNIST $138 \pm 2$、TFD $1909 \pm 66$
  - Stacked CAE: MNIST $121 \pm 1.6$、TFD $2110 \pm 50$
  - Deep GSN: MNIST $214 \pm 1.1$、TFD $1890 \pm 29$
  - Adversarial nets: MNIST $225 \pm 2$、TFD $2057 \pm 26$
- **主な結果**: MNIST では Adversarial nets が Table 1 の最大値 $225 \pm 2$ を示す。TFD では Adversarial nets は $2057 \pm 26$ で、点推定としては Stacked CAE の $2110 \pm 50$ がより大きい。
- **主な結果**: Figure 2 は MNIST、TFD、CIFAR-10 fully connected model、CIFAR-10 convolutional discriminator + "deconvolutional" generator のサンプルを示す。キャプションは、右端の列が nearest training example であり、サンプルは "fair random draws, not cherry-picked"、Markov chain mixing に依存しないので uncorrelated だと述べる。
- **主な結果**: Figure 3 は full model の $\bm{z}$ space で座標を線形補間したときに得られる digits を示す。
- **著者が主張する貢献**: 論文は、adversarial modeling framework の viable さを示したと結論づける。理論面では $p_g=p_\text{data}$ が global optimum であることと、条件付きの下で Algorithm 1 が $p_g$ を $p_\text{data}$ へ収束させることを述べる。計算面では、Markov chains が不要、backprop のみで勾配を得る、learning 中に inference が不要、広い種類の関数を model に組み込める、という利点を主張する（§Advantages and disadvantages）。

## 妥当性と限界

- **この主張を支える根拠**: Proposition 1 は固定された $G$ に対する最適 $D^*_G$ を導く。Theorem 1 は、$C(G)=-\log(4)+2\cdot JSD(p_\text{data}\|p_g)$ により、非パラメトリック極限での global minimum が $p_g=p_\text{data}$ のみであることを示す。さらに Algorithm 1 の収束命題は、$D$ が各 step で最適に達し、$p_g$ が規準を改善するように更新されるなら $p_g$ が $p_\text{data}$ に収束すると述べる。
- **この主張を支える根拠**: 実験面では、Table 1 の Parzen window log-likelihood と Figure 2 / Figure 3 のサンプル可視化が、framework の potential を示す根拠として提示される。ただし著者自身は、Figure 2 のサンプルが既存手法より良いとは claim しないと明記し、少なくとも literature の better generative models と competitive だと述べるに留める。
- **著者が認めている limitations / future work**: 主な disadvantage は、$p_g(\bm{x})$ の explicit representation がないこと、training 中に $D$ と $G$ をうまく synchronized しなければならないことである。特に $D$ を更新せずに $G$ を訓練しすぎると、$G$ が多くの $\mathbf{z}$ を同じ $\mathbf{x}$ に collapse し、$p_\text{data}$ を表すだけの diversity を失う "the Helvetica scenario" を避ける必要がある（§Advantages and disadvantages）。
- **著者が認めている limitations / future work**: Parzen window による尤度推定は "somewhat high variance" で、高次元空間ではうまく働かないと著者自身が書く。その一方で、尤度を直接評価できずサンプルだけを出す生成モデルをどう評価するかは further research を動機づける、と述べる（§Experiments）。
- **著者が認めている limitations / future work**: future work として、conditional generative model $p(\bm{x}\mid\bm{c})$、$\bm{x}$ から $\bm{z}$ を予測する auxiliary network による learned approximate inference、全ての条件付き分布 $p(\bm{x}_S \mid \bm{x}_{\not S})$ の近似、semi-supervised learning、$G$ と $D$ の coordination や $\mathbf{z}$ の sampling distribution 改善による efficiency improvements が挙げられている（§Conclusions and future work）。
- **読者として注意すべき点**: 収束証明は確率密度関数の空間での話であり、実際の MLP パラメータ空間では multiple critical points が生じると著者は述べる。したがって「GAN の実装が常に安定に大域最適へ行く」という保証ではない。
- **読者として注意すべき点**: CIFAR-10 についてはサンプル画像はあるが、Table 1 のような定量比較はない。CIFAR-10 で既存手法より数値的に優れている、とは TeX 中では主張されていない。
- **読者として注意すべき点**: 学習率、具体的な層幅、dropout 率などの詳細ハイパーパラメータは本文には列挙されず、脚注で code and hyperparameters の GitHub URL が示されるだけである。
- **追加で確認したい実験 / 疑問**: $k=1$ は "least expensive option" として選ばれているが、$k$ の値と $D$ / $G$ の同期や Helvetica scenario の関係は本文では定量化されていない。元の minimax generator loss と $\log D(G(\bm{z}))$ を最大化する代替目的の差、Parzen window 以外の評価、CIFAR-10 の定量評価、アーキテクチャとハイパーパラメータの感度は追加確認したい点である。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **adversarial nets** ... この論文では、generator と discriminator がともに multilayer perceptrons である特殊ケースを指す。より広い adversarial modeling framework の一実装である。
- **generator $G$** ... input noise variables $\bm{z}$ を受け取り、データ空間のサンプル $G(\bm{z};\theta_g)$ を出す微分可能関数。明示的な $p_g(\bm{x})$ を返す密度モデルではない。
- **discriminator $D$** ... 入力 $\bm{x}$ が generator ではなく training data から来た確率を返す MLP。固定された $G$ に対しては $D^*_G(\bm{x})=\frac{p_\text{data}(\bm{x})}{p_\text{data}(\bm{x})+p_g(\bm{x})}$ が最適形になる。
- **$p_\text{data}$** ... training data を生成する分布。理論では復元対象の真の data generating process として扱われる。
- **$p_{\bm{z}}$** ... generator への noise prior。TeX の本文では $p_{\bm{z}}(\bm{z})$ と定義される。
- **$p_g$** ... $\bm{z}\sim p_{\bm{z}}$ に対する $G(\bm{z})$ の分布。Theoretical Results ではこの分布そのものが最適化対象として扱われる。
- **non-parametric setting** ... $G,D$ を有限 MLP パラメータで制限せず、確率密度関数の空間で convergence を議論する理想化。Theorem 1 と収束命題の前提である。
- **virtual training criterion $C(G)$** ... $D$ を最適化したあとの generator の規準 $\max_D V(G,D)$。JSD と結びつけられ、global optimality の証明に使われる。
- **JSD** ... Jensen--Shannon divergence。論文では $C(G)=-\log(4)+2\cdot JSD(p_\text{data}\|p_g)$ として現れ、$p_g=p_\text{data}$ のときだけ最小になることを示すために使われる。
- **partition function** ... undirected graphical models で unnormalized potentials を正規化するための全状態にわたる総和または積分。RBM / DBM の学習で難しい量として挙げられる。
- **MCMC / Markov chain mixing** ... 従来の生成モデルで partition function 勾配や sampling に使われる近似手段。論文は adversarial nets では training / sampling に Markov chains が不要だと強調する。
- **NCE** ... Noise-contrastive estimation。discriminative criterion で生成モデルを fit する点は近いが、この論文では固定 noise distribution ではなく、学習される generator と別の discriminator の対戦を使う。
- **GSN** ... Generative stochastic network。parameterized Markov chain の一 step を学習する生成モデルとして説明され、adversarial nets は sampling に Markov chain を必要としない点で比較される。
- **Parzen window** ... この論文の定量評価で、$G$ の samples に Gaussian Parzen window を fit し、test set log-likelihood を推定する方法。高分散で高次元では弱いという注意も本文にある。
- **Helvetica scenario** ... $D$ を更新せずに $G$ を訓練しすぎたとき、$G$ が多数の $\mathbf{z}$ を同じ $\mathbf{x}$ に collapse して diversity を失う失敗状況として著者が挙げる名称。
- **maxout / rectifier linear / sigmoid / dropout** ... 実験で使われた構成要素。generator は rectifier linear と sigmoid の混合、discriminator は maxout、dropout は discriminator training に使う。

## 読む順番の提案

- まず `adversarial.tex` の abstract と §Introduction を読み、正規ノート `notes/arXiv-1406.2661v1.md` の Summary の「問題」と「手法」がどの主張に対応するか確認する。特に "minimax two-player game"、"No approximate inference or Markov chains" という位置づけを見る。
- 次に §Adversarial nets の Eq. `minimaxgame-definition`、Algorithm 1、Figure 1 を読む。ここが正規ノートの minimax objective、$k$ steps、$\log D(G(\bm{z}))$ を最大化する代替目的の説明につながる。
- その後 §Theoretical Results の Proposition 1 と Theorem 1 を読む。$D^*_G$、$C(G)$、JSD への変形を追うと、正規ノートの「JSD への対応」「global minimum が $p_g=p_\text{data}$」が読めるようになる。
- 実験は §Experiments の Table 1、Figure 2、Figure 3 を先に見る。正規ノートの Parzen window 数値、memorization でないこと、$\bm{z}$ 空間補間の記述はここに対応する。
- 最後に Table 2、§Advantages and disadvantages、§Conclusions and future work を読む。正規ノートの Critical Thoughts にある explicit likelihood の欠如、Helvetica scenario、Parzen 評価の弱さ、future work の列挙を、著者の主張と読者側の評価に分けて確認できる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1406.2661v1/`
- 正規ノート: `notes/arXiv-1406.2661v1.md`
