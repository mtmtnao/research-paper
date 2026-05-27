# Evolution Strategies as a Scalable Alternative to Reinforcement Learning（深層強化学習に対するブラックボックス最適化・分散最適化の位置づけ）

- arXiv: https://arxiv.org/abs/1703.03864
- 一次ソース: ../papers/arXiv-1703.03864v2/
- 正規ノート: ../notes/arXiv-1703.03864v2.md

---

## 一言で言うと

Evolution Strategies (ES) を、Q-learning や Policy Gradients のような MDP ベースの強化学習手法の代替として検討し、MuJoCo と Atari で「深層 RL の難しい環境でも競争力があり、特に多数 CPU へのスケールが非常に良い」と示す論文である。中心的な主張は、common random numbers に基づく通信戦略により、random seeds を事前に同期しておけば worker 間の反復中の通信を scalar return にでき、1,440 workers / CPU cores 規模でもほぼ線形に高速化できる、という点にある。

## 何を議論する論文か

- **問題設定**: ニューラルネットワーク policy のパラメータ $\theta$ を直接最適化し、環境から得られる stochastic return $F(\theta)$ を大きくする。環境は MuJoCo の連続制御タスクと Atari 2600 ゲームで、論文は environment or policy の derivatives にアクセスできない、またはそれらが存在しない場合を扱える black-box optimization として ES を位置づける。
- **対象範囲 / 仮定**: ES は black-box optimization として扱われる。論文の実装では、population distribution を平均 $\theta$、固定共分散 $\sigma^2 I$ の isotropic multivariate Gaussian とし、$\sigma$ は学習中に適応させない固定 hyperparameter とする（§2.1: "we did not see benefit from adapting $\sigma$ during training"）。
- **既存研究との差分**: 進化戦略や NES 自体は新規ではない。差分は、深層 RL ベンチマークで policy gradient 系と比較し、shared random seeds によって各 worker が摂動を再構成できる分散実装を用意し、通信を scalar return にほぼ限定した点である（Algorithm 2, §2.1）。
- **この論文で答えたい問い**: Black-box optimization methods は Q-learning や policy gradients より hard RL problems に弱いと見られがちだが、本当に現代の深層 RL タスクで実用的な解法になり得るのか。特に、長い horizon、遅延報酬、frame-skip、value function 推定、backpropagation、分散通信の観点で、MDP ベース RL とどう違うのかを議論する。

## 背景と前提

- **MDP ベース RL**: 論文が比較対象にする主流は Q-learning と Policy Gradients である。導入部では、これらを Markov Decision Process (MDP) formalism と value functions に基づく RL algorithms と位置づける。Atari from pixels、helicopter aerobatics、Go が導入部の成功例として挙げられている。
- **direct policy search / neuro-evolution**: 環境や policy の微分を使わず、policy parameter を直接探索する系統である。ニューラルネットワークに適用する場合は neuro-evolution と呼ばれる、と§1で説明される。
- **Evolution Strategies (ES)**: 各 generation で parameter vector を perturb/mutate し、fitness を評価し、高得点の vector をもとに次世代を作る black-box optimization の一群である。本論文の手法は Natural Evolution Strategies (NES) に属し、Sehnke et al. [2010] と closely related とされる。
- **policy gradient との違い**: policy gradient は action space に noise を入れて勾配を推定する。一方、本論文の ES は parameter space に Gaussian noise を入れ、episode 全体の return だけから更新方向を推定する（§3）。
- **baseline との関係**: MuJoCo では Trust Region Policy Optimization (TRPO) が主比較対象で、Atari では DQN、A3C、HyperNEAT、著者らの synchronous A3C variant である A2C と比較される（Table 2）。

## 提案手法

### コアアイデア

本論文の ES は、policy parameter $\theta$ の周りに Gaussian perturbation $\epsilon$ を加えた $\theta+\sigma\epsilon$ を複数評価し、return $F_i$ で重みづけした perturbation の和を使って $\theta$ を更新する。環境や action selection が非微分でも、目的関数を parameter space で Gaussian smoothing したものとして扱えるため、score function estimator によって勾配推定ができる。

分散化の要点は、各 worker が同じ random seed の集合を知っていることである。worker $i$ は自分の perturbation で episode を実行して scalar return $F_i$ を得る。他の worker の perturbation は seed から再生成できるので、通信する必要があるのは Algorithm 2 line 8 の "Send all scalar returns $F_i$ from each worker to every other worker" である。これにより、policy gradient methods のように full gradient を通信しない。

実装上は、antithetic sampling/mirrored sampling、rank transformation による fitness shaping、weight decay を併用する。Atari では DeepMind の convolutional architecture への単純な Gaussian parameter perturbation が、状態に関係なく同じ action を取る policy を生みやすかったため、virtual batch normalization を policy specification に入れる（§2.2）。MuJoCo では、hopping と swimming tasks で ES の action を各 action component につき 10 bins に離散化した（§4.1）。

### 重要な定義・数式

$$
\nabla_\psi \mathbb{E}_{\theta \sim p_\psi} F(\theta)
=
\mathbb{E}_{\theta \sim p_\psi}\{F(\theta)\nabla_\psi \log p_\psi(\theta)\}
$$

**式の意味**: NES が population distribution $p_\psi(\theta)$ のパラメータ $\psi$ を、平均 objective value $\mathbb{E}_{\theta \sim p_\psi}F(\theta)$ が増える方向へ更新するための score function estimator である。§2 では REINFORCE と類似の形として導入される。

**記号の定義**:
- $F(\theta)$ ... parameter $\theta$ に対する objective function。本論文の RL 設定では環境から得られる stochastic return。
- $p_\psi(\theta)$ ... parameter vector の population distribution。
- $\psi$ ... population distribution 自体を決めるパラメータ。
- $\nabla_\psi \log p_\psi(\theta)$ ... sampling distribution の log probability の score。

**この論文での役割**: ES を単なる試行錯誤ではなく、population distribution 上の stochastic gradient ascent として位置づける基礎式である。この後、$p_\psi$ を固定共分散の Gaussian に限定することで、実際の Algorithm 1 の更新式につながる。

$$
\mathbb{E}_{\theta \sim p_\psi}F(\theta)
=
\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}F(\theta+\sigma\epsilon)
$$

**式の意味**: 論文は population distribution を平均 $\theta$、固定共分散 $\sigma^2 I$ の isotropic multivariate Gaussian として扱い、元の objective $F$ を Gaussian-blurred version として最適化する。§2 では、環境や discrete action による non-smoothness をこの smoothing で扱える、と説明される。

**記号の定義**:
- $\theta$ ... policy $\pi_\theta$ の parameter vector。ここでは Gaussian population の mean parameter としても使われる。
- $\epsilon \sim \mathcal{N}(0,I)$ ... 標準正規分布から引いた perturbation vector。
- $\sigma$ ... noise standard deviation。論文の実装では fixed hyperparameter。
- $F(\theta+\sigma\epsilon)$ ... 摂動後の policy parameter で episode を実行して得る return。

**この論文での役割**: ES が parameter space に noise を置くという設計を明確にする式である。policy gradient が action space に noise を置くのに対し、この論文は parameter perturbation による smoothing を中心に議論する。

$$
\nabla_\theta \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}F(\theta+\sigma\epsilon)
=
\frac{1}{\sigma}\mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}\{F(\theta+\sigma\epsilon)\epsilon\}
$$

**式の意味**: Gaussian-smoothed objective の $\theta$ に関する勾配を、摂動後 return と perturbation の積の期待値で推定できることを示す。論文本文ではこの式の直後に、sample で近似して Algorithm 1 を得ると説明される。

**記号の定義**:
- $\nabla_\theta$ ... mean policy parameter $\theta$ に関する勾配。
- $F(\theta+\sigma\epsilon)$ ... perturbation $\epsilon$ を加えた policy の return。
- $\epsilon$ ... 勾配方向の推定に使う Gaussian perturbation。
- $\sigma$ ... perturbation の scale。

**この論文での役割**: backpropagation や環境の微分を使わず、episode return のみから更新方向を得るための中核式である。通信が scalar return で足りる理由も、この式が $F_i$ と seed から再生成できる $\epsilon_i$ だけを必要とするためである。

$$
\theta_{t+1} \leftarrow \theta_t + \alpha \frac{1}{n\sigma}\sum_{i=1}^{n}F_i\epsilon_i
$$

**式の意味**: Algorithm 1 line 5 の実際の parameter update である。$n$ 個の perturbation を評価し、return $F_i$ で重みづけした perturbation の平均方向へ、learning rate $\alpha$ で進む。

**記号の定義**:
- $\theta_t$ ... iteration $t$ における policy parameter。
- $\theta_{t+1}$ ... 更新後の policy parameter。
- $\alpha$ ... learning rate。
- $n$ ... 1 iteration で評価する perturbation/worker の数。
- $F_i$ ... $F(\theta_t+\sigma\epsilon_i)$、すなわち $i$ 番目の摂動 policy の return。
- $\epsilon_i$ ... $i$ 番目の Gaussian perturbation。

**この論文での役割**: 論文の手法を実装可能な更新式に落としたもの。Algorithm 2 では各 worker が同じ式を、他 worker の $\epsilon_j$ を seed から再構成して実行する。

$$
\begin{aligned}
\mathrm{Var}[\nabla_\theta F_{PG}(\theta)] &\approx \mathrm{Var}[R(a)]\mathrm{Var}[\nabla_\theta \log p(a;\theta)],\\
\mathrm{Var}[\nabla_\theta F_{ES}(\theta)] &\approx \mathrm{Var}[R(a)]\mathrm{Var}[\nabla_\theta \log p(\tilde{\theta};\theta)].
\end{aligned}
$$

**式の意味**: §3.1 で、simple Monte Carlo (REINFORCE) と good baseline を仮定したときの、policy gradient と ES の勾配推定分散を比較する近似式である。policy gradient 側では $\nabla_\theta \log p(a;\theta)=\sum_{t=1}^{T}\nabla_\theta \log p(a_t;\theta)$ が $T$ 個の項の和になるため、分散が episode length $T$ にほぼ線形に増えると説明される。

**記号の定義**:
- $F_{PG}(\theta)$ ... action space に noise を入れる policy gradient 側の smoothed objective。
- $F_{ES}(\theta)$ ... parameter space に noise を入れる ES 側の smoothed objective。
- $R(a)$ ... action sequence $a=\{a_1,\ldots,a_T\}$ の return。
- $p(a;\theta)$ ... policy が action sequence を生む確率。
- $\tilde{\theta}$ ... $\theta$ に Gaussian perturbation を加えた parameter。
- $T$ ... episode 内の time steps 数。

**この論文での役割**: ES が長い episode、遅延報酬、良い value function estimate がない設定で有利になり得る、という著者の理論的説明を支える。frame-skip に対する頑健性の実験（Figure 2）もこの議論につながる。

### 実装 / アルゴリズム上の要点

1. 各 iteration で $\epsilon_i \sim \mathcal{N}(0,I)$ を sample し、policy parameter を $\theta_t+\sigma\epsilon_i$ にして episode return $F_i$ を計算する。
2. 実装では、training 開始時に大きな Gaussian noise block を各 worker が作り、iteration ごとにその subset を random index で取り出して parameter に足す。この perturbation は厳密には iteration 間で independent ではないが、著者は実用上問題にならなかったと述べる（§2.1）。
3. 並列版では、worker は known random seeds を共有する。各 worker は return $F_i$ を全 worker に送り、他 worker の perturbation $\epsilon_j$ は seed から再構成し、同じ $\theta_{t+1}$ を独立に計算する（Algorithm 2）。
4. variance reduction と安定化のため、$\epsilon$ と $-\epsilon$ をペアで評価する antithetic/mirrored sampling、return の rank transformation による fitness shaping、weight decay を使う。
5. Atari では virtual batch normalization を使う。これは normalizing statistics を計算する minibatch を training 開始時に選び固定する batch normalization と説明され、random weights 初期の policy が入力画像の小さな変化に敏感になるようにする（§2.2）。
6. MuJoCo では基本的に 2 つの 64-unit hidden layers と tanh nonlinearities をもつ multilayer perceptron を TRPO と ES の両方で使う。hopping と swimming tasks では ES の action component を 10 bins に離散化する。

## 実験・結果

- **データセット / ベンチマーク**: MuJoCo physics simulator 上の OpenAI Gym continuous robotic control problems、OpenAI Gym の Atari 2600 games 51 本、3D Humanoid walking、Pong の frame-skip 実験。
- **比較対象 / baseline**: MuJoCo では highly tuned TRPO。Atari では DQN、A3C FF, 1 day、HyperNEAT、ES FF, 1 hour、A2C FF（著者らの synchronous variant of A3C）。Table 2 caption では、ES は deterministic policy evaluation を 10 re-runs、最大 30 random initial no-ops で平均したものとされる。
- **指標**: MuJoCo では TRPO が 5 million timesteps で到達した learning progress の 25%, 50%, 75%, 100% に対し、ES timesteps / TRPO timesteps の比を報告する（Table 1, Table 3）。Atari では raw pixel input で training 後の final score。parallelization では 3D Humanoid で score 6000 に達するまでの median time。frame-skip 実験では Pong の learning curves。
- **主な結果**: MuJoCo Table 1 の 100% 到達比は HalfCheetah 0.58、Hopper 6.94、InvertedDoublePendulum 1.23、InvertedPendulum 0.88、Swimmer 0.30、Walker2d 7.88。著者は hard environments で 10x 未満の sample complexity penalty、simple environments で最大 3x 良い sample complexity と要約する。
- **主な結果**: Atari では全ゲームを 1 billion frames で学習し、720 CPUs on Amazon EC2 により 1 game あたり約 1 hour と報告する。A3C の published 1-day results は 320 million frames で、ES は backpropagation と value function を使わないため、ほぼ同じ neural network computation 量と説明される。最終性能は A3C より 23 games で良く、28 games で悪い（§4.2）。
- **主な結果**: 3D Humanoid では 18 cores で 657 minutes、1,440 cores で 10 minutes で score 6000 に到達する（Figure 1）。実験は 7 回繰り返し、median time を報告する。著者は CPU cores 数に対する linear speedup を主張する。
- **主な結果**: Pong で frame-skip $\in \{1,2,3,4\}$ を変えても learning curves が very similar で、Figure 2 caption では各 run が around 100 weight updates で収束するとされる。
- **著者が主張する貢献**: ES は MuJoCo/Atari の難しい deep RL タスクで競争力があり、MDP-based RL より worker 間通信が軽く、value function approximation や temporal discounting を必要としない。さらに、parameter-space exploration により MuJoCo humanoid で sideways/backwards walking のような TRPO では観測されない gait を学習した、と§1で述べる。

## 妥当性と限界

- **この主張を支える根拠**: 手法面では Algorithm 1/2 により、更新に必要なのが scalar return $F_i$ と再構成可能な perturbation $\epsilon_i$ であることが明確に示されている。実験面では、MuJoCo Table 1/Table 3、Atari Table 2、Figure 1 の scaling、Figure 2 の frame-skip 実験が主要な根拠である。
- **この主張を支える根拠**: §3.1 の分散比較は、policy gradient の score term が time steps $T$ 個の和になるのに対し、ES の parameter perturbation term は $T$ に依存しないという形で、長い episode に対する ES の利点を説明する。ただしこの議論は "Suppose the correlation between the return and the individual actions is low" と "simple Monte Carlo (REINFORCE) with a good baseline" を仮定している。
- **著者が認めている limitations / future work**: Atari では A3C に対して better on 23 games tested, and worse on 28 と§1/§4.2で報告される。Atari では A3C に対して between 3x and 10x as much data を使ったと§1で述べる。$\sigma$ については adapting $\sigma$ during training の benefit を見なかったため固定 hyperparameter とし、indirect encodings は future work とする（§2.1）。
- **著者が認めている limitations / future work**: §6 では、MDP-based RL が less well-suited な long time horizons と complicated reward structure の問題、特に meta-learning/learning-to-learn への適用を future work とする。また、low precision neural network implementations と ES を組み合わせることも今後の課題として挙げる。
- **読者として注意すべき点**: ES の成功は、単純な Gaussian perturbation だけでなく、virtual batch normalization、MuJoCo の action discretization、antithetic/mirrored sampling、rank transformation による fitness shaping、weight decay などの実装上の工夫と一緒に報告されている。§1 finding 1 では、virtual batch normalization と other reparameterizations がないと ES は brittle だったと著者自身が述べる。
- **読者として注意すべき点**: "intrinsic dimension" の議論（§3.2）は、パラメータ数そのものではなく問題の effective dimension が重要だという説明である。ただし、実タスクの intrinsic dimension を直接測った実験は TeX/PDF 中には示されていない。
- **追加で確認したい実験 / 疑問**: virtual batch normalization、rank-based fitness shaping、antithetic sampling、action discretization をそれぞれ外した ablation は、本文・表にはまとまって示されていない。ES の本質的な寄与と parameterization/engineering の寄与を分けて確認したい。
- **追加で確認したい実験 / 疑問**: Algorithm 2 は all-to-all に scalar return を送る。本文で 1,440 workers でも total time の small fraction と明記されるのは Algorithm 2 lines 9-12 の perturbation reconstruction と update であり、より大きい worker 数やより大きい network での通信・再構成コストの詳細な分解は本文にはない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Evolution Strategies (ES)**: parameter vector の population を perturb し、fitness/return に基づいて次の parameter を作る black-box optimization の一群。本論文では NES 型の score-function estimator を使う。
- **Natural Evolution Strategies (NES)**: population distribution $p_\psi(\theta)$ のパラメータ $\psi$ を stochastic gradient ascent で最適化する ES。本論文の手法はこの系統に属する。
- **black-box optimization**: 環境や policy の微分、transition function の明示形を使わず、入力した parameter と出てきた return だけで最適化する立場。
- **stochastic return $F(\theta)$**: policy parameter $\theta$ で episode を実行して環境から得る return。論文では objective function として扱われる。
- **parameter perturbation**: action ではなく policy parameter $\theta$ に noise を加えること。ES の exploration はこれにより生じる。
- **common random numbers / shared random seeds**: worker 間で random seed を共有し、他 worker が使った perturbation を通信せず再生成する仕組み。Algorithm 2 の通信量削減の核である。
- **scalar return**: 各 worker が他 worker に送る episode return $F_i$。full gradient を送らない点が policy gradient methods との違いとして強調される。
- **antithetic sampling / mirrored sampling**: Gaussian noise vector $\epsilon$ に対して $\epsilon$ と $-\epsilon$ の両方を評価する variance reduction 手法。
- **fitness shaping**: return をそのまま使わず rank transformation して更新に使う方法。outlier individuals の影響を除き、early local optima に落ちる傾向を下げると説明される。
- **virtual batch normalization**: normalizing statistics 用の minibatch を training 開始時に固定する batch normalization。Atari の policy が入力画像の小さな変化に反応し、十分な action variety を出すために使われる。
- **frame-skip / action frequency**: agent が simulator より低い頻度で action を決める設定。MDP-based RL では重要な parameter だが、ES の gradient estimate は episode length に invariant と著者は主張する。
- **intrinsic dimension**: §3.2 で、見かけの parameter dimension ではなく optimization problem の実効的な難しさを指す説明概念。大きい network が必ず ES に不利とは限らない、という主張に使われる。
- **TRPO**: MuJoCo の主 baseline。ES と同じ 2 層 64 hidden unit tanh MLP policy architecture で比較される。
- **A3C / A2C**: Atari の主 baseline。A3C は published 1-day results、A2C は著者らの synchronous variant で、Table 2 では 320M training frames と同じ評価設定が使われる。

## 読む順番の提案

- まず Abstract と §1 を読み、著者が列挙する 5 つの key findings を確認する。正規ノートの "Summary（著者の主張）" に対応する。
- 次に §2 と Algorithm 1 を読み、NES の score function estimator、Gaussian-smoothed objective、更新式 $\theta_{t+1}$ を押さえる。正規ノートの Algorithm 1 の更新式メモに対応する。
- Algorithm 2 と §2.1 を読み、shared random seeds と scalar return broadcast の仕組みを見る。正規ノートの "shared random seeds による分散" の要点に対応する。
- §2.2 を読み、virtual batch normalization と MuJoCo action discretization がなぜ必要になったかを確認する。これは正規ノートの brittle/reparameterization に関する注意につながる。
- §3.1 と §3.2 を読み、policy gradient との分散比較、long horizon、intrinsic dimension の議論を押さえる。正規ノートの "Takeaway" の理論的な土台である。
- 最後に Table 1、Table 2、Figure 1、Figure 2、§6 を読む。MuJoCo/Atari の数値、1,440 cores の scaling、frame-skip 実験、future work を確認できる。正規ノートの "Critical Thoughts" は、この結果表を見ながら読むと評価と事実を分けやすい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1703.03864v2/`（`es.tex` は `es_arxiv_v2.pdf` を `\includepdf` するラッパー）
- 正規ノート: `notes/arXiv-1703.03864v2.md`
