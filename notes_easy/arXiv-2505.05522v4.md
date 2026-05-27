# Continuous Thought Machines（neural dynamics を表現として使う recurrent architecture）

- arXiv: https://arxiv.org/abs/2505.05522
- 一次ソース: ../papers/arXiv-2505.05522v4/
- 正規ノート: ../notes/arXiv-2505.05522v4.md

---

## 一言で言うと

この論文は、現代のニューラルネットワークが捨ててきたニューロン単位の時間的挙動を、`neuron-level temporal processing` と `neural synchronization as a latent representation` で明示的に使う **Continuous Thought Machine (CTM)** を提案する。著者は、CTM が 2D maze、ImageNet-1K、cumulative parity、Q&A MNIST などで、内部 tick に沿った逐次的処理、adaptive compute、解釈しやすい attention strategy を示すと主張している（abstract, `main.tex` L156; contributions, L212-L215）。

## 何を議論する論文か

- **問題設定**: 生物脳では neural dynamics と neural timing が情報処理に重要だが、modern NNs は大規模学習のためにニューロン間相互作用の精密なタイミングを抽象化している、という問題意識から出発する（Introduction, `main.tex` L209-L212）。論文の問いは、時間的に展開するニューロン活動そのものを、学習可能で計算可能な表現として使えるか、である。
- **対象範囲 / 仮定**: CTM は詳細な生物物理を再現するモデルではない。著者は「biological realism」と「computational tractability」のバランスを狙うと書き、continuous-valued pre-activations を扱う NLM、gradient-based deep learning に載る differentiable framework、attention、MLP synapse を使う（abstract L156; Related Work L236-L237）。
- **既存研究との差分**: ACT / PonderNet / early-exit は明示的な halting mechanism を使うが、CTM は内部 tick ごとの loss と certainty から adaptive compute が出ると位置づける（Related Work L230-L231; Loss L308-L319）。RAM や recurrent reasoning 系は recurrent state や external glimpses を使うのに対し、CTM は neuron-level histories から生成される synchronization を primary representation にする（L233-L234）。
- **この論文で答えたい問い**: CTM の 2 要素、すなわち private NLM と synchronization representation が、maze の内部 world model、ImageNet の `look around`、parity の sequential algorithm、Q&A MNIST の memory/retrieval のような挙動を支えるかを、複数タスクで示す（Experimental Evaluation L322-L324）。

## 背景と前提

- **内部 tick と recurrence**: CTM の時間軸 $t \in \{1,\dots,T\}$ は、入力データが持つ系列長とは別の「内部計算ステップ」である。静的画像でも $T$ 回処理を進め、表現を更新する（`Continuous Thought`, Eq. 周辺 L254-L255）。
- **pre-activation / post-activation**: synapse model が出す $\mathbf{a}^t$ が pre-activation、NLM が出す $\mathbf{z}^{t+1}$ が post-activation である。Appendix の glossary でも、pre-activation は NLM 入力、post-activation は tick $t$ の neuron state と定義される（Table \ref{tab:concept-glossary}, L540-L547）。
- **NLM**: Neuron-Level Model は、各ニューロン $d$ ごとの private MLP で、直近 $M$ tick の pre-activation history $\mathbf{A}_d^t$ を読む。標準的な activation function より parameter count が増えると limitations でも述べられるため、activation 関数の位置に時間履歴を読む per-neuron MLP を入れる設計として押さえるとよい（L268-L273; L513-L514）。
- **synchronization**: この論文での synchronization は、post-activation histories の dot product である。単なる可視化用の post-hoc synchrony ではなく、attention query と output logits を作る latent representation として学習される（L276-L289）。
- **baseline との関係**: 主な比較対象は LSTM と feed-forward (FF) baseline である。Related Work では ACT, PonderNet, Sparse Universal Transformer, RAM, Liquid Time-Constant Networks, SNN, Reichert & Serre の synchrony などに触れるが、実験で同条件比較する主対象は多くの場合 LSTM / FF である（Related Work L230-L239; maze setup L334; CIFAR-10 L1074-L1079）。

## 提案手法

### コアアイデア

CTM は、同じ $D$ 次元の latent neurons を内部 tick に沿って繰り返し更新する。各 tick では、まず synapse model $f_{\theta_{\text{syn}}}$ が前回の post-activation $\mathbf{z}^t$ と attention output $\mathbf{o}^t$ から pre-activation $\mathbf{a}^t$ を作る。次に、各ニューロン $d$ の private NLM $g_{\theta_d}$ が、自分の pre-activation history $\mathbf{A}_d^t$ から次の post-activation $\mathbf{z}_d^{t+1}$ を出す。最後に post-activation history $\mathbf{Z}^t$ から synchronization matrix $\mathbf{S}^t$ を作り、その一部を $\mathbf{S}^t_\text{out}$ と $\mathbf{S}^t_\text{action}$ として使って、出力 $\mathbf{y}^t$ と attention query $\mathbf{q}^t$ を得る（architecture caption, `main.tex` L245-L249; method L252-L295）。

重要なのは、下流タスクへ直接渡す表現が snapshot $\mathbf{z}^t$ ではなく synchronization である点である。著者は脚注で、snapshot representation は下流タスクに強く結びついて dynamics を制約する一方、synchronization はそれを decouple すると説明している（Synchronization, L276）。

### 重要な定義・数式

$$
\mathbf{a}^t = f_{\theta_{\text{syn}}}(\text{concat}(\mathbf{z}^t, \mathbf{o}^t)) \in \mathbb{R}^D
$$

**式の意味**: Eq. \ref{eq:synapses} の synapse update で、前 tick の neuron state と attention output から、各ニューロンへ入る pre-activation を作る。

**記号の定義**:
- $\mathbf{a}^t$ ... tick $t$ の pre-activation。NLM への入力になる。
- $f_{\theta_{\text{syn}}}$ ... neuron 間で情報共有する synapse model。本文では U-NET-esque MLP が性能がよかったと述べ、Appendix で depth $k$、bottleneck width 16 の U-Net 風構造を説明する。
- $\mathbf{z}^t$ ... tick $t$ の post-activation / latent state。
- $\mathbf{o}^t$ ... cross-attention の出力。FeatureExtractor から得た data features を query した結果。
- $D$ ... latent neurons の数、すなわち $\mathbf{z}^t$ と $\mathbf{a}^t$ の幅。

**この論文での役割**: CTM の recurrent part で、ニューロン間の情報伝達を担う。Appendix では synapse model を depth $k$、bottleneck width 16 の U-Net 風構造にする説明がある（Fig. \ref{fig:synapses}, L633-L640）。

$$
\mathbf{z}_d^{t+1} = g_{\theta_d}(\mathbf{A}_d^t)
$$

**式の意味**: Eq. \ref{eq:nlms} の NLM update で、ニューロン $d$ が自分自身の直近 $M$ tick の pre-activation history を private MLP に通して、次 tick の post-activation を作る。

**記号の定義**:
- $d \in \{1,\dots,D\}$ ... ニューロンの index。
- $g_{\theta_d}$ ... ニューロン $d$ 専用の NLM。本文では depth 1 MLP, hidden width $d_\text{hidden}$ と書かれる。
- $\mathbf{A}_d^t$ ... $\mathbf{A}^t \in \mathbb{R}^{D \times M}$ の $d$ 行。直近 $M$ 個の pre-activations。
- $\mathbf{z}_d^{t+1}$ ... ニューロン $d$ の次 tick の post-activation。

**この論文での役割**: CTM の「neuron-level temporal processing」の中核である。standard activation functions ではなく、各 neuron が private weights で時間履歴を読むため、著者は complex neuron-level activity を作れると位置づけている（L268-L273; L513-L514）。

$$
\mathbf{S}^t = \mathbf{Z}^t \cdot (\mathbf{Z}^t)^\intercal \in \mathbb{R}^{D\times D}
$$

**式の意味**: Eq. \ref{eq:synch} の synchronization matrix で、post-activation histories の内積により、ニューロン $i,j$ の活動履歴がどれだけ同期しているかを表す。

**記号の定義**:
- $\mathbf{Z}^t = [\mathbf{z}^{1}, \mathbf{z}^{2}, \cdots, \mathbf{z}^{t}] \in \mathbb{R}^{D \times t}$ ... tick 1 から $t$ までの post-activation history。
- $\mathbf{S}^t$ ... $D \times D$ の full synchronization matrix。
- $(\cdot)^\intercal$ ... 転置。

**この論文での役割**: CTM は $\mathbf{S}^t$ の一部を latent representation として使う。$\mathbf{S}^t$ は $O(D^2)$ に大きくなるため、実装では neuron pairs を sub-sampling して $\mathbf{S}^t_\text{out}$ と $\mathbf{S}^t_\text{action}$ を作る（L284-L290）。

$$
\mathbf{S}^t_{ij} =
\frac{\left ( \mathbf{Z}^t_{i} \right )^\intercal \cdot \operatorname{diag}(\mathbf{R}^t_{ij}) \cdot \left ( \mathbf{Z}^t_{j} \right )}
{\sqrt{\sum_{\tau=1}^{t}\left[\mathbf{R}^t_{ij}\right]_{\tau}}}
$$

**式の意味**: Eq. \ref{eq:synch-really} の rescaled synchronization で、古い tick と新しい tick の寄与を、pair ごとの learnable exponential decay $r_{ij}$ で重み付けする。

**記号の定義**:
- $\mathbf{S}^t_{ij}$ ... neuron pair $(i,j)$ の synchronization entry。
- $\mathbf{R}^t_{ij}$ ... Eq. \ref{eq:scaling} の decay vector。成分は $\exp(-r_{ij}(t-\tau))$ 型。
- $r_{ij}\ge 0$ ... pair $(i,j)$ の learnable decay rate。
- $\mathbf{Z}^t_i, \mathbf{Z}^t_j$ ... $\mathbf{Z}^t$ の $i$ 行、$j$ 行。

**この論文での役割**: 複数の時間スケールで synchronization を使えるようにする。著者は、ImageNet ではこの decay はほとんど使われず、2D maze ではより使われたと脚注で述べる（L297-L306）。Appendix では $\alpha^t_{ij}, \beta^t_{ij}$ の一次再帰により、選んだ pair 数 $D_\text{sub}$ に対して tick あたり $\mathcal{O}(D_\text{sub})$ で計算できると説明する（L1679-L1708）。

$$
L = \frac{\mathcal{L}^{t_1} + \mathcal{L}^{t_2}}{2}
$$

**式の意味**: Eq. \ref{eq:final_loss} の certainty-based loss で、各データ点について「最小 loss の tick」と「最大 certainty の tick」の 2 点を選び、その cross entropy を平均する。

**記号の定義**:
- $\mathcal{L}^t = \text{CrossEntropy}(\mathbf{y}^t, y_{true})$ ... tick $t$ の分類損失。
- $\mathcal{C}^t$ ... tick $t$ の certainty。本文では $1 -$ normalized entropy。
- $t_1=\text{argmin}(\mathcal{L})$ ... forward pass 内で loss が最小の tick。
- $t_2=\text{argmax}(\mathcal{C})$ ... certainty が最大の tick。
- $\theta_{\text{syn}}, \theta_{d=1\ldots D}$ ... 最適化される synapse と NLM のパラメータ。

**この論文での役割**: 明示的な halting module なしに、入力ごとに有効な内部 tick が変わる `native adaptive computation` を作るための訓練信号である（L308-L319）。ただし $t_1$ は真ラベルに依存するため、推論時の停止規準そのものではない。

### 実装 / アルゴリズム上の要点

- step1: FeatureExtractor で data を keys/values に変換する。ImageNet では constrained ResNet-152 の final average pooling / projection 前の出力を使い、$224\times224$ 入力から $14\times14$ features を得る（ImageNet architecture, L747-L748）。
- step2: 学習可能な $\mathbf{z}^{t=1}$ と initial pre-activation history から開始する（L266; Listing overview L592-L598）。
- step3: synchronization から action representation $\mathbf{S}^t_\text{action}$ を作り、$\mathbf{q}^t = \mathbf{W}_{\text{in}}\mathbf{S}^t_\text{action}$ として cross-attention query にする（Eq. \ref{eq:q}, \ref{eq:attention}）。
- step4: attention output $\mathbf{o}^t$ と $\mathbf{z}^t$ を concat し、synapse model と NLM で $\mathbf{z}^{t+1}$ を得る。
- step5: post-activation history から output representation $\mathbf{S}^t_\text{out}$ を作り、$\mathbf{y}^t = \mathbf{W}_{\text{out}}\mathbf{S}^t_\text{out}$ を出す（Eq. \ref{eq:out}）。
- step6: Dense / Semi-dense / Random pairing のいずれかで synchronization pairs を選ぶ。Dense は $J$ 個の neuron の全 pair、Semi-dense は $J_1,J_2$ の組、Random は $D_\text{out}$ / $D_\text{action}$ 個の pair をランダムに選び、必要なら $n_\text{self}$ 個の self-pair で snapshot representation を回復できる余地を残す（Appendix L646-L662）。

## 実験・結果

- **データセット / ベンチマーク**: 主要実験は 2D mazes、ImageNet-1K、64-length cumulative parity。追加実験は CIFAR-10 with CIFAR-10D/CIFAR-10H human data、CIFAR-100 ablations、$15\times15$ maze ablation、sorting 30 real numbers、Q&A MNIST、RL の CartPole / Acrobot / MiniGrid Four Rooms（L324; L475-L483; appendices）。
- **比較対象 / baseline**: Maze では LSTM 1/2/3 layers と FF baseline。Parity と Q&A MNIST では parameter-matched LSTM。CIFAR-10 では FF と LSTM と human baseline。Maze ablation では CTM (No NLMs)、CTM (No Synch)、LSTM + Synch を比較する（L334; L872; L1074-L1079; L1258-L1270）。
- **指標**: ImageNet は top-1 / top-5 accuracy と calibration、certainty threshold 0.8 での stopping 分析。Maze は route prediction の per-step accuracy / solve rate。Parity は sequence accuracy と attention strategy。Q&A MNIST は test accuracy。Sorting は wait time と CTC output。RL は episode length（CartPole は高いほど良く、Acrobot / MiniGrid 4-Rooms は低いほど良い）（L366-L384; Table \ref{tab:maze_ablation}; L1371-L1430; Fig. \ref{fig:app-rl/training} caption L1630）。
- **主な結果**: 2D maze では、$39\times39$ maze、最大 100 steps、no positional embedding、CTM は $D=2048$, $T=75$, $M=25$、31,998,330 parameters。LSTM baselines は 42,298,688 / 75,869,504 / 109,440,320 parameters、FF は 54,797,632 parameters である（Appendix L675-L700; L716-L723）。本文は CTM が baselines を有意に上回り、$99\times99$ maze へ learned policy の re-application で汎化すると述べる（L337-L359）。
- **主な結果**: ImageNet-1K では ResNet-152 backbone と 50 internal ticks で uncropped top-1 $72.47\%$、top-5 $89.89\%$。著者は SOTA を狙っておらず、hyperparameter search は scope 外と明記する（L365-L366）。certainty threshold $0.8$ なら、多数の instance は 50 tick 中 10 tick 未満で停止可能と述べる（L384）。
- **主な結果**: Parity は 64-length binary sequence の cumulative parity を全位置で予測する。CTM は internal ticks が多いほど精度が上がり、75 / 100 ticks の一部 seed で perfect accuracy に到達する。LSTM は stability / performance に苦戦する（L389-L441）。75 tick, $M=25$ の 3 seed では Run 1 と Run 3 が loss zero、Run 2 が non-zero loss に収束し、Run 1 は reverse order、Run 3 は beginning-to-end attention を示す（L1024-L1042）。
- **主な結果**: Maze ablation では、$15\times15$ maze、約 9M parameters、100000 iterations、2 seeds。CTM は Test Accuracy $94.6\pm0.7\%$ / Solve Rate $65.9\pm5.7\%$、CTM (No NLMs) は $82.9\pm4.4\%$ / $35.0\pm7.2\%$、CTM (No Synch) は $85.1\pm0.5\%$ / $37.5\pm0.7\%$、LSTM + Synch は $82.4\pm0.9\%$ / $33.8\pm3.3\%$（Table \ref{tab:maze_ablation}, L1310-L1323）。
- **主な結果**: Q&A MNIST では、digits と index/operator embeddings を順に観測し、modular addition/subtraction を行う。10 internal ticks per input の CTM は、4 digits / 4 operations の最難 in-distribution task で全 3 seed が over $96\%$ accuracy、対応する 10-tick LSTM は全 seed で at or below $21\%$ と報告される（L1371-L1396）。
- **主な結果**: CIFAR-10 では CTM が FF/LSTM より安定で test performance が良く、calibration も最良と述べられる。CIFAR-10H human baseline との比較では、CTM は human より calibration が良く、LSTM は human under-confidence に沿うと書かれる（L1091-L1097）。TeX 中にはここでの具体的 accuracy 値は明示されていない。
- **主な結果**: Sorting では $\mathcal{N}(0,I_{30})$ からの 30 real numbers を sorting indices と blank token の 31-class output として CTC loss で学習する。wait time は sequence index と `data delta` に関係し、training distribution 外の normal distributions にも generalize すると述べる（L1332-L1364）。
- **著者が主張する貢献**: CTM architecture、NLM、synchronization representation、internal tick による adaptive compute、maze / ImageNet / parity / Q&A MNIST などで観察される interpretable / emergent behavior の提示である（contributions L212-L215; Discussion L506-L517）。

## 妥当性と限界

- **この主張を支える根拠**: Architecture claim は Eq. \ref{eq:synapses}, \ref{eq:nlms}, \ref{eq:synch}, \ref{eq:final_loss} と Listing 群で具体化されている。特に maze ablation は、NLM だけ、synchronization だけ、LSTM + synchronization では solve rate が大きく落ちるため、「NLM と synchronization の組合せが重要」という主張を支える最も直接的な定量証拠である（Table \ref{tab:maze_ablation}）。
- **この主張を支える根拠**: Adaptive compute claim は、loss が tick ごとの $\mathcal{L}^t$ と $\mathcal{C}^t$ を使うこと、および ImageNet で certainty threshold 0.8 により多数の instance が 10/50 tick 未満で停止できるという分析に基づく（L308-L319; L384）。
- **著者が認めている limitations / future work**: CTM は internal sequence を使うため training time が延びる。NLM は standard activation functions より parameter count を増やす。実験は preliminary で SOTA 追求ではなく、breadth を優先したため comparison depth が限定的である（Limitations, L513-L514）。Future Work は language modeling、self-supervised video understanding、lifelong-learning、biologically-inspired memory and plasticity、multi-modal systems など（L516-L517）。
- **読者として注意すべき点**: ImageNet の 72.47% / 89.89% は「新しい計算原理の挙動を見る」ための値であり、同 backbone の標準設定との厳密な SOTA 比較ではない。著者自身も hyperparameter search は scope 外と述べる（L366）。
- **読者として注意すべき点**: $t_1=\arg\min(\mathcal{L})$ は真ラベルを使うため、推論時の early stopping rule とは異なる。論文は certainty threshold による停止可能性を示すが、訓練時の best-loss tick と推論時の stopping を完全に同一視しない方がよい（Loss L309-L319; ImageNet L384）。
- **読者として注意すべき点**: Pairing strategy はタスクごとに異なる。Maze は Dense、ImageNet は Random with $n_\text{self}=32$、parity と Q&A MNIST は Semi-dense と記載されるが、どの戦略をどう選ぶべきかの一般原理は TeX 中には明示されていない（L675-L690; L750-L765; L962-L973; L1442-L1453）。
- **追加で確認したい実験 / 疑問**: ACT / PonderNet / Looped Transformers / Sparse Universal Transformers など、本文で近い位置づけとして言及される adaptive / recurrent reasoning methods との同条件比較は TeX 中にはない。CTM の優位が neural synchronization 固有なのか、内部 tick を持つ別の強い recurrent baseline でも出るのかは、追加検証が必要である。
- **追加で確認したい実験 / 疑問**: Q&A MNIST で synchronization が memory として働くという解釈は、観測 digits が memory window 外に出る設計と高精度に支えられているが、Q&A MNIST 内で `No Synch` ablation は示されていない。Maze ablation と合わせて読む必要がある。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Continuous Thought Machine (CTM)**: 内部 tick に沿って neural dynamics を展開し、synchronization を attention と output の表現に使う recurrent architecture。
- **internal tick**: 入力系列の時間ではなく、モデル内部の計算ステップ。$T=50$ や $T=75$ のように設定される。
- **memory length $M$**: NLM が読む rolling FIFO pre-activation history の長さ。Maze と ImageNet では $M=25$、Q&A MNIST の 10-tick 設定では $M=30$ など。
- **synapse model**: $\mathbf{z}^t$ と $\mathbf{o}^t$ から $\mathbf{a}^t$ を作る recurrent model。本文では U-NET-esque MLP がよかったと述べ、Appendix で U-Net 風構造を説明する。
- **pre-activation $\mathbf{a}^t$**: synapse model の出力で、NLM への入力。
- **post-activation $\mathbf{z}^t$**: NLM の出力で、tick ごとの neuron state。
- **NLM / Neuron-Level Model**: 各 neuron が private parameters を持つ MLP。$\mathbf{A}_d^t$ を入力し、$\mathbf{z}_d^{t+1}$ を出す。
- **neural synchronization**: post-activation histories の dot product。$\mathbf{S}^t=\mathbf{Z}^t(\mathbf{Z}^t)^\intercal$ として定義される。
- **action synchronization**: attention query $\mathbf{q}^t$ を作るための sampled synchronization representation。
- **output synchronization**: logits $\mathbf{y}^t$ を作るための sampled synchronization representation。
- **Dense pairing**: 選んだ $J$ neurons の全 pair を使う synchronization sampling。Maze などで使われる。
- **Semi-dense pairing**: 2 つの subset $J_1,J_2$ 間の pair を使う方式。Parity と Q&A MNIST で使われる。
- **Random pairing**: $D_\text{out}$ / $D_\text{action}$ 個の pair をランダムに選ぶ方式。ImageNet では self-pair $n_\text{self}=32$ も使う。
- **certainty**: $1 -$ normalized entropy。loss の $t_2$ と ImageNet stopping analysis に使われる。
- **native adaptive computation**: explicit halting module を追加せず、内部 tick ごとの loss/certainty dynamics から入力ごとに有効 tick が変わる、という著者の表現。
- **world model / cognitive map**: Maze task で、direct route output と no positional embedding により必要になる内部空間表現として説明される概念（Appendix L736-L739）。
- **over-thinking**: ImageNet の追加例で、正解を一度通過した後に不正解へ移る現象として caption に書かれる（Fig. \ref{fig:imagenet-21202}, L848-L851）。

## 読む順番の提案

- まず abstract と Introduction の contributions（`main.tex` L156, L209-L224）を読み、著者が「neural timing を再導入する」と何を主張しているかを押さえる。正規ノートの Summary / Takeaway はここに対応する。
- 次に Method の Fig. \ref{fig:architecture} caption と Eq. \ref{eq:synapses}, \ref{eq:nlms}, \ref{eq:synch}, \ref{eq:final_loss}（L243-L319）を読む。正規ノートの「NLM」「synchronization」「2 点損失」の記述はこの部分の要約である。
- その後、Appendix の sampling synchronization neurons（L646-L662）を読む。Dense / Semi-dense / Random pairing が実験設定の違いを理解する鍵になる。
- 実験は、まず maze setup/results（L326-L359）と maze architecture/baselines（L668-L723）を読む。次に ablation Table \ref{tab:maze_ablation}（L1310-L1323）を見ると、NLM と synchronization の必要性が最も直接的に分かる。
- ImageNet は、L365-L387 と Appendix L747-L813 を読む。accuracy 値よりも、certainty threshold、calibration の計算方法、prediction aggregation が論点である。
- Parity は L389-L441 を読んでから Appendix L867-L1063 を読む。seed 分散と forward/reverse attention strategy の扱いは正規ノートの Critical Thoughts とつながる。
- Q&A MNIST と sorting は追加実験だが、synchronization を memory / sequential output に使う主張を読むには L1326-L1430 が重要である。
- 最後に Limitations / Future Work（L506-L517）を読み、正規ノートの Critical Thoughts にある「SOTA 比較の浅さ」「training time / parameter count」「比較対象の不足」と照合する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2505.05522v4/`
- 正規ノート: `notes/arXiv-2505.05522v4.md`
