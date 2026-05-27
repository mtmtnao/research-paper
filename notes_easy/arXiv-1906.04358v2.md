# Weight Agnostic Neural Networks（重み学習を外してアーキテクチャの帰納バイアスを測る研究）

- arXiv: https://arxiv.org/abs/1906.04358
- 一次ソース: ../papers/arXiv-1906.04358v2/
- 正規ノート: ../notes/arXiv-1906.04358v2.md

---

## 一言で言うと

この論文は、ニューラルネットワークの性能を「学習された個別の重み」ではなく「アーキテクチャそのもの」がどこまで担えるかを調べるため、全接続に単一の共有重みを割り当てても動く Weight Agnostic Neural Networks (WANNs) を探索する。著者は、連続制御 3 タスクと MNIST で、重みを明示的に学習しなくても有用な挙動や「much higher than chance test accuracy of $\sim$ 92\%」が得られると主張している（`00_intro.tex`）。

## 何を議論する論文か

- **問題設定**: NAS は「訓練後に良いアーキテクチャ」を探すが、著者は「アーキテクチャ自体が解をどの程度エンコードできるか」を問う。`21_methodOverview.tex` では、NAS の解では「The weights are the solution」と述べ、WANN では重み訓練を重みサンプリングに置き換える。
- **対象範囲 / 仮定**: 探索対象はフィードフォワードなネットワーク位相で、接続重みは全接続で単一値を共有する。実験では共有重み値として $[-2,-1,-0.5,+0.5,+1,+2]$ を使い、入力から出力への接続、隠れノード、活性化関数を進化的に変える（`23_methodDetails.tex`）。
- **既存研究との差分**: NEAT は位相と重みを同時に最適化するが、本論文は重み値を無視し、位相操作だけを使う。Network pruning は大きな訓練済みネットワークから接続を削るのに対し、WANN は接続が少ない初期ネットワークから複雑さを足す（`10_relwork.tex`）。
- **この論文で答えたい問い**: 個別重みを学習せず、単一共有重みをランダムまたは少数の値で与えるだけでも、タスクに有効な帰納バイアスを持つ小さなネットワーク位相を見つけられるか。

## 背景と前提

- NAS / architecture search では、候補アーキテクチャごとに重みを訓練してから評価する inner loop が高コストになる。本論文はその inner loop を避け、アーキテクチャを「weight training なし」で評価する。
- Bayesian Neural Networks では重みを分布からサンプルするが、多くの場合は分布のパラメータを学習する。WANN は全ネットワークで共有される 1 個の重みを固定分布から扱う点が異なる。
- Algorithmic Information Theory / MDL は、「性能が近いなら記述長の短いモデルを好む」という動機づけとして使われる。WANN 探索では接続数を複雑さの proxy として、性能だけでなく小ささも評価する。
- 本論文での「weight agnostic」は、完全に重みが不要という意味ではない。全接続で一貫した単一共有重みが使える、という制約下で動くという意味であり、Table 1 では個別ランダム重みでは性能が大きく落ちる。
- 連続制御の比較対象は文献中の fixed topology feed-forward network policies で、BipedalWalker-v2 は `ha2018designrl`、CarRacing-v0 は `ha2018worldmodels` の設定と結びついている（`31_control.tex`, `60_appendix.tex`）。

## 提案手法

### コアアイデア

WANN Search は、重み空間を高次元に探索する代わりに、全接続に同じ共有重みを入れてネットワークを評価する。`21_methodOverview.tex` の表現では、weight-sharing によって「the number of weight values is reduced to one」となり、単一重み値を少数回サンプルするだけでアーキテクチャ性能を近似できる。

探索は、隠れノードなし・入出力間が疎に接続された最小ネットワーク集団から始まる。各候補は複数の共有重み値で rollout され、累積報酬を記録される。その後、性能と複雑さで順位付けされ、tournament selection で選ばれた上位ネットワークに `insert node`、`add connection`、`change activation` のいずれかを適用して次世代を作る。

活性化関数は、linear, step, sin, cosine, Gaussian, tanh, sigmoid, absolute value, invert (negative linear), ReLU の 10 種である（Figure `fig:topsearch`, `24_topSearch.tex`）。著者は appendix で、具体的な重み値に頼れないため入力間の関係をネットワークにエンコードする必要があり、symmetry や repetition のような関係を含めるとより compact になると説明している。

### 重要な定義・数式

TeX 中には式番号付きの目的関数や更新式はほぼない。以下は、TeX 中の数式表記と本文の評価条件を、読解用に最小限だけ式の形で整理したものである。

$$
w \in [-2,-1,-0.5,+0.5,+1,+2]
$$

**式の意味**: WANN 探索で各 rollout に使う共有重み値の固定列である。`23_methodDetails.tex` では、この固定列を使う理由を「decrease the variance between evaluations」と説明している。

**記号の定義**:
- $w$ ... 全接続に同時に割り当てられる単一の共有重み値
- $[-2,-1,-0.5,+0.5,+1,+2]$ ... 実験で使われる共有重み値の列

**この論文での役割**: 高次元の重み空間探索を 1 次元の共有重みサンプリングに落とし、性能がアーキテクチャ由来かどうかを観察するための中核的制約である。脚注では、$|w|>3$ では活性化が飽和しやすく、$w$ が 0 付近では情報が出力にほとんど流れないため省いたと述べられている。

$$
\text{mean performance} =
\frac{1}{6}\sum_{w \in \{-2,-1,-0.5,+0.5,+1,+2\}} R(w)
$$

**式の意味**: `23_methodDetails.tex` の「averaging its cumulative reward over all rollouts using these different weight values」を記号化したもの。論文中にこの式そのものは置かれていないが、平均性能の定義はこの文章で与えられている。

**記号の定義**:
- $R(w)$ ... 共有重み $w$ を全接続に割り当てた rollout で得られる累積報酬
- $6$ ... 評価に使う共有重み値の個数

**この論文での役割**: 候補位相が「特定の 1 つの重み値でだけ偶然よい」のではなく、複数の共有重み値で期待性能を持つかを評価する。Table 1 の `Random Shared Weight` や `Tuned Shared Weight` の解釈にもつながる。

$$
\text{ranking criteria} =
\begin{cases}
(\text{mean performance},\ \#\text{connections}) & 80\% \\
(\text{mean performance},\ \text{max performance}) & 20\%
\end{cases}
$$

**式の意味**: `23_methodDetails.tex` の 80% / 20% の順位付け規則を整理したもの。80% の場合は平均性能と接続数で ranking し、20% の場合は平均性能と最大性能で ranking する。

**記号の定義**:
- $\text{mean performance}$ ... 複数共有重み値での平均累積報酬
- $\#\text{connections}$ ... ネットワークの接続数。少ないほど低複雑度として好まれる
- $\text{max performance}$ ... 共有重み値のうち最良値での性能

**この論文での役割**: MDL 的な小ささを保ちながら、構造追加が複数段階そろわないと性能向上しない場合の stepping stone を残すための緩和である。TeX では dominance relations に基づく multi-objective optimization として説明され、NSGA-II が引用されている。

$$
w_i \sim \mathcal{U}(-2,2), \qquad
w_{\mathrm{shared}} \sim \mathcal{U}(-2,2)
$$

**式の意味**: Table 1 の 4 条件のうち、`Random weights` と `Random shared weight` の違いを表す。`31_control.tex` では前者を「individual weights drawn from $\mathcal{U}(-2,2)$」、後者を「a single shared weight drawn from $\mathcal{U}(-2,2)$」と定義している。

**記号の定義**:
- $w_i$ ... 接続 $i$ ごとの個別重み
- $w_{\mathrm{shared}}$ ... 全接続で共有される単一重み
- $\mathcal{U}(-2,2)$ ... 区間 $(-2,2)$ の一様分布

**この論文での役割**: WANN が「個別重みをランダムにしてもよい」わけではなく、「同じ共有重みを全接続に使う」ことが重要である点を切り分ける。実際、Table 1 では WANN の Swing Up が `Random Weights` で $57 \pm 121$、`Random Shared Weight` で $515 \pm 58$ と大きく異なる。

### 実装 / アルゴリズム上の要点

- step1: 隠れノードなし、入出力間が一部だけ接続された sparse な初期集団を作る。
- step2: 各ネットワークを複数 rollout し、rollout ごとに全接続へ単一共有重み値を割り当てる。
- step3: 共有重み値ごとの累積報酬から mean performance と max performance を求め、接続数も含めて dominance relations で ranking する。
- step4: tournament selection で親を選び、`insert node`、`add connection`、`change activation` のいずれかを適用する。`add connection` は feed-forward property を保つように行う。
- step5: 世代を繰り返し、徐々に複雑で高性能な weight agnostic topologies を得る。
- ハイパーパラメータは appendix Table にあり、Population Size / Generations は SwingUp が 192 / 1024、Biped が 480 / 2048、CarRace が 64 / 1024、MNIST が 960 / 4096。変異確率は全タスクで Change Activation 50%、Add Node 25%、Add Connection 25%。

## 実験・結果

- **データセット / ベンチマーク**: 連続制御は `CartPoleSwingUp`、`BipedalWalker-v2`、`CarRacing-v0`。MNIST は [28x28] から [16x16] に downsample し、OpenCV で deskew、pixel intensity を 0 から 1 に正規化する（`60_appendix.tex`）。
- **比較対象 / baseline**: 連続制御では文献由来の fixed topology feed-forward policies と比較する。Swing Up Cartpole baseline は 1 hidden layer of 10 units、71 weight parameters。Bipedal Walker は `estool` / `ha2018designrl`、Car Racing は `ha2018worldmodels` のコードとモデルを使い、Car Racing baseline は VAE と RNN を固定して controller の 867 parameters を自由重みとして扱う（`60_appendix.tex`）。
- **指標**: 連続制御は 100 random trials の平均累積報酬（Table 1）。MNIST は accuracy と softmax cross entropy に基づく reward で評価され、各評価では training set から 1000 samples をランダムに与える（`34_class.tex`, `60_appendix.tex`）。
- **主な結果**: Table 1 の累積報酬は次の通り。`Random Shared Weight` 条件で WANN が fixed topology を大きく上回る一方、`Tuned Weights` では fixed topology も強く、Biped と CarRacing では fixed topology 側が最高値である。

| Task / Model | Random Weights | Random Shared Weight | Tuned Shared Weight | Tuned Weights |
|---|---:|---:|---:|---:|
| Swing Up / WANN | $57 \pm 121$ | $515 \pm 58$ | $723 \pm 16$ | $932 \pm 6$ |
| Swing Up / Fixed Topology | $21 \pm 43$ | $7 \pm 2$ | $8 \pm 1$ | $918 \pm 7$ |
| Biped / WANN | $-46 \pm 54$ | $51 \pm 108$ | $261 \pm 58$ | $332 \pm 1$ |
| Biped / Fixed Topology | $-129 \pm 28$ | $-107 \pm 12$ | $-35 \pm 23$ | $347 \pm 1$ |
| CarRacing / WANN | $-69 \pm 31$ | $375 \pm 177$ | $608 \pm 161$ | $893 \pm 74$ |
| CarRacing / Fixed Topology | $-82 \pm 13$ | $-85 \pm 27$ | $-37 \pm 36$ | $906 \pm 21$ |

- **主な結果**: Figure `fig:nets` の champion networks は Swing up 52 connections、Biped 210 connections、Car Racing 245 connections。Biped の WANN は 17 of the 25 possible inputs のみを使い、SOTA baseline の 2804 connections より少ないと著者は述べる。
- **主な結果**: CarRacing の WANN controller は VAE の 16 latent dimensions のみを入力とし、`ha2018worldmodels` baseline が使う pre-trained RNN world model の hidden states は使わない。それでも comparable score を達成した、と著者は述べる。
- **主な結果**: MNIST では 256 inputs と 10 outputs を持つ WANN を使う。Figure `fig:mnistfull` の caption は MNIST classifier network を 1849 connections として示す。本文は、単一重み値に制限されても「a single layer neural network with thousands of weights trained by gradient descent」と同程度に分類できると述べ、introduction では test accuracy $\sim$ 92% と書く。
- **著者が主張する貢献**: 重みを明示的に学習せず、共有重みを少数値でサンプルするだけで、連続制御と MNIST に有効な小さいアーキテクチャを探索できることを示した点。さらに、得られたネットワークが小さいため、CartPoleSwingUp の generation 32 / 128 のように、入力間の関係や制御機構を図から解釈できる点も強調されている。

## 妥当性と限界

- **この主張を支える根拠**: Table 1 は、fixed topology が random shared weight でほぼ機能しない一方、WANN は同じ条件で一定の累積報酬を出すことを示す。これは「性能が重み訓練ではなく位相の帰納バイアスに由来する」という主張の主要根拠である。
- **この主張を支える根拠**: `31_control.tex` の CartPoleSwingUp 解析では、generation 32 で $x$ position に対する three inverters が中心への attractor を作り、Gaussian activation on $d\theta$ と位置制御の相互作用が swing-up behavior を作る、と著者は説明する。これは小さな WANN を解釈できるという主張を支える。
- **著者が認めている limitations / future work**: WANN は個別ランダム重みには弱い。Table 1 で Swing Up の WANN は `Random Weights` が $57 \pm 121$ だが、`Random Shared Weight` は $515 \pm 58$ であり、`31_control.tex` も consistency of sign が重要だと述べる。
- **著者が認めている limitations / future work**: CarRacing では WANN は feed-forward controller であり、baseline は RNN world model の hidden states も使う。著者は future work として feed-forward constraint を外し、recurrent connections with memory states を発達させることを挙げている。
- **著者が認めている limitations / future work**: Discussion では、WANN が CNN の性能に一致しないのは驚きではないとし、既存構造の再配置だけでなく new building blocks を発見すべきだと述べる。appendix では ReLU / sigmoid でも可能だったかもしれないが、linear activations だけで達成できたとは確信していない、としている。
- **読者として注意すべき点**: MNIST の具体的な test accuracy は introduction の $\sim$ 92% と Figure の視覚情報に依存し、連続制御の Table 1 ほど詳細な数値表は本文 TeX にない。したがって MNIST の比較は proof of concept として読むのが安全である。
- **読者として注意すべき点**: 探索計算量は軽いとは限らない。appendix の hyperparameters では MNIST が population 960、generations 4096 であり、勾配学習や他の NAS と同一計算予算での比較は TeX 中に明示されていない。
- **追加で確認したい実験 / 疑問**: 活性化関数プールを削った ablation、共有重みの個数を 1 から少数個に増やす実験、同一計算予算での NAS / pruning / differentiable search との比較、VAE に依存しない CarRacing 入力での比較は、この論文の主張範囲をさらに明確にする。

## 用語メモ

- **Weight Agnostic Neural Network (WANN)**: 全接続に単一共有重みを入れてもタスク性能を示すよう探索されたネットワーク位相。重み値が不要というより、個別重みを細かく学習しない設計を指す。
- **shared weight**: すべての接続に同じ値として割り当てる重み。実験では固定列 $[-2,-1,-0.5,+0.5,+1,+2]$ と、一様分布 $\mathcal{U}(-2,2)$ の条件が出てくる。
- **topology / architecture**: ノード、接続、活性化関数を含むネットワーク構造。本論文ではこれが解をエンコードする対象になる。
- **rollout**: ある共有重み値を割り当てたネットワークをタスク上で実行し、累積報酬を測る評価単位。
- **mean performance / max performance**: 複数共有重み値に対する平均性能と、単一の最良共有重み値での性能。探索時の ranking に使われる。
- **connection cost / MDL**: 接続数を複雑さとして扱い、同程度の性能ならより小さいネットワークを好む考え方。`23_methodDetails.tex` では MDL と connection cost technique が動機づけとして出る。
- **dominance relations**: 複数目的を手作りの単一報酬にまとめず、性能と複雑さの支配関係で順位を決める方法。TeX では NSGA-II の文献が引用される。
- **NEAT**: 位相と重みを同時に進化させる neuroevolution アルゴリズム。本論文では NEAT に着想を得た位相操作だけを使い、重み値は探索しない。
- **Random weights / Random shared weight**: 前者は接続ごとに独立に $\mathcal{U}(-2,2)$ から引く条件、後者は全接続で共有する 1 つの値を $\mathcal{U}(-2,2)$ から引く条件。Table 1 の解釈で重要な区別。
- **Tuned shared weight / Tuned weights**: 前者は共有重み 1 個を範囲 $(-2,2)$ で選ぶ条件、後者は個別重みを population-based REINFORCE で調整する条件。
- **self-contained ensemble**: MNIST で、同じ WANN を複数の共有重み値で instantiate し、それぞれを異なる classifier とみなして vote する使い方。Figure `fig:mnist` では digit ごとに有利な重み値が異なることが示される。
- **VAE latent dimensions**: CarRacing-v0 で画像ピクセルを 16 次元に圧縮した入力表現。WANN controller はこの 16 latent dimensions のみを受け取る。
- **supermask**: concurrent work `zhou2019deconstructing` が扱う、ランダム初期化重みでも画像認識性能を出す pruning mask。本論文は pruning を補完的手法として位置づける。

## 読む順番の提案

- まず正規ノート `notes/arXiv-1906.04358v2.md` の `Summary` と `Takeaway` を読み、共有重み・位相探索・Table 1 の要点を掴む。
- 次に `main.tex` の abstract と `tex/00_intro.tex` を読み、問いが「訓練後に強い architecture」ではなく「architecture alone がどこまで解を持つか」であることを確認する。
- 方法は `tex/21_methodOverview.tex`、Figure `fig:overview`、`tex/23_methodDetails.tex`、Figure `fig:topsearch` の順に読む。共有重み値、3 つの位相操作、80% / 20% の ranking が正規ノートの手法要約に対応する。
- 結果は `tex/31_control.tex` と `tex/32_controlTable.tex` を先に読む。Table 1 の 4 条件、特に `Random weights` と `Random Shared Weight` の差が、この論文の主張を読む中心になる。
- MNIST は `tex/34_class.tex`、Figure `fig:mnist`、`tex/37_classDiagram.tex`、appendix の `MNIST` 節を読む。本文の数値表が少ないため、正規ノートの MNIST への批判的コメントと合わせて確認する。
- 限界と発展は `tex/40_discuss.tex` と `tex/60_appendix.tex` の「Have you also thought about trying ... ?」を読む。固定単一重み 0.7 から 0.6 への弱さ、JAX fine-tuning、活性化関数プールの不確実性が、正規ノートの `Critical Thoughts` に直接つながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1906.04358v2/`
- 正規ノート: `notes/arXiv-1906.04358v2.md`
