# Training Compute-Optimal Large Language Models（大規模言語モデルの compute-optimal training を見直したスケーリング則の論文）

- arXiv: https://arxiv.org/abs/2203.15556
- 一次ソース: ../papers/arXiv-2203.15556v1/
- 正規ノート: ../notes/arXiv-2203.15556v1.md

---

## 一言で言うと

固定された訓練 FLOPs 予算 $C$ の下で、dense autoregressive Transformer language model のパラメータ数 $N$ と訓練トークン数 $D$ をどう配分すべきかを、400 超の訓練 run から推定した論文。著者らは、既存の大規模 LM は model size に対して training tokens が少ない「undertrained」な状態であり、compute-optimal training では $N$ と $D$ をほぼ同じ割合で増やすべきだと主張し、その検証として Gopher と同じ compute budget で Chinchilla 70B を 1.4T tokens 訓練して、多数の downstream task で Gopher 280B を上回ることを示す。

## 何を議論する論文か

- **問題設定**: 事前に決まった training compute budget の下で、final pre-training loss $L(N,D)$ を最小にする model parameters $N$ と training tokens $D$ の組を選ぶ問題。TeX の中心的な問いは “Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens?”（main.tex, Section `Introduction` / `sec:method`）。
- **対象範囲 / 仮定**: 対象は autoregressive Transformer language model。主解析では compute と optimal model size / token count の関係を power law と仮定する。FLOPs は model parameters と seen training tokens の決定的関数として扱われ、Approach 3 では近似 $\text{FLOPs}(N,D) \approx 6ND$ も使う（main.tex, Eq. `eq:model`, Section `FLOPs computation`）。
- **既存研究との差分**: Kaplan et al. (2020) は、compute が増えると model size を training data より速く増やすべきだとし、表では $N_{opt} \propto C^{0.73}$, $D_{opt} \propto C^{0.27}$ と比較される。一方、この論文の 3 手法はおおむね $a \approx b \approx 0.5$ を得る（Table `tab:comparison`）。
- **この論文で答えたい問い**: 近年の大規模 dense LM が本当に compute-optimal か、もし同じ FLOPs でより小さいモデルをより長く訓練したら性能が上がるか。具体的には、Gopher と同じ $5.76 \times 10^{23}$ FLOPs 付近で、70B parameters / 1.4T tokens の Chinchilla が 280B parameters / 300B tokens の Gopher より良いかを検証する。

## 背景と前提

- この論文での **compute budget** は、訓練に使える FLOPs の総量を指す。著者は「利用できる accelerator 数と利用時間は実務上しばしば事前に決まっている」と述べ、巨大モデルは通常 1 回しか訓練できないため、事前に $N$ と $D$ を決めることが重要だとする。
- **$N$** は model parameters の数、**$D$** は training tokens の数、**$L(N,D)$** はそのモデルをその token 数だけ訓練した後の final pre-training loss。簡単化のため、解析では smoothed training loss を test loss の unbiased estimate として扱う。脚注では infinite data regime、すなわち training tokens が corpus 全体の token 数より少ない条件を挙げている（main.tex, Eq. `eq:model` 前の脚注）。
- **compute-optimal** とは、同じ FLOPs で最も低い loss に到達する設定のこと。ここでの最適性は主に pre-training loss と、その検証としての downstream evaluation で議論される。
- 既存の代表的 dense Transformer LM は、LaMDA 137B / 168B tokens、GPT-3 175B / 300B tokens、Jurassic 178B / 300B tokens、Gopher 280B / 300B tokens、MT-NLG 530B / 270B tokens と整理される（Table `tab:llms`）。著者は、LaMDA 以外の多くが約 300B tokens で訓練されている点を問題視する。
- Kaplan et al. (2020) との重要な違いは learning rate schedule の扱い。著者は、Kaplan et al. が固定 token 数・固定 schedule を使ったため、短い training horizon の intermediate loss が、対応する短い schedule で最適化した場合の final loss を過大評価し、結果として data より model size を速く増やす結論につながったと説明する（main.tex, Section `Related Work`, Figure `fig:cosine`）。
- Chinchilla の評価は Gopher 論文の task subset を大きく踏襲し、直接比較できるようにする。ただし、Chinchilla は Gopher と同じ dataset family を使うものの、sampling distribution、optimizer、tokenizer、optimizer state の precision などにも差がある（main.tex, Section `Model and training details`）。

## 提案手法

### コアアイデア

著者らは、固定 FLOPs で loss を最小にする $N$ と $D$ の関係を 1 つの fitting 手法だけに依存せず、3 つの独立した empirical approach で推定する。すべての approach は、under 70M から over 16B parameters のモデルを、5B から数百B tokens の範囲で訓練した run に基づく。Abstract では “5 to 500 billion tokens”、Introduction では “5B to over 400B tokens” と書かれている。

1 つ目は training curve envelope を使う方法で、固定 model size ごとに異なる training horizons を走らせ、各 FLOP count で最も低い loss を与える run を選ぶ。2 つ目は IsoFLOP profiles で、FLOP 予算を固定し、model size を変えたときの final loss の谷を探す。3 つ目は $\hat L(N,D)$ という parametric loss function を全 run の final loss に fit し、そこから efficient frontier を閉形式で求める。

3 手法の推定が近いため、著者は「model size と training tokens は compute の増加に対してほぼ等しい割合で増やすべき」と結論づける。この予測を実物で試すため、Gopher と同じ compute budget で Chinchilla 70B を 1.4T tokens 訓練する。

### 重要な定義・数式

$$
N_{opt}(C), D_{opt}(C) = \argmin_{N, D \text{ s.t. } \text{FLOPs}(N, D) = C} L(N, D).
$$

**式の意味**: 固定された compute budget $C$ の下で、final pre-training loss $L(N,D)$ を最小にする model size と training tokens の組を定義する式（main.tex, Eq. `eq:model`）。

**記号の定義**:
- $C$ ... training compute budget。FLOPs 単位で測られる。
- $N$ ... model parameters の数。
- $D$ ... training tokens の数。
- $L(N,D)$ ... $N$ parameters のモデルを $D$ tokens 訓練した後の loss。
- $\text{FLOPs}(N,D)$ ... $N$ と $D$ から決まる訓練計算量。
- $N_{opt}(C), D_{opt}(C)$ ... compute budget $C$ に対して loss を最小にする最適な配分。

**この論文での役割**: 論文全体の問題設定そのもの。3 つの approach はすべて、この関数 $N_{opt}(C),D_{opt}(C)$ を経験的に推定する方法として位置づけられる。

$$
N_{opt} \propto C^a, \qquad D_{opt} \propto C^b.
$$

**式の意味**: compute budget が増えたとき、最適な model size と training tokens が power law で増えるという仮定・推定結果の表現。Table `tab:comparison` はこの $a,b$ を比較する。

**記号の定義**:
- $C$ ... training FLOPs。
- $a$ ... compute を増やしたときの optimal parameter count の scaling exponent。
- $b$ ... compute を増やしたときの optimal training tokens の scaling exponent。
- $\propto$ ... 比例関係を表す。

**この論文での役割**: Kaplan et al. (2020) との主要な対立点。Table `tab:comparison` では、Approach 1 が $a=0.50$, $b=0.50$、Approach 2 が $a=0.49$, $b=0.51$、Approach 3 が $a=0.46$, $b=0.54$、Kaplan et al. が $a=0.73$, $b=0.27$ と報告される。

$$
\hat L(N,D) \triangleq E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
$$

**式の意味**: Approach 3 で使う parametric loss function。loss を、理想的生成過程の loss、有限の model size による近似誤差、有限の training tokens / optimization steps による誤差の 3 項に分ける（main.tex, Eq. `eq:decompose`）。

**記号の定義**:
- $\hat L(N,D)$ ... $N,D$ から予測される loss。
- $E$ ... ideal generative process on the data distribution の loss。Appendix では Bayes risk、つまり “entropy of natural text” と説明される。
- $A/N^\alpha$ ... transformer の hypothesis space が有限の $N$ parameters に制限されることによる項。
- $B/D^\beta$ ... 有限個の data points / optimization steps で訓練することによる stochastic approximation / optimization suboptimality の項。
- $A,B,\alpha,\beta$ ... run の loss から fit される係数。

**この論文での役割**: 全 run の final losses を統合して、明示的な efficient frontier を導くためのモデル。Appendix `app:parametric` では empirical fit として $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ が報告される。

$$
\min_{A, B, E, \alpha, \beta}\quad \sum_{\text{Runs }i} \text{Huber}_\delta \Big(\log \hat L(N_i, D_i) - \log L_i\Big).
$$

**式の意味**: Approach 3 の model fitting objective。予測 log loss と観測 log loss の差を Huber loss で測り、$(A,B,E,\alpha,\beta)$ を推定する（main.tex, Eq. `eq:huber`）。

**記号の定義**:
- $i$ ... 各 training run。
- $N_i,D_i,L_i$ ... run $i$ の model parameters、training tokens、観測 loss。
- $\text{Huber}_\delta$ ... 外れ値に比較的頑健な loss。TeX では $\delta=10^{-3}$ を使う。
- L-BFGS ... この最適化に使われる algorithm。local minima 対策として grid of initialisations から best fit を選ぶ。

**この論文での役割**: Approach 3 の推定が、単に手で曲線を引いたものではなく、明示的な objective と optimization procedure による fit であることを示す。小 compute regime への過剰適合を避けるため Huber loss が重要だと著者は述べる。

$$
N_{opt}(C) = G {\left(\frac{C}{6}\right)}^{a}, \quad
D_{opt}(C) = G^{-1} {\left(\frac{C}{6}\right)}^{b}, \quad
G = {\left(\frac{\alpha A}{\beta B} \right)}^{\frac{1}{\alpha + \beta}},\quad
a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha + \beta}.
$$

**式の意味**: $\hat L(N,D)$ と $\text{FLOPs}(N,D) \approx 6ND$ から得られる efficient computational frontier の閉形式（main.tex, Approach 3）。

**記号の定義**:
- $G$ ... $A,B,\alpha,\beta$ から決まる比例係数。
- $a,b$ ... optimal parameter count と token count の scaling exponent。
- $C/6$ ... FLOPs 近似 $C \approx 6ND$ による項。
- $\alpha,\beta,A,B$ ... parametric loss の fit parameters。

**この論文での役割**: Approach 3 が $a=0.46$, $b=0.54$ を出す直接の理由を与える。$\alpha=0.34$, $\beta=0.28$ を入れると、data 側を model 側よりやや速く増やす予測になる。

### 実装 / アルゴリズム上の要点

- **Approach 1: training curve envelope**。70M から 10B parameters の範囲で、各 model size を 4 種類の cosine cycle length / training horizon で訓練する。各 run の training loss curve を smoothing / interpolation し、1500 個の logarithmically spaced FLOP values で最小 loss を達成する model size と token count を取り、$N_{opt} \propto C^a$, $D_{opt} \propto C^b$ を fit する。結果は $a=0.50$, $b=0.50$（Figure `fig:approach1`, Table `tab:comparison`）。
- **Approach 2: IsoFLOP profiles**。$6 \times 10^{18}$ から $3 \times 10^{21}$ FLOPs までの 9 種類の fixed FLOP budgets で、model size を変えて final loss を見る。各 IsoFLOP curve の loss が最小になる model size を parabola fit で推定し、そこから scaling exponent を fit する。結果は $a=0.49$, $b=0.51$（Figure `fig:isoflop`, Table `tab:comparison`）。
- **Approach 3: parametric loss fitting**。Approach 1 と 2 の final losses を $\hat L(N,D)$ に fit し、Huber loss + L-BFGS で parameters を求める。結果は $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$、scaling exponent は $a=0.46$, $b=0.54$（main.tex, Appendix `app:parametric`）。
- **Learning rate schedule の扱い**。著者は cosine cycle length を target number of training steps に合わせるべきだとする。Figure `fig:cosine` では、cosine cycle length が training steps より 25% を超えて長いと性能低下が明確になると説明される。
- **FLOPs の計算**。Embedding、attention、dense block、final logits、backward pass まで含める。Appendix `sec:flops` では、一般的な近似 $C=6DN$ との比を比較し、差は小さく分析に影響しないと述べる。
- **Chinchilla の訓練**。Gopher compute budget での optimal model size は 40B から 70B parameters の間と予測されるため、著者は dataset と computational efficiency を考慮して range の大きい側である 70B を採用し、1.4T tokens で訓練する（main.tex, Section `Chinchilla`）。

## 実験・結果

- **データセット / ベンチマーク**: 訓練には MassiveText を使い、Chinchilla では subset distribution を Gopher から少し変える。Table `tab:data_makeup` では MassiveWeb 45% (Gopher 48%)、Books 30% (27%)、C4 10% (10%)、News 10% (10%)、GitHub 4% (3%)、Wikipedia 1% (2%)。1.4T tokens では MassiveWeb が 1.24 epochs、Wikipedia が 3.40 epochs 使われる。評価は Language Modelling 20 tasks、Reading Comprehension 3 tasks、Question Answering 3 tasks、Common Sense 5 tasks、MMLU 57 tasks、BIG-bench 62 tasks（Table `tab:task_summary`）。
- **比較対象 / baseline**: 主比較は同じ compute budget の Gopher 280B。ほかに GPT-3 175B、Jurassic-1 178B、MT-NLG 530B、いくつかの supervised / open-book SOTA、人間評価者・人間 expert・human forecasters が表に出る。Scaling exponent の比較対象は Kaplan et al. (2020)。
- **指標**: pre-training / language modelling では loss、perplexity、bits-per-byte (bpb)。Downstream では accuracy、QA では exact match accuracy として扱われる値、toxicity では PerspectiveAPI score、Winogender では pronoun coreference resolution の accuracy。
- **Scaling の主結果**: Table `tab:comparison` では、Approach 1 が $a=0.50$, $b=0.50$、Approach 2 が $a=0.49$, $b=0.51$、Approach 3 が $a=0.46$, $b=0.54$。Kaplan et al. の $0.73/0.27$ と対照的に、model size と training data をほぼ同じ割合で増やす結論になる。C4 と GitHub code の IsoFLOP analysis でも、C4 は $a=0.50,b=0.50$、GitHub は $a=0.53,b=0.47$（Table `tab:comparison_c4_github`）。
- **Compute frontier の予測**: Table `tab:compute` では、67B parameters に $5.76\times10^{23}$ FLOPs / 1.5T tokens、175B に $3.85\times10^{24}$ FLOPs / 3.7T tokens、280B に $9.90\times10^{24}$ FLOPs / 5.9T tokens、1T に $1.27\times10^{26}$ FLOPs / 21.2T tokens を挙げる。本文では別途、175B には $4.41\times10^{24}$ FLOPs と 4.2T tokens 超、280B Gopher-like model には約 $10^{25}$ FLOPs と 6.8T tokens と書かれており、Table の数値とは少し異なる表現が併存する。
- **Chinchilla の構成**: Chinchilla 70B は Gopher 280B と同じ FLOPs で訓練されるが、parameters は 1/4、training tokens は 1.4T。Table `tab:arch` では、Chinchilla は 80 layers、64 heads、key/value size 128、$d_{\text{model}}=8192$、max LR $1\times10^{-4}$、batch size 1.5M $\rightarrow$ 3M tokens。Gopher は 80 layers、128 heads、$d_{\text{model}}=16384$、max LR $4\times10^{-5}$、batch size 3M $\rightarrow$ 6M tokens。訓練実装では forward / backward を `bfloat16`、distributed optimiser state の weight copy を `float32` とする。
- **MMLU**: Table `tab:mmlu` では Chinchilla 5-shot accuracy が 67.6%、Gopher 60.0%、GPT-3 43.9%、random 25.0%、average human rater 34.5%、average human expert 89.8%。June 2023 forecast は 63.4%。Figure `fig:mmlu` では Chinchilla が 57 tasks 中 51 tasks で Gopher を上回り、2 tasks で同点、4 tasks（`college_mathematics`, `econometrics`, `moral_scenarios`, `formal_logic`）で下回る。
- **BIG-bench**: Chinchilla は 62 tasks の平均 accuracy 65.1%、Gopher は 54.4%。差は 10.7%。Gopher より悪いのは `crash_blossom`, `dark_humor_detection`, `mathematical_induction`, `logical_args` の 4 tasks（main.tex, Section `BIG-bench`, Table `tab:bigbench`）。
- **Reading comprehension**: LAMBADA zero-shot は Chinchilla 77.4、Gopher 74.5、GPT-3 76.2、MT-NLG 76.6。RACE-m few-shot は 86.8 vs Gopher 75.1、RACE-h few-shot は 82.3 vs Gopher 71.6。TeX は RACE-h/m について GPT-3 と MT-NLG は prompt format が違うため Gopher / Chinchilla と直接比較できないと注記する（Table `tab:reading`）。
- **Common sense / TruthfulQA**: HellaSWAG 80.8%、PIQA 81.8%、Winogrande 74.9%、SIQA 51.3%、BoolQ 83.7%。PIQA では MT-NLG 82.0% が Chinchilla 81.8% をわずかに上回るが、その他では Chinchilla が Gopher と GPT-3 を上回り、MT-NLG もほぼ上回る（Table `tab:commonsense`）。TruthfulQA は 0-shot 43.6%、5-shot 58.5%、10-shot 66.7%。Gopher は 0-shot 29.5%、10-shot 43.7%。
- **Closed-book QA**: Natural Questions (dev) は Chinchilla 0-shot 16.6%、5-shot 31.5%、64-shot 35.5%。Gopher は 10.1%、24.5%、28.2%。TriviaQA unfiltered は 0-shot 67.0%、5-shot 73.2%、64-shot 72.3%。TriviaQA filtered は 0-shot 55.4%、5-shot 64.1%、64-shot 64.6%（Table `tab:QA`）。
- **Language modelling**: The Pile の全 subset で Chinchilla は Gopher より低い bpb を示す。Jurassic-1 は `dm_mathematics` と `ubuntu_irc` の 2 subsets で Chinchilla より良い（Table `tab:pile_nums`）。WikiText103 perplexity は Chinchilla 7.16、Gopher 7.75。
- **Bias / toxicity**: Winogender では全体 78.3% vs Gopher 71.4%、male 71.2% vs 68.0%、female 79.6% vs 71.3%、neutral 84.2% vs 75.0%。gotcha examples では female gotcha が 76.7% vs 66.7% で +10%。著者は改善が pronoun groups 間で uneven であり、bias を示唆すると述べる（Table `tab:fairness`）。Toxicity は 25,000 unprompted samples の PerspectiveAPI score で、mean / median が Chinchilla 0.087 / 0.066、Gopher 0.081 / 0.064、95th percentile が 0.238 vs 0.230。差は negligible とされる。
- **著者が主張する貢献**: 3 つの異なる empirical approach が近い scaling exponent を出すこと、Kaplan et al. との差を learning rate schedule / training horizon の扱いから説明すること、Gopher と同じ compute で小さく長く訓練した Chinchilla が広い評価で上回ること、そして小さいモデルであるため inference と downstream fine-tuning の compute / memory footprint が小さくなること。

## 妥当性と限界

- **この主張を支える根拠**: 最も強い根拠は、Approach 1 / 2 / 3 が異なるデータの使い方と仮定を持つにもかかわらず、いずれも $a,b$ をほぼ 0.5 付近に推定している点。また、Gopher と同じ FLOPs で訓練した Chinchilla が、多くの downstream tasks で Gopher を上回るため、単なる loss fitting ではなく大規模実物モデルで予測を検証している。
- **この主張を支える評価設計**: Chinchilla と Gopher はどちらも DeepMind の MassiveText 系 dataset と類似 architecture を使い、同じ compute budget で比較されるため、「同じ FLOPs なら大きいモデルを短く訓練するより、小さいモデルを長く訓練する方が良い」という主張を直接試している。評価も Gopher 論文の task subset を大きく踏襲する。
- **著者が認めている limitations / future work**: 大規模で直接比較できる訓練 run は Chinchilla と Gopher の 2 点だけで、中間スケールの追加検証はない。Efficient frontier を power-law と仮定しているが、高 compute budgets で $\log(N_{opt})$ に concavity が見られ、large models の optimal size をまだ過大評価している可能性がある。Scaling analysis の training runs はすべて 1 epoch 未満であり、multiple epoch regime は future work とされる（main.tex, Section `Discussion & Conclusion`, Appendix `Curvature of the FLOP-loss frontier`）。
- **データと leakage への注意**: Chinchilla は Gopher より 4 倍多い data で訓練されるため、The Pile や WikiText103 のような language modelling benchmarks では train/test leakage が結果を人工的に良くする可能性がある。著者はそのため MMLU、BIG-bench、closed-book QA、common sense など、leakage が less of a concern とする task をより重視すると明記する（main.tex, Section `Language modelling`）。
- **読者として注意すべき点**: Chinchilla と Gopher は model size / token count 以外にも、AdamW vs Adam、NFKC normalization なしの SentencePiece、data mixture、`bfloat16` / `float32` precision handling などの違いがある。Appendix `Other differences between Chinchilla and Gopher` では AdamW などの ablation が示されるが、downstream task 全体の改善が純粋に $N,D$ の配分だけに由来するとは切り分けきれない。
- **社会的・安全性の限界**: Model card では、Chinchilla は English data で訓練され、公開予定はなく、primary intended users は DeepMind researchers とされる。Downstream applications には additional safety and fairness mitigations が必要と書かれる。Intersectional biases は調べておらず、training data には toxic / biased content や private information が含まれる可能性がある。
- **追加で確認したい実験 / 疑問**: 175B など中間スケールで compute-optimal frontier を実物訓練で検証する実験、decontamination した evaluation で MMLU / BIG-bench の改善がどの程度残るか、multi-epoch で同じ scaling law が成り立つか、MoE や retrieval-augmented models でも同じ $N$ と $D$ の trade-off が出るか、AdamW / tokenizer / data mixture の差を大規模で分離する ablation があると主張の範囲がより明確になる。

## 用語メモ

- **compute-optimal training**: この論文では、固定された FLOPs 予算で final pre-training loss を最小にする model size と training tokens の配分を選ぶこと。
- **undertrained**: model size に対して training tokens が少なく、同じ compute でより小さいモデルをより長く訓練した方が良い状態。著者は近年の大規模 LM の多くをこの状態だと見る。
- **FLOPs budget $C$**: 訓練に使える総計算量。実務上の accelerator 数と訓練時間に対応するものとして導入される。
- **$N$ / $D$**: $N$ は model parameters、$D$ は training tokens。論文の主張は、この 2 つを compute の増加に対してほぼ同じ割合で増やすべきという点にある。
- **IsoFLOP profile**: FLOPs を固定し、その予算内で model size を変えたときの final loss の曲線。谷の位置が、その FLOP budget での optimal model size を与える。
- **training curve envelope**: 複数の model size / training horizon の training curves を重ね、各 FLOPs で最小 loss を達成する点を集めた frontier。
- **cosine cycle length**: learning rate を cosine schedule で下げる horizon。著者は target training steps に合わせるべきだとし、25% を超える過大な overshoot は性能を悪化させるとする。
- **MassiveText / MassiveWeb**: Gopher と Chinchilla が使う訓練 corpus family と、その web subset。Chinchilla は同じ dataset family を使うが sampling proportion は少し異なる。
- **Chinchilla**: この論文で訓練された 70B parameters の autoregressive Transformer LM。Gopher と同じ compute budget で、1.4T tokens を使って訓練される。
- **Gopher**: 主 baseline の 280B parameters LM。Chinchilla と同じ compute budget の比較対象として使われる。
- **MMLU**: 57 の academic subjects からなる exam-like benchmark。Chinchilla の中心的な downstream evidence の 1 つ。
- **BIG-bench**: 62 tasks の subset で評価される benchmark。Chinchilla は平均 65.1%、Gopher は 54.4%。
- **bits-per-byte (bpb)**: language modelling benchmark で使われる指標。小さいほど圧縮・予測が良い。The Pile の表では Chinchilla が Gopher より低い。
- **train/test leakage**: 訓練データに評価データが混入している可能性。Chinchilla は Gopher より多い data を見るため、language modelling benchmarks では特に注意が必要だと著者が述べる。
- **PerspectiveAPI toxicity score**: generated samples の toxicity を自動 classifier で測る指標。この論文では unprompted 25,000 samples の distribution を Gopher と比較する。

## 読む順番の提案

- まず `main.tex` の `Introduction` を読み、Eq. `eq:model` と Table `tab:llms` で、論文が「固定 FLOPs 下の $N,D$ 配分」を問うていることを確認する。正規ノートの `Summary（著者の主張）` の問題設定に対応する。
- 次に `Related Work` を読み、Kaplan et al. (2020) との差が model size だけでなく learning rate schedule / training horizon の扱いにあることを押さえる。正規ノートの Kaplan+ への反論の箇条書きにつながる。
- その後 `Estimating the optimal parameter/training tokens allocation` を、Approach 1 → Approach 2 → Approach 3 の順に読む。特に Figure `fig:approach1`, Figure `fig:isoflop`, Eq. `eq:decompose`, Eq. `eq:huber`, Table `tab:comparison` が中心。
- Chinchilla の実物検証は `Chinchilla` セクションで読む。Table `tab:arch` と Table `tab:task_summary` を見て、Gopher との比較条件と評価範囲を確認する。正規ノートの Chinchilla architecture / data mixture の記述に対応する。
- 結果は MMLU と BIG-bench を先に読む。Table `tab:mmlu`, Figure `fig:mmlu`, Table `tab:bigbench` が著者の主要な downstream evidence。次に reading comprehension、common sense、closed-book QA、language modelling を読む。
- 最後に `Discussion & Conclusion` と Appendix の `Training dataset`, `Optimal cosine cycle length`, `Consistency of scaling results across datasets`, `Curvature of the FLOP-loss frontier`, `Other differences between Chinchilla and Gopher`, `Model Card` を読む。正規ノートの `Critical Thoughts（評価・疑問）` の根拠になる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2203.15556v1/`
- 正規ノート: `notes/arXiv-2203.15556v1.md`
