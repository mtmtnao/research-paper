# DeepSeek-V3 Technical Report（大規模 MoE LLM の効率的訓練・推論・post-training 技術報告）

- arXiv: https://arxiv.org/abs/2412.19437
- 一次ソース: ../papers/arXiv-2412.19437v2/
- 正規ノート: ../notes/arXiv-2412.19437v2.md

---

## 一言で言うと

DeepSeek-V3 は、671B total parameters / 37B activated parameters の MoE 言語モデルを、MLA、DeepSeekMoE、auxiliary-loss-free load balancing、MTP、FP8 mixed precision training、DualPipe で効率よく訓練する技術報告である。著者は、14.8T tokens の pre-training と SFT/RL 後、open-source models を上回り、GPT-4o / Claude-3.5-Sonnet などの closed-source models と比較可能な性能を、2.788M H800 GPU hours で達成したと主張する（`main.tex` abstract, Table `tab:cost`）。

## 何を議論する論文か

- **問題設定**: open-source LLM を frontier closed-source models に近づけるために、モデルを 671B 規模へ拡大しながら、各トークンで動かす計算量、訓練通信、数値精度、post-training の reasoning 能力移植を同時に扱う。
- **対象範囲 / 仮定**: Transformer ベースの MoE language model が対象である。訓練環境は 2048 NVIDIA H800 GPUs、8 GPUs/node、node 内 NVLink/NVSwitch、node 間 IB であり、訓練は 16-way PP、64-way EP spanning 8 nodes、ZeRO-1 DP を前提に設計されている（`main.tex` §Infrastructures）。
- **既存研究との差分**: DeepSeek-V2 で検証済みの MLA と DeepSeekMoE を引き継ぎ、(1) pure auxiliary loss に頼らない load balancing、(2) sequential な Multi-Token Prediction、(3) extremely large-scale model での FP8 mixed precision training、(4) DualPipe による computation-communication overlap、(5) DeepSeek-R1 からの reasoning distillation を組み合わせる。
- **この論文で答えたい問い**: 大規模 MoE LLM を、性能を保ったまま低コスト・安定・高効率に pre-train / context-extend / post-train できるか。その結果として、open-source model が closed-source frontier に近い benchmark performance を出せるか。

## 背景と前提

- Transformer の attention と FFN が基本構造である。DeepSeek-V3 は attention に MLA、FFN の大部分に DeepSeekMoE を使う。最初の 3 層以外の FFN を MoE layers に置き換え、各 MoE layer は 1 shared expert + 256 routed experts を持つ（`main.tex` §Hyper-Parameters）。
- MoE では、各 token が全 expert を使うのではなく routed experts の一部だけを使う。DeepSeek-V3 では routed experts のうち 8 experts が activated され、各 token は最大 4 nodes にだけ送られる（`main.tex` §Model Hyper-Parameters, §Node-Limited Routing）。
- load balancing は MoE の効率と安定性の前提である。従来法は Switch Transformer / GShard などの auxiliary loss に依存するが、著者は大きな auxiliary loss が model performance を損なうとし、bias term による auxiliary-loss-free routing を採る（`main.tex` §Auxiliary-Loss-Free Load Balancing）。
- FP8 training は計算・メモリ効率のための低精度訓練である。ただし outliers、dynamic range、accumulation precision が問題になるため、DeepSeek-V3 は activation を `1x128` tile、weight を `128x128` block で scale し、$N_C=128$ 間隔で CUDA Cores へ昇格して FP32 accumulation する（`content/fp8.tex`）。
- post-training では SFT と RL を行う。RL は DeepSeekMath 由来の GRPO を使い、critic model を置かず group scores から baseline を推定する。reasoning data では DeepSeek-R1 series の出力を使うが、overthinking、poor formatting、excessive length を抑えることも目標に含める（`main.tex` §Post-Training）。
- `main.bbl` 上の近い参照は、Transformer、DeepSeek-V2、DeepSeekMoE、Auxiliary-loss-free load balancing、Meta MTP、EAGLE、FP8-LM、microscaling formats、YaRN、DeepSeekMath/GRPO、RewardBench、Constitutional AI である。

## 提案手法

### コアアイデア

DeepSeek-V3 は「全体として巨大だが、各 token では一部だけを動かす」MoE LLM である。671B total parameters のうち、各 token で activated されるのは 37B parameters であり、これにより dense 405B model のような全パラメータ計算を避ける。著者は、MLA による KV cache 削減、DeepSeekMoE による sparse activation、bias-based load balancing による expert load 制御、MTP による training signal の densification、FP8/DualPipe による訓練効率化を、単独の工夫ではなく system co-design として提示している。

手法の中心は次の 5 点である。

- **MLA**: keys/values と queries を低ランク latent vector に圧縮し、generation 時に cache する量を減らす。TeX は「only the blue-boxed vectors ... need to be cached」と説明する（`main.tex` §Multi-Head Latent Attention）。
- **DeepSeekMoE + auxiliary-loss-free load balancing**: routed expert の選択には $s_{i,t}+b_i$ を使うが、FFN 出力に掛ける gating value は元の affinity $s_{i,t}$ から作る。routing と weighting を分離する点が重要である。
- **MTP**: next token だけでなく additional token も予測する。Meta MTP と異なり、independent output heads で parallel に予測するのではなく、sequential modules で complete causal chain を保つ（`main.tex` §Multi-Token Prediction）。
- **FP8 mixed precision training**: Fprop/Dgrad/Wgrad の GEMM を FP8 で実行し、embedding、output head、MoE gating、normalization、attention は original precision に残す。FP8 化される tensors の format は E4M3 に統一する。
- **DualPipe と all-to-all kernels**: cross-node expert parallelism の computation-to-communication ratio 約 1:1 というボトルネックに対し、forward/backward chunk を attention、all-to-all dispatch、MLP、all-to-all combine に分解して overlap する。

### 重要な定義・数式

$$
\begin{aligned}
\mathbf{c}_{t}^{KV} &= W^{DKV} \mathbf{h}_{t}, \\
\mathbf{k}_{t}^{C} &= W^{UK} \mathbf{c}_{t}^{KV}, \\
\mathbf{k}_{t}^{R} &= \operatorname{RoPE}({W^{KR}} \mathbf{h}_{t}), \\
\mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
\mathbf{v}_{t}^{C} &= W^{UV} \mathbf{c}_{t}^{KV}.
\end{aligned}
$$

**式の意味**: MLA の key/value 側の低ランク圧縮である。$\mathbf{h}_t$ から compressed latent vector $\mathbf{c}_{t}^{KV}$ を作り、そこから compressed key/value と RoPE を持つ decoupled key を構成する（`main.tex` §Multi-Head Latent Attention）。

**記号の定義**:
- $\mathbf{h}_t \in \mathbb{R}^d$ ... attention layer への $t$ 番目 token の入力表現
- $\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}$ ... keys/values 用の compressed latent vector
- $W^{DKV}$ ... KV の down-projection matrix
- $\mathbf{k}_{t}^{C}$ ... all attention heads の compressed key を連結した表現
- $\mathbf{k}_{t,i}^{C}$ ... $i$ 番目 attention head の compressed key 成分
- $\mathbf{k}_{t,i}$ ... $\mathbf{k}_{t,i}^{C}$ と RoPE 付き key を連結した $i$ 番目 head の key
- $\mathbf{v}_{t}^{C}$ ... all attention heads の compressed value を連結した表現
- $W^{UK}, W^{UV}$ ... keys/values の up-projection matrices
- $W^{KR}$ ... RoPE を運ぶ decoupled key を作る matrix
- $\mathbf{k}_{t}^{R}$ ... RoPE を運ぶ decoupled key
- $\operatorname{RoPE}(\cdot)$ ... Rotary Positional Embedding を適用する operation
- $[\cdot;\cdot]$ ... concatenation

**この論文での役割**: generation 時には $\mathbf{c}_{t}^{KV}$ と $\mathbf{k}_{t}^{R}$ だけを cache すればよい、という KV cache 削減の根拠になる。DeepSeek-V3 が efficient inference を主張する基本部品である。

$$
\begin{aligned}
\mathbf{h}_{t}^{\prime} &=
\mathbf{u}_{t}
+ \sum_{i=1}^{N_{s}} {\operatorname{FFN}^{(s)}_{i}\left( \mathbf{u}_{t} \right)}
+ \sum_{i=1}^{N_r} {g_{i,t} \operatorname{FFN}^{(r)}_{i}\left( \mathbf{u}_{t} \right)}, \\
g_{i,t} &= \frac{g^{\prime}_{i,t}}{\sum_{j=1}^{N_r} g^{\prime}_{j,t}}, \\
g^{\prime}_{i,t} &=
\begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk}(\{s_{j,t}\mid 1 \leq j \leq N_r\}, K_r), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} &= \operatorname{Sigmoid}({\mathbf{u}_{t}}^{T}\mathbf{e}_{i}).
\end{aligned}
$$

**式の意味**: DeepSeekMoE の FFN 出力である。shared experts は常に足され、routed experts は affinity score の top-$K_r$ に入ったものだけが gating value $g_{i,t}$ で重み付けされる（`main.tex` §Basic Architecture of DeepSeekMoE）。

**記号の定義**:
- $\mathbf{h}_{t}^{\prime}$ ... DeepSeekMoE FFN block の出力
- $\mathbf{u}_t$ ... FFN input
- $\operatorname{FFN}^{(s)}_i$ ... $i$ 番目の shared expert
- $\operatorname{FFN}^{(r)}_i$ ... $i$ 番目の routed expert
- $N_s, N_r$ ... shared experts / routed experts の数
- $K_r$ ... activated routed experts の数。DeepSeek-V3 では 8
- $g_{i,t}$ ... $i$ 番目 routed expert の gating value
- $g^{\prime}_{i,t}$ ... top-$K_r$ selection 後、normalization 前の expert weight
- $s_{i,t}$ ... token-to-expert affinity
- $\mathbf{e}_i$ ... $i$ 番目 routed expert の centroid vector
- $\operatorname{Topk}(\cdot,K_r)$ ... $K_r$ 個の最大 score の集合

**この論文での役割**: 671B total parameters を持ちながら、各 token で 37B activated parameters に抑える sparse computation の定義である。以後の load balancing は、この top-$K_r$ routing をどう安定させるかという問題になる。

$$
g^{\prime}_{i,t} =
\begin{cases}
s_{i,t}, & s_{i,t} + b_i \in \operatorname{Topk}(\{s_{j,t} + b_j \mid 1 \leq j \leq N_r\}, K_r), \\
0, & \text{otherwise}.
\end{cases}
$$

**式の意味**: auxiliary-loss-free load balancing の routing rule である。top-$K_r$ を決めるときだけ expert ごとの bias $b_i$ を affinity に足すが、選ばれた expert の gating value は元の $s_{i,t}$ から作る。

**記号の定義**:
- $b_i$ ... $i$ 番目 expert の bias term
- $s_{i,t}$ ... token $t$ と expert $i$ の affinity score
- $N_r$ ... routed experts の数
- $K_r$ ... activated routed experts の数
- $\operatorname{Topk}(\cdot,K_r)$ ... $K_r$ 個の最大 score の集合
- $g^{\prime}_{i,t}$ ... normalization 前の routed expert weight

**この論文での役割**: load balance のための補助損失を主役にせず、expert load を batch step ごとの bias update で制御する核である。本文では、各 training step の終わりに overloaded expert は $b_i$ を $\gamma$ だけ下げ、underloaded expert は $\gamma$ だけ上げると説明される。$\gamma=0.001$ は最初の 14.3T tokens、残り 500B tokens では $\gamma=0.0$ である（`main.tex` §Training Hyper-Parameters）。

$$
\begin{aligned}
\mathcal{L}_{\mathrm{Bal}} &= \alpha \sum_{i=1}^{N_r}{f_i P_i}, \\
f_i &= \frac{N_r}{K_r T} \sum_{t=1}^{T} \mathds{1}\left(s_{i,t} \in \operatorname{Topk}(\{s_{j,t}\mid 1 \leq j \leq N_r\}, K_r)\right), \\
s^{\prime}_{i,t} &= \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}}, \\
P_i &= \frac{1}{T}\sum_{t=1}^{T}{s^{\prime}_{i,t}}.
\end{aligned}
$$

**式の意味**: single sequence 内で極端な expert imbalance が起きるのを抑える complementary sequence-wise balance loss である。DeepSeek-V3 は主に auxiliary-loss-free strategy に頼るが、sequence-wise loss も非常に小さな重みで併用する。

**記号の定義**:
- $\mathcal{L}_{\mathrm{Bal}}$ ... sequence-wise balance loss
- $\alpha$ ... balance factor。DeepSeek-V3 では 0.0001
- $f_i$ ... sequence 内で expert $i$ が top-$K_r$ に入る頻度を正規化した量
- $N_r$ ... routed experts の数
- $K_r$ ... activated routed experts の数
- $s_{i,t}$ ... token $t$ と expert $i$ の affinity score
- $s^{\prime}_{i,t}$ ... sequence-wise loss で使う normalized affinity
- $\mathds{1}(\cdot)$ ... 条件が真なら 1、偽なら 0 の indicator function
- $P_i$ ... sequence 内での normalized affinity の平均
- $T$ ... sequence length

**この論文での役割**: 論文の load balancing は「完全に balance loss が無い」わけではない、という読み違いを防ぐ。主な load balance は bias update だが、single sequence の extreme imbalance には小さな $\alpha$ の補助項を使う。

$$
\begin{aligned}
\mathcal{L}_{\text{MTP}}^{k}
&= \operatorname{CrossEntropy}(P_{2+k:T+1}^{k}, t_{2+k:T+1}) \\
&= -\frac{1}{T}\sum_{i=2+k}^{T+1}\log P_i^k[t_i], \\
\mathcal{L}_{\text{MTP}}
&= \frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{\text{MTP}}^k.
\end{aligned}
$$

**式の意味**: $k$ 番目の MTP module が追加で予測する future token に対する cross-entropy loss と、全 depth の平均に重み $\lambda$ を掛けた MTP loss である（`main.tex` §MTP Training Objective）。

**記号の定義**:
- $\mathcal{L}_{\text{MTP}}^k$ ... $k$ 番目 MTP module の cross-entropy loss
- $\mathcal{L}_{\text{MTP}}$ ... depth 全体を平均して重み付けした追加 training objective
- $k$ ... MTP prediction depth の index
- $D$ ... MTP depth。DeepSeek-V3 では $D=1$
- $t_i$ ... $i$ 番目位置の ground-truth token
- $P_{2+k:T+1}^{k}$ ... $k$ 番目 MTP module が予測する token 確率列
- $P_i^k[t_i]$ ... $k$ 番目 MTP module が ground-truth token $t_i$ に割り当てた確率
- $T$ ... input sequence length
- $\lambda$ ... MTP loss weight。最初の 10T tokens では 0.3、残り 4.8T tokens では 0.1

**この論文での役割**: MTP が単なる inference trick ではなく、pre-training objective として入っていることを示す。inference 時には MTP modules を discard できるため、ablation では同じ inference cost で性能比較されている（Table `tab:ablation_nextn`）。

$$
\begin{split}
\mathcal{J}_{GRPO}(\theta)
&= \mathbb{E}[q \sim P(Q), \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(O|q)] \\
&\frac{1}{G}\sum_{i=1}^{G}
\left(
\min\left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i,
\operatorname{clip}\left(
\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},1-\epsilon,1+\epsilon
\right)A_i
\right)
- \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref})
\right),
\end{split}
$$

**式の意味**: GRPO の policy optimization objective である。clipped ratio と KL penalty を使うが、critic model は使わず、同じ question から sampled outputs の group reward で advantage を作る（`main.tex` Equation `eq:GRPO-obj`）。

**記号の定義**:
- $q$ ... question / prompt
- $P(Q)$ ... question distribution
- $O$ ... output space
- $\{o_i\}_{i=1}^{G}$ ... old policy $\pi_{\theta_{old}}$ から sampling した $G$ 個の outputs
- $G$ ... group 内で sampling する output 数
- $\pi_\theta$ ... 更新対象の policy model
- $\pi_{\theta_{old}}$ ... sampling に使う old policy model
- $\pi_{ref}$ ... reference model
- $A_i$ ... group rewards から計算される advantage
- $\epsilon, \beta$ ... clipping と KL penalty の hyper-parameters
- $\mathbb{D}_{KL}(\pi_\theta || \pi_{ref})$ ... policy と reference model の KL penalty term

**この論文での役割**: post-training の RL 部分で、DeepSeek-V3 を human preferences と benchmark performance に合わせる最適化手法である。TeX では、$A_i=(r_i-\operatorname{mean}(\{r_1,\ldots,r_G\}))/\operatorname{std}(\{r_1,\ldots,r_G\})$ と定義される。

### 実装 / アルゴリズム上の要点

1. **Model hyper-parameters**: 61 Transformer layers、hidden dimension 7168、initialization std 0.006。MLA は $n_h=128$、$d_h=128$、$d_c=512$、$d_c'=1536$、$d_h^R=64$。各 expert の intermediate hidden dimension は 2048。
2. **MoE routing**: first three layers 以外の FFNs を MoE layers に置き換える。各 MoE layer は 1 shared expert + 256 routed experts。各 token は 8 routed experts を activate し、最大 4 nodes に送られる。no token-dropping を training/inference の両方で主張する。
3. **Pre-training data and schedule**: corpus は 14.8T tokens。FIM は Prefix-Suffix-Middle format で rate 0.1。tokenizer は Byte-level BPE、128K vocabulary。AdamW は $\beta_1=0.9,\beta_2=0.95,\mathrm{weight\_decay}=0.1$、gradient clipping norm 1.0。
4. **Learning rate / batch schedule**: 2K steps で $0 \to 2.2\times10^{-4}$、10T tokens まで一定、その後 4.3T tokens で $2.2\times10^{-5}$ まで cosine decay、最後 500B tokens は 333B tokens を $2.2\times10^{-5}$、残り 167B tokens を $7.3\times10^{-6}$。batch size は最初の 469B tokens で 3072 から 15360 へ増やす。
5. **Long context extension**: YaRN を decoupled shared key $\mathbf{k}_t^R$ に適用し、1000 steps ずつの 2 phases で 4K から 32K、さらに 128K へ拡張する。$s=40,\alpha=1,\beta=32,\sqrt{t}=0.1\ln{s}+1$、learning rate は $7.3\times10^{-6}$。
6. **Training framework**: HAI-LLM framework、2048 H800 GPUs、16-way PP、64-way EP spanning 8 nodes、ZeRO-1 DP。Tensor Parallelism は training では使わない、と本文で述べる。
7. **DualPipe**: forward/backward chunks を `attention`、`all-to-all dispatch`、`MLP`、`all-to-all combine` に分け、backward は input backward と weight backward に分ける。Table `tab:dualpipe-bubble` は 1F1B / ZB1P / DualPipe の bubble と memory usage を比較する。
8. **All-to-all communication**: token を最大 4 nodes に制限し、IB は 50 GB/s、NVLink は 160 GB/s と説明される。20 SMs を 10 communication channels に分け、warp specialization、customized PTX、communication chunk size auto-tuning を使う。
9. **FP8 details**: Linear の Fprop/Dgrad/Wgrad GEMMs を FP8 で行う。activation は `1x128` tile、weight は `128x128` block。FP8 化される tensors では E4M3 を使い、$N_C=128$ ごとに CUDA Cores へ partial results を移して FP32 accumulation する。optimizer の first/second moments は BF16、master weights と gradients は FP32。
10. **Post-training pipeline**: SFT dataset は 1.5M instances。reasoning data は internal DeepSeek-R1 model を利用し、domain-specific expert model を SFT+RL で作る。SFT sample は `<problem, original response>` と `<system prompt, problem, R1 response>` の 2 形式。最終 SFT data は rejection sampling で curated される。Non-reasoning data は DeepSeek-V2.5 responses と human annotators を使う。
11. **SFT/RL settings**: DeepSeek-V3-Base を SFT dataset で 2 epochs fine-tune し、cosine decay learning rate は $5\times10^{-6}$ から $1\times10^{-6}$。sample masking により packed samples を互いに見えないようにする。RL は rule-based RM と model-based RM を併用する。

## 実験・結果

- **データセット / ベンチマーク**: base model では MMLU、MMLU-Redux、MMLU-Pro、MMMLU、C-Eval、CMMLU、HellaSwag、PIQA、ARC、BBH、TriviaQA、NaturalQuestions、RACE、DROP、C3、CMRC、CLUEWSC、WinoGrande、Pile-test、CCPM、GSM8K、MATH、MGSM、CMath、HumanEval、LiveCodeBench-Base、MBPP、CRUXEval、AGIEval を使う。chat model では追加で IFEval、FRAMES、LongBench v2、GPQA、SimpleQA、C-SimpleQA、SWE-Bench Verified、Aider、LiveCodeBench、Codeforces、CNMO 2024、AIME 2024 などを使う（`main.tex` §Evaluation Benchmarks）。
- **比較対象 / baseline**: base は DeepSeek-V2 Base、Qwen2.5 72B Base、LLaMA-3.1 405B Base。chat は DeepSeek-V2-0506、DeepSeek-V2.5-0905、Qwen2.5 72B Instruct、LLaMA-3.1 405B Instruct、Claude-Sonnet-3.5-1022、GPT-4o-0513。
- **指標**: EM、F1、Pass@1、Pass@1-COT、Correct、Acc.、Resolved、Percentile、BPB、length-controlled win rate など。Pile-test は tokenizer 差を避けるため Bits-Per-Byte (BPB) を使う。AIME/CNMO 2024 は temperature 0.7 で 16 runs average、MATH-500 は greedy decoding、全 benchmark で output length max 8192 tokens とされる。
- **主な結果**: base model の Table `tab:main` では、DeepSeek-V3-Base は BBH 87.5、MMLU 87.1、MMLU-Redux 86.2、MMLU-Pro 64.4、DROP 89.0、HumanEval 65.2、LiveCodeBench-Base 19.4、GSM8K 89.3、MATH 61.6、MGSM 79.8、CMath 90.7、C-Eval 90.1、MMMLU-non-English 79.4。caption は「best performance on most benchmarks, especially on math and code tasks」と述べる。
- **chat model の主な数値**: Table `tab:chat` では DeepSeek-V3 が MMLU 88.5、MMLU-Redux 89.1、MMLU-Pro 75.9、DROP 91.6、IF-Eval 86.1、GPQA-Diamond 59.1、SimpleQA 24.9、FRAMES 73.3、LongBench v2 48.7、HumanEval-Mul 82.6、LiveCodeBench Pass@1-COT 40.5、LiveCodeBench Pass@1 37.6、Codeforces Percentile 51.6、SWE Verified 42.0、Aider-Edit 79.7、Aider-Polyglot 49.6、AIME 2024 39.2、MATH-500 90.2、CNMO 2024 43.2、C-Eval 86.5、C-SimpleQA 64.8。
- **open-ended evaluation**: Table `tab:open` では Arena-Hard 85.5、AlpacaEval 2.0 length-controlled win rate 70.0。本文は DeepSeek-V3 を「first open-source model to surpass 85% on the Arena-Hard benchmark」と述べる。ただし同じ段落に「over 86%」という表現もあり、表の値は 85.5 である。
- **RewardBench**: Table `tab:rewardbench` では DeepSeek-V3 が Chat 96.9、Chat-Hard 79.8、Safety 87.0、Reasoning 84.3、Average 87.0。DeepSeek-V3 (maj@6) は Average 89.6。
- **Ablation**: MTP ablation の Table `tab:ablation_nextn` では large MoE 228.7B total / 20.9B activated / 540B tokens で HumanEval 44.5 -> 53.7、GSM8K 72.3 -> 74.0、MATH 38.6 -> 39.8。auxiliary-loss-free ablation の Table `tab:ablation_noaux_tc` では large MoE 228.7B total / 20.9B activated / 578B tokens で HumanEval 40.2 -> 46.3、GSM8K 70.7 -> 74.5、MATH 37.2 -> 39.6。
- **MTP inference evaluation**: §Multi-Token Prediction Evaluation は、second token prediction の acceptance rate が 85% から 90% の範囲で、1.8 times TPS を達成したと述べる。
- **DeepSeek-R1 distillation ablation**: Table `tab:distill` では DeepSeek-V2.5 Baseline と +R1 Distill を比較し、LiveCodeBench-CoT は Pass@1 31.1 / Length 718 から 37.4 / 783、MATH-500 は Pass@1 74.6 / Length 769 から 83.2 / 1510 へ改善する。著者は、performance 向上と average response length 増加の trade-off として読むべきだと述べる。
- **FP8 validation**: Appendix `app:fp8_cp_bf16` では、約 16B total model を 1.33T tokens、約 230B total model を around 0.9T tokens で比較し、high-precision accumulation と fine-grained quantization により BF16 との差の relative error が 0.25% 未満と述べる。
- **training cost / stability**: Table `tab:cost` は Pre-Training 2664K、Context Extension 119K、Post-Training 5K、Total 2788K H800 GPU Hours、総額 \$5.576M を示す。pre-training は 1T tokens あたり 180K H800 GPU hours、2048 H800 GPUs で 3.7 days。abstract と Introduction は、irrecoverable loss spikes と rollbacks が無かったと述べる。
- **著者が主張する貢献**: architecture では auxiliary-loss-free strategy と MTP objective、pre-training では FP8 mixed precision training と cross-node MoE communication bottleneck の克服、post-training では DeepSeek-R1 の long-CoT reasoning capability を standard LLM へ distill する方法、結果として economical cost で strongest open-source base/chat model class の性能を得たことを挙げる。

## 妥当性と限界

- **この主張を支える根拠**: model performance については base/chat の多数 benchmark と closed/open baselines がある。効率については Table `tab:cost`、DualPipe の Table `tab:dualpipe-bubble`、all-to-all 実装の bandwidth/SM 設計、FP8/BF16 の Appendix 実験が根拠として示される。手法要素については MTP と auxiliary-loss-free load balancing の ablation がある。
- **著者が認めている limitations / future work**: Conclusion は、efficient inference のための recommended deployment unit が大きく small-sized teams の負担になること、DeepSeek-V2 の 2 倍超の end-to-end generation speed を得たがさらに改善余地があることを limitations とする。deployment の具体値は prefilling stage が 4 nodes / 32 GPUs、decoding stage が 40 nodes / 320 GPUs である（`main.tex` §Inference and Deployment）。future directions は、architecture refinement と infinite context support、Transformer limitations の突破、training data の quantity/quality と additional training signals、reasoning length/depth の拡張、fixed benchmark への過最適化を避ける evaluation method を挙げる。
- **読者として注意すべき点**: \$5.576M は official training のみで、prior research and ablation experiments のコストを含まない、と本文が明記する。pre-training data の具体的な混合比率や filtering 詳細、constitutional AI の constitution 内容、self-rewarding voting の詳細は TeX 中には明示されていない。SimpleQA は 24.9 で GPT-4o 38.2 と Claude-Sonnet 28.4 に劣り、著者は design focus and resource allocation と Chinese knowledge への token allocation で説明している。
- **追加で確認したい実験 / 疑問**: MTP の本番設定は $D=1$ で、$D>1$ の depth 比較や Meta MTP 型 independent heads との直接比較は TeX 中には示されていない。auxiliary-loss-free と batch-wise auxiliary loss は、1B MoE で 2.258 (sequence-wise auxiliary loss) / 2.253 (auxiliary-loss-free) / 2.253 (batch-wise auxiliary loss)、3B MoE で 2.085 / 2.080 / 2.080 と validation loss が近いため、なぜ本番で bias update を選ぶのかは効率・inference imbalance 対策まで含めて読む必要がある。R1 distillation は math/code 中心と本文が述べ、他 domain で同じ効果が出るかは future work 扱いである。安全性や red-teaming の詳細は、RewardBench Safety 以外には TeX 中にまとまった実験として明示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **DeepSeek-V3** ... 671B total parameters、37B activated parameters/token の MoE language model。technical report の対象モデル。
- **MLA (Multi-head Latent Attention)** ... KV cache 削減のため、keys/values を compressed latent vector $\mathbf{c}_{t}^{KV}$ に落とし、RoPE を持つ decoupled key と合わせる attention 構造。
- **DeepSeekMoE** ... fine-grained routed experts と shared experts を持つ MoE FFN。DeepSeek-V3 では 1 shared expert + 256 routed experts、各 token 8 routed experts。
- **activated parameters** ... 各 token の forward で実際に使うパラメータ数。DeepSeek-V3 では 37B。
- **auxiliary-loss-free load balancing** ... expert load を均す主機構として auxiliary loss ではなく bias $b_i$ を routing score にだけ足す方法。完全に balance loss が無いわけではなく、sequence-wise balance loss は $\alpha=0.0001$ で残る。
- **routing / gating の分離** ... $s_{i,t}+b_i$ は top-$K_r$ expert の選択に使うが、FFN output の重みは元の $s_{i,t}$ から作る、という設計上の区別。
- **node-limited routing** ... 各 token を最大 $M=4$ nodes にだけ送る制約。cross-node all-to-all communication のコストを抑える。
- **MTP (Multi-Token Prediction)** ... next token 以外の future token も予測する training objective。DeepSeek-V3 では $D=1$ で、inference 時には MTP module を捨てられる。
- **complete causal chain** ... MTP で additional token を予測するときも未来を不正に見ないよう、sequential modules で causal relation を保つという著者の表現。
- **FP8 / E4M3** ... 8-bit floating point format。DeepSeek-V3 は prior work の E4M3/E5M2 hybrid ではなく、fine-grained quantization により FP8 化される tensors で E4M3 を使う。
- **fine-grained quantization** ... activation を `1x128` tile、weight を `128x128` block で scale する量子化。outliers の影響を小さな group ごとに吸収する狙いがある。
- **increased-precision accumulation** ... H800 の FP8 GEMM accumulation が around 14 bits に制約されるため、$N_C=128$ ごとに CUDA Cores で FP32 accumulation する方法。
- **DualPipe** ... forward/backward を双方向 pipeline scheduling し、all-to-all と PP communication を computation と overlap する training framework 上の手法。
- **EP / PP / DP / TP** ... Expert Parallelism、Pipeline Parallelism、Data Parallelism、Tensor Parallelism。DeepSeek-V3 training は 16-way PP、64-way EP、ZeRO-1 DP で、training に TP を使わない。
- **SFT** ... 1.5M instances の instruction-tuning dataset による supervised fine-tuning。packed samples は sample masking で互いに見えない。
- **GRPO** ... Group Relative Policy Optimization。critic model を置かず、group outputs の rewards から advantage を標準化する RL objective。
- **DeepSeek-R1 distillation** ... R1-generated reasoning data の accuracy を使いつつ、overthinking / poor formatting / excessive length を抑えるため、domain expert model、RL、rejection sampling を挟んで final SFT data を作る手順。
- **Self-Rewarding** ... general scenarios で hard-coded feedback が難しい場合に、constitutional AI approach と DeepSeek-V3 自身の voting evaluation results を feedback source に使う方法。constitution の具体内容は TeX 中には明示されていない。

## 読む順番の提案

- まず `main.tex` abstract と Introduction を読み、671B/37B、14.8T tokens、2.788M H800 GPU hours、no irrecoverable loss spikes/no rollbacks、contributions の範囲を押さえる。正規ノートでは `Summary（著者の主張）` の上半分につながる。
- 次に §Architecture の MLA、DeepSeekMoE、auxiliary-loss-free load balancing、MTP を読む。式としては MLA の $\mathbf{c}_{t}^{KV}$、DeepSeekMoE の $\mathbf{h}'_t$、bias routing の $g'_{i,t}$、MTP loss を見る。正規ノートでは `Takeaway` の auxiliary-loss-free と MTP に対応する。
- その次に §Infrastructures と `content/fp8.tex` を読む。DualPipe の Figure `fig:transformer-overlap` / `fig:dualpipe-schedules`、Table `tab:dualpipe-bubble`、FP8 の Figure `fig:fp8_framework` / `fig:fp8_quantization` を見る。正規ノートでは FP8 と DualPipe の項目に対応する。
- Pre-Training では Data Construction、Hyper-Parameters、Long Context Extension、Evaluation Results、Discussion の ablation tables を読む。Table `tab:main`、`tab:ablation_nextn`、`tab:ablation_noaux_tc` が主要な根拠である。
- Post-Training では SFT/RL、GRPO equation、Table `tab:chat`、Table `tab:open`、Table `tab:rewardbench`、Table `tab:distill` を読む。正規ノートでは R1 蒸留、RewardBench、SimpleQA/C-SimpleQA の議論につながる。
- 最後に Conclusion と Appendix `app:fp8_cp_bf16` / `app:fp8_blockwise` / `app:detailed_expert_load` を読む。limitations、future directions、FP8 の BF16 比較、expert specialization patterns の根拠を確認する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2412.19437v2/`
- 正規ノート: `notes/arXiv-2412.19437v2.md`
