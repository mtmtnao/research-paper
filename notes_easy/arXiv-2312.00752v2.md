# Mamba: Linear-Time Sequence Modeling with Selective State Spaces（線形時間・attention-free 系列モデルとしての選択的 SSM）

- arXiv: https://arxiv.org/abs/2312.00752
- 一次ソース: ../papers/arXiv-2312.00752v2/
- 正規ノート: ../notes/arXiv-2312.00752v2.md

---

## 一言で言うと

Transformer の self-attention は長い系列で高コストだが、従来の structured state space models (SSMs) は LTI（linear time invariant）制約のため、言語や DNA のような離散・情報密度の高いデータで必要な情報を内容に応じて選べない、という問題を扱う。Mamba は $\Delta,\B,\C$ を入力依存にする selective SSM (S6)、それを実用速度で動かす hardware-aware selective scan、attention も MLP block も持たない単純な Mamba block により、言語・DNA・音声で線形時間の汎用 sequence model backbone を主張する論文である。

## 何を議論する論文か

- **問題設定**: Foundation models の backbone である sequence model を、Transformer のように強く、かつ系列長 $L$ に対して線形にスケールするものにしたい。Transformer は self-attention により context window 内の情報を密に route できる一方、window 外を直接扱えず、training は quadratic-time、autoregressive inference は KV cache のため過去 context を保持する必要がある（`src/intro.tex`, `src/method.tex`）。
- **対象範囲 / 仮定**: 論文中の "SSM" は広い意味の状態空間モデル一般ではなく、S4 系の structured SSM / S4 models を指す（`src/background.tex`, "Throughout this entire paper..."）。入力は batch size $\mathtt{B}$、sequence length $\mathtt{L}$、channel/model dimension $\mathtt{D}$ を持つ系列 $x:\mathtt{(B,L,D)}$ として扱われ、SSM state dimension は $N$ である。
- **既存研究との差分**: S4, S4D, S5, H3, Hyena, RetNet, RWKV などの多くは LTI recurrences / global convolutions として効率化されてきた。Mamba の差分は、SSM の系列方向の動力学そのものを入力依存にし、convolution が使えなくなる代わりに scan を GPU memory hierarchy に合わせて実装する点である。
- **この論文で答えたい問い**: 小さな recurrent state に context を圧縮するモデルが、必要な入力だけを選んで保持・破棄できれば、Transformer-quality performance と長文での線形スケーリングを両立できるか。

## 背景と前提

- Transformer の強さは、context window 内で token 同士を密に参照できる self-attention にある。一方で、window length に対して training cost が二次に増え、inference では過去 key/value を KV cache として保持する必要がある、という効率上の制約がある。
- Structured SSM / S4 は、連続時間の状態方程式を離散化し、recurrent mode または convolution mode で計算する系列変換である。従来の SSM は training では convolution、autoregressive inference では recurrence を使うのが典型で、効率のために $(\Delta,\A,\B,\C)$ が時刻によらず固定される。
- LTI（linear time invariant）は「系列中の全時刻で dynamics が同じ」という性質である。LTI なら convolution kernel を一度作って全位置に適用できるが、入力内容に応じて「この token は state に入れる / 無視する」を変えることはできない。
- 論文は、この弱点を synthetic tasks で説明する。Selective Copying は覚えるべき token の間隔がランダムで、Induction Heads は過去 context から対応する次 token を思い出す associative recall である（`src/method.tex`, `fig:copying`）。
- 既存 baseline との関係は次のように整理される。H3 は S4 を linear attention 風 architecture に入れたもの、Hyena は H3 の S4 layer を MLP-parameterized global convolution に置き換えたもの、RetNet は state dimension $N=1$ の単純な SSM と見なせる部分を持つもの、RWKV は LTI recurrence を含む RNN 系言語モデルである（`src/background.tex`, `src/related.tex`）。
- 論文中の "selection" は単なる multiplicative gate や hypernetwork 一般を指さない。著者は、系列長方向に信号を propagate する仕組みを入力依存にし、state が何を保持・忘却するかを制御する機構として使っている（`src/related.tex`, `sec:discussion:selection`）。

## 提案手法

### コアアイデア

著者の中心的な見方は、sequence modeling の根本問題は「長い context を小さな state に圧縮すること」だというものである。Attention は context を明示的に圧縮しないため強いが高コストで、recurrent model は有限 state なので効率的だが、state に必要情報を残せるかが性能を決める（`src/method.tex`, "Selection as a Means of Compression"）。

Mamba はこの圧縮問題に対し、SSM の $\Delta,\B,\C$ を入力 $x$ の関数にする。これにより、モデルは token ごとに state を更新する強さ、入力を state に入れる経路、state から出力を読む経路を変えられる。著者はこの time-varying SSM を "S6" とも呼ぶ。S6 は S4 に selection mechanism を加え、convolution ではなく scan で計算するという意味である（`src/method.tex`, `alg:s6`, remark）。

ただし入力依存にすると、LTI で成り立っていた convolution 表現が使えない。そこで Mamba は selective scan を kernel fusion, parallel scan, recomputation で実装し、巨大な state tensor $\mathtt{(B,L,D,N)}$ を HBM に materialize しない。さらに architecture としては H3 block と MLP block を 1 つにまとめ、同じ Mamba block を homogeneous に積む。

### 重要な定義・数式

$$
\begin{aligned}
h'(t) &= \A h(t) + \B x(t) \\
y(t) &= \C h(t) \\
h_t &= \dA h_{t-1} + \dB x_t \\
y_t &= \C h_t
\end{aligned}
$$

**式の意味**: 上 2 行は連続時間 SSM（`eq:ssm`）、下 2 行は離散化後の recurrent form（`eq:ssm:recurrence`）である。入力 $x$ を latent state $h$ に取り込み、state から出力 $y$ を作る sequence-to-sequence transformation を定義している。

**記号の定義**:
- $x(t), x_t$ ... 連続時間または時刻 $t$ の入力系列要素
- $h(t), h_t$ ... SSM の latent state。diagonal SSM では各 channel に $N$ 次元 state がある
- $y(t), y_t$ ... 出力系列要素
- $\A,\B,\C$ ... 連続時間側の SSM parameters
- $\dA,\dB$ ... discretization 後の SSM parameters

**この論文での役割**: Mamba はこの基本形を捨てるのではなく、$\Delta,\B,\C$ を入力依存にして time-varying にする。従来 SSM の弱点と提案手法の差分を読むための基礎式である。

$$
\begin{aligned}
\K &= (\C\dB, \C\dA\dB, \dots, \C\dA^{k}\dB, \dots) \\
y &= x \ast \K
\end{aligned}
$$

**式の意味**: LTI SSM では recurrent form を global convolution（`eq:ssm:convolution`）としても計算できる。kernel $\K$ を作れば、sequence 全体に convolution を適用して出力 $y$ を得られる。

**記号の定義**:
- $\K$ ... 離散化後の SSM から作られる convolution kernel
- $\ast$ ... convolution
- $\dA,\dB,\C$ ... 時刻によらず固定された離散 SSM parameters
- $k$ ... convolution kernel 内の lag index

**この論文での役割**: 従来 SSM が training で速い理由であると同時に、LTI 制約の源でもある。Mamba では parameters が時刻・入力ごとに変わるため、この convolution equivalence を失い、selective scan が必要になる。

$$
\dA = \exp(\Delta \bm{A})
\qquad
\dB = (\Delta \bm{A})^{-1}(\exp(\Delta \bm{A}) - \bm{I}) \cdot \Delta \bm{B}
$$

**式の意味**: Zero-order hold (ZOH) による discretization rule（`eq:zoh`）である。連続時間 parameters $(\Delta,\A,\B)$ から離散時間 recurrence に使う $(\dA,\dB)$ を作る。

**記号の定義**:
- $\Delta$ ... discretization step size。論文では $\dt$ と書かれる
- $\A,\B$ ... 連続時間側の SSM parameters
- $\dA,\dB$ ... 離散時間側の transition / input parameters
- $\bm{I}$ ... identity matrix

**この論文での役割**: $\Delta$ は gate のように state を保持するか現在入力へ寄せるかを制御する。著者が $\A$ を入力依存にしない理由もここにあり、$\dA=\exp(\Delta\A)$ により $\Delta$ 経由で $\dA,\dB$ に選択性が入る、と説明している（`src/method.tex`, "Interpretation of A"）。

$$
\begin{aligned}
s_B(x) &= \mathsf{Linear}_N(x) \\
s_C(x) &= \mathsf{Linear}_N(x) \\
s_\Delta(x) &= \mathsf{Broadcast}_D(\mathsf{Linear}_1(x)) \\
\Delta &= \tau_\Delta(\mathsf{Parameter} + s_\Delta(x)), \qquad \tau_\Delta=\mathsf{softplus}
\end{aligned}
$$

**式の意味**: S6 で使う選択機構の具体的な parameterization である（`src/method.tex`, `alg:s6`）。$\B,\C,\Delta$ を入力 $x$ の線形射影から作り、時刻方向の長さ次元 $L$ を持たせる。

**記号の定義**:
- $s_B,s_C,s_\Delta$ ... 入力 $x$ から SSM parameters を作る関数
- $\mathsf{Linear}_N$ ... 出力次元 $N$ の learned linear projection
- $\mathsf{Linear}_1$ ... 出力次元 1 の learned linear projection
- $\mathsf{Broadcast}_D$ ... $D$ channels へ broadcast する操作
- $\tau_\Delta$ ... $\Delta$ を正にするための変換。論文では $\mathsf{softplus}$ を使う

**この論文での役割**: ここが S4 から S6 への最小変更である。Alg. S4 では $\B,\C,\Delta$ は parameter だが、Alg. S6 では $\B:\mathtt{(B,L,N)}$, $\C:\mathtt{(B,L,N)}$, $\Delta:\mathtt{(B,L,D)}$ となり、time-invariant から time-varying へ変わる。

$$
\begin{aligned}
g_t &= \sigma(\mathsf{Linear}(x_t)) \\
h_t &= (1-g_t)h_{t-1} + g_t x_t
\end{aligned}
$$

**式の意味**: Theorem 3.1 (`thm:gating`, `eq:gates`) の結論である。$N=1,\A=-1,\B=1,s_\Delta=\mathsf{Linear}(x),\tau_\Delta=\mathsf{softplus}$ のとき、selective SSM recurrence は RNN の gate と同じ形になる。

**記号の定義**:
- $g_t$ ... 入力依存 gate。0 に近いと過去 state を保ち、1 に近いと現在入力へ寄せる
- $\sigma$ ... sigmoid function
- $x_t$ ... 時刻 $t$ の入力
- $h_{t-1},h_t$ ... 前時刻と現在時刻の state

**この論文での役割**: 著者は「SSM の discretization は heuristic gating mechanisms の principled foundation」と述べる。Mamba の選択性を、古典的 RNN gating と SSM 理論の接点として読むための式である。

### 実装 / アルゴリズム上の要点

- **S4 から S6 への変更**: Alg. S4 では $\A,\B,\C,\Delta$ が固定 parameter で、$\dA,\dB:\mathtt{(D,N)}$ を作り、recurrence または convolution で計算する。Alg. S6 では $\A:\mathtt{(D,N)}$ は固定のまま、$\B,\C,\Delta$ を $x$ から作り、$\dA,\dB:\mathtt{(B,L,D,N)}$ を scan only で使う。
- **selective scan**: naive recurrence は $O(BLDN)$ FLOPs、convolution は $O(BLD\log L)$ FLOPs だが、state dimension $N$ が大きすぎなければ recurrence 側の定数は小さい。問題は sequential recurrence と memory usage である。
- **memory hierarchy の利用**: 通常実装なら scan input の $\dA,\dB:\mathtt{(B,L,D,N)}$ と scan output $\mathtt{(B,L,D,N)}$ を HBM に書き、$\C$ と掛けて出力を作る。Mamba は $\Delta,\A,\B,\C$ を HBM から SRAM に読み、SRAM 内で discretization, parallel associative scan, $\C$ との積和を行い、最終出力 $\mathtt{(B,L,D)}$ だけを HBM に戻す（`src/method.tex`, `src/appendix.tex`）。
- **recomputation**: backward pass に必要な中間 state を保存せず、backward で再計算する。これにより activation memory は FlashAttention を使った最適化 Transformer と同程度になる、と著者は説明する。
- **Mamba block**: H3 block と Transformer の MLP block を 1 つに統合し、homogeneous に積む。各 block では model dimension $D$ を expansion factor $E$ で広げる。実験では常に $E=2$ とし、2 stacks of the block で Transformer の interleaved MHA + MLP blocks の $12D^2$ parameters に合わせる（`src/method.tex`, `fig:architecture`）。
- **追加 detail**: デフォルトでは real-valued SSM を使う。complex parameterization は多くの perceptual modalities で有効とされるが、本文が complex へ切り替えた唯一の実験として明記しているのは YouTubeMix の audio waveform pretraining である（`src/method.tex`, `src/experiments.tex`）。

## 実験・結果

- **データセット / ベンチマーク**:
  - Synthetic: Selective Copying と Induction Heads。Selective Copying は sequence length 4096、vocab size 16、memorize 16 data tokens、2-layer $D=64$、400K steps、learning rate 0.0001、batch size 64（`src/appendix.tex`）。Induction Heads は sequence length 256 で学習し、$2^6=64$ から $2^{20}=1048576$ まで評価する。
  - Language: Pile で autoregressive language modeling。Scaling laws は $\approx125M$ から $\approx1.3B$ parameters、Chinchilla protocol。Zero-shot は 300B tokens まで学習したモデルを LAMBADA, HellaSwag, PIQA, ARC-easy, ARC-challenge, WinoGrande で評価する。
  - DNA: HG38 human genome dataset。training split は $S=34021$ segments、各 segment length $2^{17}=131072$、合計約 4.5B DNA base-pair tokens。Context length は $2^{10}$ から $2^{20}$ まで扱う。Downstream は great apes 5 species classification。
  - Audio: YouTubeMix は 4 hours of solo piano music、16000 Hz。SC09 は digit "zero" through "nine" の 1-second clips、16000 Hz。
- **比較対象 / baseline**:
  - Synthetic: S4, H3, Hyena, Mamba architecture と inner layer S4 / Hyena / S6 の組み合わせ。Induction Heads では MHA-Abs, MHA-RoPE, MHA-xPos, H3, Hyena, Mamba。
  - Language: Transformer (GPT3 architecture), Transformer++ (rotary embedding, SwiGLU MLP, RMSNorm, no linear bias, higher learning rates など), Hyena, H3++, RWKV, RetNet。Zero-shot table では Pythia, RWKV, GPT-Neo, OPT, GPT-J も比較される。
  - DNA: Transformer++, HyenaDNA, Mamba。
  - Audio: SaShiMi, WaveNet, SampleRNN, WaveGAN, DiffWave, +SaShiMi, Mamba。
- **指標**:
  - Synthetic は accuracy。
  - Language は perplexity (ppl, lower is better) と zero-shot accuracy。HellaSwag と ARC-challenge は normalized accuracy を使う（`src/appendix.tex`）。
  - DNA pretraining は perplexity、species classification は accuracy。
  - Audio pretraining は BPB（bits per byte）。SC09 generation は NLL, FID, IS, mIS, AM。
  - Efficiency は scan speed, inference throughput, memory consumption。
- **主な結果**:
  - Selective Copying (`tab:copying`): S4 / no gate / S4 は 18.3、no gate architecture で inner layer を S6 にすると 97.0。H3 + S4 は 57.0、H3 + S6 は 99.7。Mamba + S4 は 56.4、Mamba + Hyena は 28.4、Mamba + S6 は 99.8。
  - Induction Heads (`fig:induction`, `tab:induction`): Mamba は train length $2^8=256$ から test length $2^{20}=1048576$ まで perfect generalization。著者は "4000x longer than it saw during training" と述べる。他の method は $2\times$ を超えて安定しない。
  - Language scaling (`fig:lm-scaling`): 著者は Mamba を「strong Transformer++ recipe に match した first attention-free model」と主張する。特に sequence length が伸びるほど差が出る、と figure caption と本文で述べる。
  - Zero-shot (`table:downstream_zeroshot`): Mamba-1.4B は Pile ppl 6.80、Average 59.7。Pythia-1.4B は Pile ppl 7.51、Average 55.2、RWKV-1.5B は Pile ppl 7.70、Average 54.3。Mamba-2.8B は Pile ppl 6.22、Average 63.3 で、Pythia-2.8B の 6.73 / 59.1、RWKV-3B の 7.00 / 59.6 を上回る。Mamba-2.8B の Average 63.3 は GPT-J-6B の 63.0、OPT-6.7B の 62.9、Pythia-6.9B の 61.7、RWKV-7.4B の 62.5 と同水準または上回る。
  - DNA scaling (`fig:dna`): 短い context length $2^{10}=1024$ で model size を $\approx200K$ から $\approx40M$ に増やすと、Mamba は HyenaDNA と Transformer++ より良く scaling する。最大 $\approx40M$ parameters の曲線では、Mamba は Transformer++ / HyenaDNA と同程度の perplexity を roughly $3\times$ to $4\times$ fewer parameters で達成できる、と著者は述べる。
  - DNA context length (`fig:dna`): 6 layers, width 128（約 1.3M-1.4M parameters）で sequence length $2^{10}$ から $2^{20}$ まで伸ばすと、Mamba の pretraining perplexity は context とともに改善する。一方 HyenaDNA は長くすると悪化する、と本文で説明される。
  - Great Apes DNA Classification (`tab:species`): random guessing は 20%。$2^{20}=1048576$ で HyenaDNA 1.4M は 54.87、Mamba 1.4M は 71.67、Mamba 7M は 81.31。短い長さでは Mamba が常に上ではなく、例えば $2^{12}$ では HyenaDNA 28.43、Mamba 1.4M 27.50 である。
  - YouTubeMix (`fig:youtubemix`): Mamba と SaShiMi はどちらも長い context で改善し、Mamba は全体を通じて良く、長くなるほど gap が広がる、と著者は述べる。最大長は clips の制約により 960000 付近で、実験表では $468\times2048=958464$ が最長（`tab:youtubemix-lengths`）。
  - SC09 (`tab:sc09`): Mamba 6.1M は NLL 1.852、FID 0.94、IS 6.26、mIS 88.54、AM 0.52。SaShiMi 5.8M は NLL 1.873、FID 1.99、IS 5.13、mIS 42.57、AM 0.74。Mamba 24.3M は FID 0.67、IS 7.33、mIS 144.9、AM 0.36。
  - Efficiency (`fig:scan_benchmark`, `src/benchmarks.tex`): SSM scan は state expansion $N=16$ で評価され、FlashAttention-2 より sequence length 2K 超で速く、標準 PyTorch scan より 20-40x 速い。Mamba は同程度サイズの Transformer より inference throughput が 4-5x 高く、untrained Mamba-6.9B は $5\times$ 小さい Transformer-1.3B より高い throughput とされる。
  - Ablation (`tab:ablations-variable`): selective parameter なしは ppl 10.93、selective $\B$ only は 10.15、selective $\C$ only は 9.98、selective $\Delta$ only は 9.81、$\Delta,\B,\C$ 全部 selective は 8.71。著者は $\Delta$ が最重要だが、複数 parameter の組み合わせに synergy があると述べる。
  - Ablation (`tab:ablations-arch`, `tab:ablations-N`): Mamba architecture + S6 は ppl 8.69、Mamba + S4(real) は 10.56、Mamba + S4(complex) は 10.54。state dimension $N$ は、$\B,\C$ が constant だと $N=1$ で 9.88、$N=16$ で 9.81 と小さい改善だが、$\B,\C$ が selective だと $N=1$ で 9.73、$N=16$ で 8.71 まで改善する。
- **著者が主張する貢献**:
  - SSM parameters を入力依存にする selection mechanism により、discrete modalities で content-based reasoning を可能にする。
  - convolution を失った time-varying SSM を、hardware-aware parallel scan で実用的に計算する。
  - attention も MLP blocks もない単純な Mamba architecture を提案する。
  - Language, audio, genomics の複数 modality で、強い Transformer や既存 SSM / convolution / recurrent baseline と同等以上の性能を線形時間で示す。

## 妥当性と限界

- **この主張を支える根拠**:
  - Selective Copying と Induction Heads は、LTI model が苦手な「内容に応じた保持」と「context に基づく recall」を直接測るため、selection mechanism の必要性を説明する実験として機能している。
  - `tab:ablations-variable` は $\Delta,\B,\C$ の selective 化を個別に切り分けており、とくに selective $\Delta$ と全部 selective の差を数値で示している。
  - `tab:ablations-N` は state dimension $N$ を増やすだけではなく、$\B,\C$ が selective なときに大きな改善が出ることを示す。これは「大きな state を持つだけでなく、何を入れるかを選ぶ必要がある」という本文の圧縮の議論と整合する。
  - Language, DNA, audio という性質の異なる modality で評価しているため、Mamba を domain-specific trick ではなく general sequence model backbone として主張する構成になっている。
  - Hardware benchmark は selective scan 単体、end-to-end inference、memory benchmark を分けており、selection で convolution を失っても実装上成立することを示す。
- **著者が認めている limitations / future work**:
  - `src/discussion.tex` の Scaling で、empirical evaluation は small model sizes に限られ、Llama, RWKV, RetNet のような 7B parameter scale and beyond で Mamba が有利なままかは未評価だと明記する。
  - 同じく `src/discussion.tex` で、fine-tuning, adaptation, prompting, in-context learning, instruction tuning, RLHF, quantization など、Transformer-based foundation models が持つ downstream affordances を SSM 代替が同様に持つかは関心事項として残されている。
  - "No Free Lunch: Continuous-Discrete Spectrum" で、selection mechanism は text / DNA のような discrete modalities では弱点を補うが、LTI SSM が得意な continuous-time data modalities では性能を阻害し得ると述べる。YouTubeMix ablation でも、audio waveform では S4 から S6 への変更が常に有益ではないと説明される。
  - `src/discussion.tex` は、SSM scaling には本論文で扱っていない engineering challenges and adjustments が必要かもしれないと述べる。
- **読者として注意すべき点**:
  - Language scaling の主要な曲線は `fig:lm-scaling` で示され、本文中に全 raw numbers が並ぶわけではない。数値比較を行うなら `table:downstream_zeroshot` と付録の model/training details を併読する必要がある。
  - Zero-shot table は tokenizers や pretraining dataset が異なる open source models も含む。Pile ppl は同じ dataset/tokenizer で学習した Pythia / RWKV と比較する、と `src/appendix.tex` に制限が書かれている。
  - Synthetic tasks の 1M 外挿は重要だが、自然言語の 1M-token downstream task を直接評価した結果ではない。著者の long-context 実証は DNA / audio の pretraining と synthetic tasks が中心である。
  - YouTubeMix の audio waveform pretraining では complex parameterization に切り替えており、real-valued default が全 modality で最適という主張ではない。
- **追加で確認したい実験 / 疑問**:
  - 7B 以上で Transformer++, RWKV, RetNet と同じ規模・token budget・implementation 条件で比較したとき、scaling law が保たれるか。
  - Fine-tuning, instruction tuning, RLHF, quantization を実施したとき、Transformer と同じ実用上の性質を持つか。
  - Long natural-language context で、Induction Heads のような合成外挿ではなく、実タスクの精度が context length とともに改善するか。
  - 同じ target perplexity / downstream score に到達するまでの総 GPU time や energy で、Mamba と Transformer++ がどう比較されるか。
  - 学習済み Mamba の $\Delta,\B,\C$ が、実際にどの token や boundary で state を保持・reset しているかを可視化できるか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Mamba**: selective SSM を組み込んだ attention-free architecture。H3 block と MLP block を統合した homogeneous block を積む。
- **SSM / structured SSM / S4**: この論文では広義の状態空間モデル一般ではなく、S4 系の structured state space sequence models を指す。
- **S6**: S4 に selection mechanism を加え、scan で計算する selective SSM の略称。論文の remark では S4 + selection + scan に由来すると説明される。
- **selection mechanism**: 入力内容に応じて、情報を sequence state に propagate するか、filter out するかを制御する機構。単なる GLU 的な multiplicative interaction とは区別される。
- **LTI (linear time invariant)**: $(\Delta,\A,\B,\C)$ が全 time-steps で固定される性質。convolution 計算を可能にするが、content-aware な選択を阻む。
- **$\Delta$ / $\dt$**: discretization step size。Mamba では入力依存にし、RNN gate の一般化として state を保持・reset する強さを制御する。
- **$\A,\B,\C$**: SSM parameters。Mamba では $\A$ は固定 parameter のまま、$\B,\C$ を入力依存にする。
- **$\dA,\dB$**: discretization 後の recurrence parameters。$\dA=\exp(\Delta\A)$ なので、$\Delta$ を入力依存にすれば $\dA,\dB$ にも選択性が入る。
- **state dimension $N$**: 各 channel の SSM latent state の次元。計算上は total hidden state が $DN$ になり、$\mathtt{(B,L,D,N)}$ の大きな tensor が問題になる。
- **selective scan**: time-varying SSM を recurrence として並列化するための scan。Mamba では kernel fusion と recomputation で HBM IO を抑える。
- **HBM / SRAM**: GPU memory hierarchy。HBM は大きいが遅く、SRAM は小さいが速い。Mamba は大きな expanded state を HBM に書かず、SRAM 内で処理する。
- **Transformer++**: PaLM / LLaMA 系の改良を反映した強い Transformer recipe。本文では rotary embedding, SwiGLU MLP, RMSNorm, no linear bias, higher learning rates などが挙げられる。
- **H3 / Hyena / RetNet / RWKV**: Mamba の主要比較対象。H3 は S4 を用いる SSM architecture、Hyena は global convolution 系、RetNet と RWKV は recurrent / attention-free 系 baseline として扱われる。
- **BPB**: bits per byte。audio pretraining で使われる metric で、negative log-likelihood の定数倍と説明される。
- **FID / IS / mIS / AM**: SC09 speech generation の自動評価指標。FID と AM は低いほどよく、IS と mIS は高いほどよい。

## 読む順番の提案

- まず `notes/arXiv-2312.00752v2.md` の Summary と Takeaway を読み、論文が「LTI では content-based reasoning ができない」という診断に立っていることを確認する。
- 次に `papers/arXiv-2312.00752v2/main.tex` で title, authors, `\input{...}` の流れを見る。本文は `src/abstract.tex` → `src/intro.tex` → `src/background.tex` → `src/method.tex` → `src/experiments.tex` → `src/discussion.tex` の順で読むとよい。
- 数式は `src/background.tex` の `eq:ssm`, `eq:ssm:recurrence`, `eq:ssm:convolution`, `eq:zoh` を先に読む。これが正規ノートの「LTI vs データ依存」「$\Delta$ が gate の正体」に対応する。
- 手法は `src/method.tex` の `alg:s4`, `alg:s6`, `thm:gating`, `fig:architecture` を見る。とくに S4 と S6 の tensor shape の違いを確認すると、なぜ convolution ではなく scan が必要になるかが分かる。
- 実験は `src/experiments.tex` の `tab:copying`, `fig:induction`, `table:downstream_zeroshot`, `fig:dna`, `tab:sc09`, `fig:scan_benchmark` を先に見る。正規ノートの数値 summary は主にここに対応する。
- 詳細な裏取りは `src/appendix.tex` を使う。`tab:induction`, `tab:gpt3`, downstream evaluation details, HG38 details, `tab:species`, `tab:youtubemix-lengths`, `tab:memory` が、本文の主張を支える条件と raw numbers を補う。
- 限界は `src/discussion.tex` の "No Free Lunch", "Downstream Affordances", "Scaling" を読む。正規ノートの Critical Thoughts のうち、著者自身が認めている点と読者の追加疑問を分けて読むと混乱しにくい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2312.00752v2/`
- 正規ノート: `notes/arXiv-2312.00752v2.md`
