# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（IO-aware exact attention と長系列 Transformer 高速化）

- arXiv: https://arxiv.org/abs/2205.14135
- 一次ソース: ../papers/arXiv-2205.14135v2/
- 正規ノート: ../notes/arXiv-2205.14135v2.md

---

## 一言で言うと

Transformer の self-attention は系列長 $N$ に対して時間・メモリが二乗で増えるが、この論文は attention の数学的定義を変えずに、GPU の HBM と SRAM の間の読み書き、つまり IO を減らすことで高速化する。提案手法 FlashAttention は tiling と recomputation により $N \times N$ の attention matrix を HBM に materialize せず、BERT-large、GPT-2、LRA、Path-X / Path-256 で速度・長文脈性能の改善を示す。

## 何を議論する論文か

- **問題設定**: Transformer は広く使われるが、中心部の self-attention は sequence length に対して time complexity と memory complexity が quadratic である。標準実装では $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$ と $\mathbf{P}=\operatorname{softmax}(\mathbf{S})$ という $N \times N$ 行列を HBM に書き出すため、長い系列で runtime と memory footprint が大きくなる（`src/background.tex`, Algorithm 0）。
- **対象範囲 / 仮定**: 主対象は GPU 上の attention computation で、入力 $\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$、head dimension $d$、on-chip SRAM size $M$ を使う。理論解析では $d \leq M \leq Nd$ を仮定する（Theorem `thm:io_complexity`）。実験の主要環境は A100 GPU で、BERT/GPT-2/LRA/long document/path tasks を扱う。
- **既存研究との差分**: Reformer, Linformer, Performer, Longformer, BigBird などの approximate attention は FLOPs を線形または準線形に減らすが、著者らは「wall-clock speedup に結びつかないことが多い」原因として memory access overhead を指摘する。FlashAttention は近似ではなく exact attention を計算し、FLOPs より HBM reads/writes を減らすことを主軸にする。
- **この論文で答えたい問い**: attention algorithm を IO-aware に設計すると、attention の値を変えずに training を速くし、長い context を使えるようになるか。また、その HBM access は標準 attention より少なく、exact attention として理論的にどの程度まで改善できるか。

## 背景と前提

- GPU memory hierarchy では、小さいメモリほど速い。A100 の例として、HBM は 40-80GB、bandwidth 1.5-2.0TB/s、on-chip SRAM は streaming multiprocessor あたり 192KB、108 SM、bandwidth は約 19TB/s と説明される（`src/background.tex`, "GPU Memory Hierarchy"）。この差が、FLOPs だけでなく IO を設計対象にする理由である。
- この論文での **IO-aware** は、GPU on-chip SRAM と HBM のような fast / slow memory 間の reads and writes を数える、という意味で使われる。出典として Aggarwal and Vitter (1988) の IO complexity が参照される。
- **memory-bound** な操作では、計算回数より HBM access が runtime を支配する。softmax, dropout, masking, layer norm などの reduction / elementwise 操作は memory-bound の例として挙げられる。attention では matrix multiply だけでなく softmax と中間行列の保存が関わるため、単純な FLOP 削減だけでは wall-clock time を説明しにくい。
- 標準 attention は $\mathbf{S}$ と $\mathbf{P}$ を HBM に materialize する。特に GPT-2 の例では $N=1024$, $d=64$ とされ、$N \gg d$ なので $N \times N$ 行列が支配的になる。
- FlashAttention が使う既存技術は、online softmax / softmax scaling（Milakov and Gimelshein 2018; Rabe and Staats 2021 など）と selective gradient checkpointing / recomputation である。ただし著者らの差分は、これらを IO-aware な CUDA kernel として組み合わせ、速度向上まで狙う点にある。
- block-sparse FlashAttention は exact attention ではなく approximate attention への拡張である。論文では固定の butterfly sparsity pattern を downstream experiments に使うと書かれている（`src/extension.tex`）。

## 提案手法

### コアアイデア

FlashAttention の目的は、attention matrix $\mathbf{S},\mathbf{P} \in \mathbb{R}^{N \times N}$ を HBM に保存せずに、最終出力 $\mathbf{O}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ を正確に計算することである。中心は 2 つの技術である。

1. **Tiling**: $\mathbf{Q},\mathbf{K},\mathbf{V}$ をブロックに分ける。外側ループで $\mathbf{K}_j,\mathbf{V}_j$ を HBM から SRAM にロードし、内側ループで $\mathbf{Q}_i,\mathbf{O}_i,\ell_i,m_i$ をロードする。各ブロックで $\mathbf{S}_{ij}=\mathbf{Q}_i\mathbf{K}_j^\top$ を計算し、row-wise な最大値 $m$ と正規化和 $\ell$ を更新しながら $\mathbf{O}_i$ を再正規化する（Algorithm `alg:stream_attn`）。
2. **Recomputation**: backward pass に必要な $\mathbf{S},\mathbf{P}$ を forward で保存しない。代わりに $\mathbf{O}$ と softmax statistics $(m,\ell)$ を保存し、backward 時に $\mathbf{Q},\mathbf{K},\mathbf{V}$ のブロックから SRAM 上で $\mathbf{S},\mathbf{P}$ を再計算する。FLOPs は増えるが HBM access が減るため、Figure `fig:micros` では standard attention より速い。

このため FlashAttention は approximate attention ではない。Theorem `thm:correctness` は Algorithm `alg:stream_attn` が $\mathbf{O}=\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ を返し、FLOPs は $O(N^2d)$、input/output 以外の追加 memory は $O(N)$ であると述べる。

### 重要な定義・数式

$$
\mathbf{S} = \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times N}, \quad
\mathbf{P} = \operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad
\mathbf{O} = \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d}
$$

**式の意味**: 標準 self-attention の定義である。$\mathbf{Q}\mathbf{K}^\top$ で全 token pair の score matrix $\mathbf{S}$ を作り、row-wise softmax で attention weight $\mathbf{P}$ に変換し、$\mathbf{V}$ に掛けて出力 $\mathbf{O}$ を得る（`src/background.tex`, "Standard Attention Implementation"）。

**記号の定義**:
- $\mathbf{Q},\mathbf{K},\mathbf{V}$ ... input sequences、すべて $\mathbb{R}^{N \times d}$ の行列
- $N$ ... sequence length
- $d$ ... head dimension
- $\mathbf{S}$ ... score matrix、$\mathbb{R}^{N \times N}$
- $\mathbf{P}$ ... row-wise softmax 後の attention matrix
- $\mathbf{O}$ ... attention output、$\mathbb{R}^{N \times d}$

**この論文での役割**: 標準 attention が $N \times N$ の $\mathbf{S},\mathbf{P}$ を HBM に materialize する点が、FlashAttention の解くべきボトルネックである。提案手法はこの式の結果を変えずに、計算の順序と保存する中間量を変える。

$$
m(x) := \max_i x_i,\quad
f(x) := \begin{bmatrix}e^{x_1-m(x)} & \cdots & e^{x_B-m(x)}\end{bmatrix},\quad
\ell(x) := \sum_i f(x)_i,\quad
\operatorname{softmax}(x) := \frac{f(x)}{\ell(x)}
$$

**式の意味**: 数値安定な softmax の定義である。最大値 $m(x)$ を引いて exponentiation し、和 $\ell(x)$ で割る。FlashAttention はブロックごとにこの統計量を更新する。

**記号の定義**:
- $x \in \mathbb{R}^{B}$ ... softmax をかける 1 行またはそのブロック
- $m(x)$ ... $x$ の最大値
- $f(x)$ ... 最大値で shift した指数ベクトル
- $\ell(x)$ ... $f(x)$ の総和、softmax normalizer
- $B$ ... ブロック内の要素数

**この論文での役割**: $\mathbf{S}$ の全列を同時に見なくても、各行の $m$ と $\ell$ を持ち回れば softmax を正確に集約できる。これは tiling を exact にするための基礎である（`src/algo.tex`, "Tiling"）。

$$
m_i^{\mathrm{new}} = \max(m_i, \tilde{m}_{ij}), \quad
\ell_i^{\mathrm{new}}
= e^{m_i - m_i^{\mathrm{new}}}\ell_i
  + e^{\tilde{m}_{ij} - m_i^{\mathrm{new}}}\tilde{\ell}_{ij}
$$

**式の意味**: Algorithm `alg:stream_attn` の各ブロック更新で、これまで処理した列ブロックの統計量 $(m_i,\ell_i)$ と、新しい score block $\mathbf{S}_{ij}$ の統計量 $(\tilde{m}_{ij},\tilde{\ell}_{ij})$ を、共通の最大値 $m_i^{\mathrm{new}}$ に揃えて合成する。

**記号の定義**:
- $i$ ... $\mathbf{Q}$ と $\mathbf{O}$ の row block index
- $j$ ... $\mathbf{K},\mathbf{V}$ の column block index
- $\tilde{m}_{ij}$ ... $\mathbf{S}_{ij}$ の rowmax
- $\tilde{\ell}_{ij}$ ... $\exp(\mathbf{S}_{ij}-\tilde{m}_{ij})$ の row sum
- $m_i,\ell_i$ ... block $i$ について、これまでの列ブロックから得た softmax statistics

**この論文での役割**: この更新により、ブロック単位で処理しても、最後の $\mathbf{O}$ は一括で $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ を計算した結果と一致する。Algorithm `alg:stream_attn` の正しさと、$N \times N$ attention matrix を保存しない設計を支える式である。

$$
\text{Standard attention}: \Theta(Nd + N^2), \qquad
\text{FlashAttention}: \Theta(N^2 d^2 M^{-1})
$$

**式の意味**: Theorem `thm:io_complexity` の HBM access complexity である。標準 attention は $\mathbf{S},\mathbf{P}$ の読み書きにより $\Theta(Nd+N^2)$ HBM access を必要とする。一方、FlashAttention は SRAM に載る block を使って $\Theta(N^2d^2/M)$ に減らす。

**記号の定義**:
- $\Theta(\cdot)$ ... 漸近的に同じオーダーの量
- $N$ ... sequence length
- $d$ ... head dimension
- $M$ ... SRAM size、仮定は $d \leq M \leq Nd$
- $M^{-1}$ ... $1/M$。SRAM が大きいほど HBM access が減ることを表す

**この論文での役割**: 論文の理論的主張の中心である。著者らは typical values として $d=64$-128、$M$ around 100KB では $d^2$ が $M$ より何倍も小さいため、標準実装より HBM access が少ないと説明する。また Proposition `thm:lower_bound` は、すべての $M \in [d,Nd]$ に対して exact attention で $o(N^2d^2M^{-1})$ HBM access を達成する algorithm は存在しないと述べる。

$$
\Theta\left(Nd + N^2 d^2 M^{-1}s\right)
$$

**式の意味**: Proposition `thm:io_complexity_blocksparse` の block-sparse FlashAttention の HBM access complexity である。$s$ は block-sparsity mask の nonzero block fraction で、ゼロ block を skip することで大きい項が $s$ 倍になる。

**記号の定義**:
- $s$ ... block-sparsity mask の非ゼロ block の割合
- $\mathbf{M} \in \{0,1\}^{N/B_r \times N/B_c}$ ... block sparsity mask
- $\tilde{\mathbf{M}} \in \{0,1\}^{N \times N}$ ... element-level mask。ただし block form を持つことが要求される
- $Nd$ ... 出力 $\mathbf{O} \in \mathbb{R}^{N \times d}$ の書き出しなどで残る項

**この論文での役割**: FlashAttention が exact attention の高速化 primitive であるだけでなく、approximate / sparse attention にも IO-aware 実装を与えられることを示す。downstream experiments では fixed butterfly sparsity pattern を使う。

### 実装 / アルゴリズム上の要点

- Algorithm 0 の標準実装は、$\mathbf{S}=\mathbf{Q}\mathbf{K}^\top$ を HBM に書き、$\mathbf{S}$ を読んで $\mathbf{P}=\operatorname{softmax}(\mathbf{S})$ を HBM に書き、さらに $\mathbf{P}$ と $\mathbf{V}$ を読んで $\mathbf{O}$ を書く。ここが $O(N^2)$ memory と HBM access の原因である。
- Algorithm 1 の forward pass は、まず $\mathbf{O}=(0)_{N\times d}$、$\ell=(0)_N$、$m=(-\infty)_N$ を HBM に初期化する。block size は $B_c=\lceil M/(4d)\rceil$、$B_r=\min(\lceil M/(4d)\rceil,d)$ とし、$\mathbf{Q}$ を $B_r \times d$、$\mathbf{K},\mathbf{V}$ を $B_c \times d$ の block に分ける。
- 外側ループは $\mathbf{K}_j,\mathbf{V}_j$ を SRAM に読み込む。内側ループは $\mathbf{Q}_i,\mathbf{O}_i,\ell_i,m_i$ を読み込み、on chip で $\mathbf{S}_{ij}$、$\tilde{m}_{ij}$、$\tilde{\mathbf{P}}_{ij}$、$\tilde{\ell}_{ij}$ を計算する。
- $\mathbf{O}_i$ は各 block ごとに再正規化して HBM に書き戻すが、$\mathbf{S}$ と $\mathbf{P}$ 全体は HBM に保存しない。これが「avoid reading and writing the attention matrix to and from HBM」という本文の主目標に対応する。
- backward pass では $\mathbf{P}$ や dropout mask を $O(N^2)$ で保存しない。pseudo-random number generator state ${\cal R}$ を保存して dropout mask を再生成し、$\mathbf{O}$ と $(m,\ell)$ から block-wise に $\mathbf{P}_{ij}$ を再計算する（`src/algo_details.tex`, Algorithm `alg:bwd_full`）。
- 実装は CUDA kernel fusion を使う。matrix multiply、softmax、optional masking/dropout、matrix multiply を 1 つの GPU kernel に融合し、PyTorch/TensorFlow の高水準 API では難しい fine-grained memory access control を行う。
- block-sparse FlashAttention は Algorithm 1 と同じだが、block sparsity mask $\mathbf{M}$ で $\mathbf{M}_{ij}=0$ の block を skip する。論文の記述では "the algorithm is identical ... except we skip zero blocks" と説明される。

## 実験・結果

- **データセット / ベンチマーク**: BERT-large on Wikipedia、GPT-2 small/medium on OpenWebText、Long-Range Arena (ListOps, Text, Retrieval, Image, Pathfinder)、long document classification の MIMIC-III と ECtHR、Path-X と Path-256、attention 単体 benchmark を用いる。Path-X は $128 \times 128$、Path-256 は $256 \times 256$ の白黒画像を pixel sequence として入力し、2 点が path で接続されているかを分類する。
- **比較対象 / baseline**: BERT では Nvidia MLPerf 1.1 training speed record、GPT-2 では HuggingFace `transformers` と Megatron-LM、LRA では Transformer、Linformer、Linear Attention、Performer、Local Attention、Reformer、Smyrf を比較する。attention benchmark では PyTorch/HuggingFace, Megatron, Reformer, Local Attention, Linformer, Smyrf, LSFormer, OpenAI Block-Sparse, Longformer, BigBird などの reference implementation と比較する（`src/exp_supp.tex`, "Baselines"）。
- **指標**: training time、wall-clock speedup、OpenWebText perplexity (ppl)、LRA accuracy / average / speedup、micro $F_1$、Path-X / Path-256 accuracy、attention forward+backward runtime、memory usage を使う。
- **主な結果**: Figure `fig:micros` の GPT-2 medium 相当の micro benchmark では、standard attention は 66.6 GFLOPs、40.3 GB HBM R/W、41.7 ms、FlashAttention は 75.2 GFLOPs、4.4 GB HBM R/W、7.3 ms である。FLOPs は増えるが HBM R/W が大きく減り、runtime が短くなる。
- **主な結果**: BERT-large は MLPerf benchmark の同じ initialization から masked language modeling target accuracy 72.0% に到達する時間を測る。8 x A100 GPUs、10 runs average で Nvidia MLPerf 1.1 は $20.0 \pm 1.5$ minutes、FlashAttention は $\mathbf{17.4} \pm 1.4$ minutes で、本文では 15% faster と述べる（Table `table:bert_speed`）。
- **主な結果**: GPT-2 small on OpenWebText は HuggingFace 18.2 ppl / 9.5 days、Megatron-LM 18.2 ppl / 4.7 days、FlashAttention 18.2 ppl / 2.7 days (3.5x)。GPT-2 medium は HuggingFace 14.2 ppl / 21.0 days、Megatron-LM 14.3 ppl / 11.5 days、FlashAttention 14.3 ppl / 6.9 days (3.0x)（Table `table:gpt_finetune`）。
- **主な結果**: GPT-2 small の long context では、Megatron-LM context 1k が 18.2 ppl / 4.7 days、FlashAttention context 1k が 18.2 ppl / 2.7 days、2k が 17.6 ppl / 3.0 days、4k が 17.5 ppl / 3.6 days。本文は context length 4K でも Megatron-LM 1K より 30% faster で、0.7 better perplexity と述べる（Table `table:gpt2_long_context`）。
- **主な結果**: Long Document performance は micro $F_1$。MIMIC-III は sequence length 512 で 52.8、1024 で 50.7、2048 で 51.7、4096 で 54.6、8192 で 56.4、16384 で 57.1。ECtHR は 512 で 72.2、1024 で 74.3、2048 で 77.1、4096 で 78.6、8192 で 80.7、16384 で 79.2（Table `tab:mimic`）。
- **主な結果**: LRA では Transformer average 59.3、FlashAttention 59.8 / 2.4x、Block-sparse FlashAttention 59.6 / 2.8x。近似 baseline は Linformer 54.9 / 2.5x、Linear Attention 59.6 / 2.3x、Performer 58.9 / 1.8x、Local Attention 56.0 / 1.7x、Reformer 57.6 / 1.3x、Smyrf 57.9 / 1.7x（Table `table:lra`）。
- **主な結果**: Path-X / Path-256 では、Transformer, Linformer, Linear Attention, Performer, Local Attention, Reformer, SMYRF は表中で `\xmark`。FlashAttention は Path-X 61.4、Path-256 は `\xmark`。Block-sparse FlashAttention は Path-X 56.0、Path-256 63.1（Table `table:pathx`）。本文は Path-X について "first Transformer"、Path-256 について "first sequence model that we know of" と述べる。
- **主な結果**: attention benchmark では、A100 40GB HBM、dropout と padding mask ありで sequence length を変える。FlashAttention は common sequence lengths 128 to 2K で PyTorch implementation より up to 3x faster、memory footprint は linear in sequence length で exact attention baselines より up to 20x more memory efficient と本文が報告する。approximate attention は 512 から 1024 の間で runtime が FlashAttention と交差し始めるが、block-sparse FlashAttention は著者らの知る限り全 sequence lengths で exact/sparse/approximate implementations より速いと述べられる。
- **著者が主張する貢献**: exact attention の IO-aware algorithm、tiling と recomputation を使った CUDA implementation、HBM access の上限と lower bound、block-sparse FlashAttention への拡張、BERT/GPT-2/LRA/Path-X/Path-256 での速度・長文脈性能、実装の open-source 公開である。

## 妥当性と限界

- **この主張を支える根拠**: 理論面では Theorem `thm:correctness` が exactness と $O(N)$ additional memory を示し、Theorem `thm:io_complexity` が standard attention と FlashAttention の HBM access を比較し、Proposition `thm:lower_bound` が exact attention の HBM access に関する lower bound を与える。実験面では Figure `fig:micros` が、FLOPs が増えても HBM R/W が 40.3 GB から 4.4 GB に減り runtime が 41.7 ms から 7.3 ms になることを示す。
- **この主張を支える根拠**: end-to-end training では BERT-large と GPT-2 で、モデル定義を変えずに training time を短縮している。GPT-2 では perplexity が HuggingFace / Megatron-LM と一致または同等であり、`src/exp_supp.tex` の Figure `fig:gpt2_training_curve` は validation perplexity curves がほぼ重なると述べる。
- **この主張を支える根拠**: 長文脈の有用性は、GPT-2 context 4K で 17.5 ppl、long document classification の MIMIC-III / ECtHR micro $F_1$ 改善、Path-X / Path-256 の non-random performance によって示される。ただし、これらは FlashAttention の algorithmic speed/memory 改善によって長い sequence length を扱えるようになった結果として提示されている。
- **著者が認めている limitations / future work**: 現在の IO-aware attention implementation は、新しい attention implementation ごとに新しい CUDA kernel を書く必要があり、PyTorch より低水準で大きな engineering effort を要する。GPU architecture 間で implementation が transferrable でない可能性もある（`src/discussion.tex`, "Compiling to CUDA"）。
- **著者が認めている limitations / future work**: 著者らは、PyTorch のような高水準言語で attention algorithm を書き、IO-aware CUDA implementation に compile する方法を future direction として挙げる。例として Halide に似た方向が述べられる。
- **著者が認めている limitations / future work**: single GPU 上の attention では constants まで含めて optimal と述べるが、multi-GPU では GPU 間 transfer という追加の memory hierarchy があり、別の IO analysis が必要になる。attention 以外の deep learning modules、sparse MLP layers、kernel machine learning への拡張も potential extensions として述べられる。
- **読者として注意すべき点**: FlashAttention 本体は exact attention だが、block-sparse FlashAttention は mask によりゼロ block を skip する approximate attention である。この 2 つを同じ「正確な attention」として扱ってはいけない。
- **読者として注意すべき点**: 論文の主要実験は A100 を中心にしている。Appendix では RTX 3090、T4、head dimension 128 の speedup も示されるが、discussion で著者自身が GPU architecture 間の移植性を限界として認めている。
- **読者として注意すべき点**: Long Document Classification の MIMIC-III は sequence length と性能が単調ではない。本文も MIMIC-III と ECtHR の discrepancies について、MIMIC-III の specialized medical text が document length の distribution shift により影響を受けやすい可能性を述べている。
- **追加で確認したい実験 / 疑問**: 本文では approximate attention baseline の多くを reference implementation として比較している。Linformer / Performer などを同じ程度に IO-aware に実装した場合の比較は、TeX 中には示されていない。block-sparse FlashAttention はその方向の proof of concept として読めるが、すべての近似手法に対する結論ではない。
- **追加で確認したい実験 / 疑問**: downstream experiments で使う fixed butterfly sparsity pattern の pattern choice に対する感度分析は、TeX 中には詳しく示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **FlashAttention**: tiling と recomputation により、exact attention を $N \times N$ attention matrix の HBM materialization なしに計算する IO-aware CUDA algorithm。
- **IO-aware**: FLOPs だけでなく、HBM と SRAM の間の reads/writes を設計・解析の対象にすること。この論文では attention の HBM access を減らすことに直接結びつく。
- **HBM**: GPU の high bandwidth memory。容量は大きいが SRAM より遅い。A100 の例では 40-80GB、1.5-2.0TB/s。
- **SRAM / on-chip SRAM**: GPU の streaming multiprocessor 近くにある小さいが速いメモリ。A100 の例では SM あたり 192KB、108 SM、推定 bandwidth around 19TB/s。
- **materialize**: 中間行列を実際にメモリ上に書き出すこと。標準 attention は $\mathbf{S},\mathbf{P}$ を HBM に materialize する。
- **tiling**: 入力や行列を block に分け、SRAM に載る単位で処理すること。FlashAttention では $\mathbf{K},\mathbf{V}$ の block と $\mathbf{Q},\mathbf{O}$ の block を二重ループで扱う。
- **online softmax / softmax scaling**: 全列を一度に持たず、最大値 $m$ と正規化和 $\ell$ を更新して row-wise softmax を正確に集約する技術。
- **recomputation**: backward pass のために $\mathbf{S},\mathbf{P}$ を保存せず、必要になった時に block-wise に再計算すること。この論文では selective gradient checkpointing の一種として説明される。
- **kernel fusion**: 複数の GPU operations を 1 つの kernel にまとめ、途中の HBM 読み書きを減らすこと。FlashAttention は matrix multiply、softmax、masking/dropout、matrix multiply を融合する。
- **memory-bound / compute-bound**: runtime が memory access に支配される場合が memory-bound、arithmetic operations に支配される場合が compute-bound。論文は attention runtime で HBM access が primary factor だと主張する。
- **arithmetic intensity**: memory access 1 byte あたりの arithmetic operations 数。compute-bound / memory-bound を見るための指標として background で説明される。
- **block-sparse FlashAttention**: block sparsity mask $\mathbf{M}$ に従ってゼロ block を skip する FlashAttention の拡張。exact attention ではなく approximate attention として位置づけられる。
- **butterfly sparsity**: downstream experiments で使われる fixed block sparsity pattern。`src/extension.tex` では Dao et al. 2022a と Dao et al. 2020 が関連文献として参照される。
- **LRA / Long-Range Arena**: 長い系列を扱う efficient Transformer の benchmark。本文表では ListOps, Text, Retrieval, Image, Pathfinder の 5 tasks を比較する。
- **Path-X / Path-256**: LRA 系の長文脈 classification task。$128 \times 128$ または $256 \times 256$ の白黒画像を 1 pixel ずつ Transformer に入力し、2 点が path で接続されるかを分類する。
- **perplexity (ppl)**: GPT-2 実験で用いられる language modeling の評価指標。Table `table:gpt_finetune` と `table:gpt2_long_context` では OpenWebText (ppl) として報告される。
- **micro $F_1$**: Long Document Classification の評価指標。MIMIC-III と ECtHR の sequence length ごとの性能として Table `tab:mimic` に報告される。

## 読む順番の提案

- まず `src/background.tex` の "Standard Attention Implementation" と Algorithm 0 を読む。ここで $\mathbf{S},\mathbf{P}$ を HBM に materialize する標準実装のどこが問題かを押さえる。正規ノートでは Summary の「問題」と Takeaway の「FLOPs を減らすより HBM アクセスを減らす」に対応する。
- 次に `src/algo.tex` の Algorithm `alg:stream_attn` を読む。最初は全行を追い切らなくてよいが、$B_c,B_r$、外側ループの $\mathbf{K}_j,\mathbf{V}_j$、内側ループの $\mathbf{Q}_i,\mathbf{O}_i,\ell_i,m_i$、$m_i^{\mathrm{new}},\ell_i^{\mathrm{new}}$、$\mathbf{O}_i$ 更新式を確認する。正規ノートでは Summary の "Tiling" と Notes / Quotes の統計量更新に対応する。
- その後 `src/theory.tex` の Theorem `thm:io_complexity` と Proposition `thm:lower_bound` を読む。ここで「速い」という主張が単なる実装観察ではなく、HBM access complexity と lower bound に基づく主張であることを確認する。
- block-sparse 版は `src/extension.tex` を読む。$\tilde{\mathbf{M}}$ と $\mathbf{M}$ の block form、zero blocks を skip する設計、$\Theta(Nd+N^2d^2M^{-1}s)$ の意味を確認する。正規ノートでは block sparsity との直交性に対応する。
- 実験は `src/experiments.tex` を、BERT/GPT-2/LRA/long context/attention benchmark の順に読む。数値は Table `table:bert_speed`, `table:gpt_finetune`, `table:lra`, `table:gpt2_long_context`, `tab:mimic`, `table:pathx` と Figure `fig:micros`, `fig:benchmark` を参照する。
- 最後に `src/discussion.tex` を読む。CUDA kernel を手書きする必要、GPU architecture 間の移植性、multi-GPU IO analysis の未対応という限界は、正規ノートの Critical Thoughts と対応する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2205.14135v2/`
- 正規ノート: `notes/arXiv-2205.14135v2.md`
