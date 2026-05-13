# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

- arXiv: https://arxiv.org/abs/2205.14135
- source: ../papers/arXiv-2205.14135v2/
- authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- venue / year: NeurIPS 2022
- tags: [attention, transformer, systems, GPU, IO-complexity, long-context]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: Transformer の self-attention は系列長 $N$ に対して計算量・メモリとも $O(N^2)$。既存の近似 attention（Reformer / Linformer / Performer / Longformer / BigBird など）は FLOPs を線形〜準線形に落とすが、ほとんどが標準実装に対する wall-clock 速度向上を出せておらず採用が進んでいない。著者らは原因を「アルゴリズムが IO 非対応であること」だと主張する——GPU 計算速度はメモリ速度に対して年々開いており、attention は実際は memory-bound だが、PyTorch/TF からは HBM ↔ SRAM のアクセス制御ができない。
- **手法**: FlashAttention は exact attention（近似ではない）を、HBM アクセスを劇的に減らす形で計算する単一 CUDA kernel。鍵となるテクニックは2つ:
    1. **Tiling**: $Q, K, V$ をブロックに分割し、外側ループで $K_j, V_j$ を SRAM にロード、内側ループで $Q_i, O_i$ をロード。online softmax（Milakov & Gimelshein 2018, Rabe & Staats 2021）流に各行ごとに $m_i$（rowmax）と $\ell_i$（行 exp 和）を持ち回り、ブロックごとに再正規化しながら $O$ を更新する（Algorithm 1）。これで $N \times N$ の $S, P$ を HBM に書き出さずに済む。
    2. **Recomputation**: 後方計算で必要な attention 行列 $S, P$ は保存せず、$O$ と統計量 $(m, \ell)$ だけ保存し、backward 時に SRAM 上で再計算する。FLOPs は増えるが HBM アクセスが減るので結果的に速い。Selective gradient checkpointing と見なせる。
- 拡張として、ブロック単位の sparsity mask $\tilde M$ を受け取り、ゼロブロックをスキップする **block-sparse FlashAttention** を提案（butterfly sparsity を使用）。IO 複雑度は sparsity $s$ 倍だけ良くなる。
- **理論**: 系列長 $N$、head 次元 $d$、SRAM サイズ $M$（$d \le M \le Nd$）とおくと、標準 attention は $\Theta(Nd + N^2)$ HBM アクセス、FlashAttention は $\Theta(N^2 d^2 M^{-1})$ HBM アクセス（Theorem 2）。典型値 $d \in [64, 128]$, $M \approx 100\mathrm{KB}$ では $d^2 \ll M$ なので大幅に少ない。さらに「すべての $M \in [d, Nd]$ に対して $o(N^2 d^2 / M)$ を達成する exact attention は存在しない」という lower bound も示す（Proposition 3）。Block-sparse 版は $\Theta(Nd + N^2 d^2 M^{-1} s)$。
- **結果**:
    - マイクロ計測（GPT-2 medium, $N=1024, d=64$, 16 heads, batch 64, A100）: 標準 66.6 GFLOPs / 40.3 GB HBM R/W / 41.7 ms → FlashAttention 75.2 GFLOPs / 4.4 GB / **7.3 ms**（Fig. 3 表）。FLOPs は増えても HBM が約 9× 削減され速度は約 5.7× に。
    - **BERT-large** (seq 512, 8×A100): MLPerf 1.1 record 20.0 分 → FlashAttention **17.4 分** で 15% 高速（Table 1）。
    - **GPT-2 small** (seq 1K, OpenWebText, 8×A100): HuggingFace 9.5 日 / Megatron-LM 4.7 日 / FlashAttention **2.7 日（3.5×）**。GPT-2 medium: 21.0 日 / 11.5 日 / **6.9 日（3.0×）**。perplexity は同等（small 18.2, medium 14.3）。
    - **GPT-2 長文脈**: context 4K + FlashAttention は context 1K + Megatron より 30% 速く、perplexity 18.2 → **17.5**（0.7 改善, Table 3）。
    - **Long Document classification** (micro F1, Table 4): MIMIC-III は seq 512 で 52.8 → seq 16K で **57.1**（+4.3）、ECtHR は 72.2 → seq 8K で **80.7**（+8.5、本文の "6.4 points of lift" は2タスク平均）。
    - **LRA** (Table 2): Transformer 平均 59.3 / FlashAttention 59.8 / block-sparse 59.6。速度は標準比 **2.4×（FlashAttention）, 2.8×（block-sparse）**。Linformer 2.5×、Performer 1.8× などより block-sparse 版は同等以上に速い。
    - **Path-X (seq 16K)**: 既存 Transformer 全滅 → FlashAttention で初の non-random、**61.4%**。**Path-256 (seq 64K)**: block-sparse FlashAttention で **63.1%**（Table 5）。Path-X 上の block-sparse は 56.0。
    - **Attention 単体ベンチ** (A100 40GB): seq 128–2K で標準 PyTorch 比 最大 **3× 高速**、メモリ最大 **20× 効率**。系列長 1024 付近で近似 attention に交差するが、block-sparse 版は全長で最速。
- **貢献**:
    1. tiling + recomputation で attention を1つの CUDA kernel に融合した exact algorithm。
    2. IO 複雑度の上限と下限（exact attention の HBM アクセス下界）。
    3. block-sparse 拡張と、Path-X / Path-256 を Transformer で初めて解けることを示した実証。
    4. 実装をオープンソース化（github.com/HazyResearch/flash-attention）。

## Takeaway（自分にとっての要点）

- **「FLOPs を減らすより HBM アクセスを減らす」**: attention は memory-bound なので、FLOPs を増やしてでも HBM 往復を減らした方が速い。これは attention に限らず、reduction や elementwise が混ざる layer 全般に効く設計原則。
- **online softmax + tiling は exact**: 近似ではない。$m_i, \ell_i$ を持ち回ってブロックごとに再正規化すれば数学的に同一の softmax(QK^⊤)V が出る。「近似 attention は使いたくないが長文脈は欲しい」という用途に直撃する。
- **Backward の recomputation が "checkpointing for speed"**: 既存の gradient checkpointing は「メモリのために速度を犠牲にする」のが定石だったが、IO の枠で考えると recompute の方が HBM 往復より速い、という逆転がある。Backward でも HBM 削減効果が出る点が肝。
- **Lower bound の意味**: $\Theta(N^2 d^2 / M)$ は SRAM サイズ $M$ をパラメータに残した形で漸近最適。「もうこの方向では伸びしろが小さい」のと「ハードを変える（SRAM を増やす / multi-GPU の IO を考える）と話が変わる」の両方を示唆。
- **block sparsity との直交性**: 近似 attention 自体は否定されておらず、「IO-aware に実装し直せば速くなる」という土台を提供する形。butterfly sparsity を採用しているのは著者らの過去研究の流用。
- **Path-X 16K / Path-256 64K** という、それまで S4 などの非 Transformer でしか解けていなかったタスクを vanilla Transformer + 長系列で解いた、というインパクトが大きい。長系列が解けない理由は「アーキの限界」というより「実装の限界」だった可能性。
- ブロックサイズ $B_c = \lceil M/(4d) \rceil$, $B_r = \min(B_c, d)$ という具体公式まで提示されていて、再現実装の指針として有用。
- IO-aware の思想は「コンパイラ（Halide 的）に attention を書いて自動で IO-aware CUDA に落としたい」という future work に直結。これが後の Triton ベース実装や FlashAttention-2/3 の系譜になっていく（本論文では未言及だが文脈として重要）。

## Critical Thoughts（評価・疑問）

- **強み**:
    - 「exact」「同じ model definition」「同じ perplexity」を維持したまま速度・メモリを劇的に改善できる点が圧倒的。論文中も「perplexity 一致」を明示している（GPT-2）。近似手法と違いユーザは数値が変わる心配をしなくてよい。
    - 理論（IO 上限・下限）と実装（CUDA kernel）と応用（BERT/GPT-2/LRA/Path-X）の三層が揃っていて、systems 論文として完成度が高い。
    - Path-X / Path-256 を Transformer で解けたという qualitative jump は数字以上に印象的。LRA の Pathfinder で苦戦していた Transformer が、純粋に「長い系列を流せるようになっただけ」で勝つことを示した。
    - block-sparse 版が LRA で標準 attention と同等精度かつ最速、という結果は近似系の正当な置き換え候補。
- **弱み / 疑問**:
    - **限界として著者自身が認めている点**（§ Limitations and Future Directions）:
        - attention アルゴリズムごとに CUDA を手書きする必要があり、PyTorch のような高水準言語から書きたい（Halide 的コンパイラが欲しい）。
        - 実装は GPU アーキ間で必ずしも transferrable でない（A100 で書かれたものを他世代に持っていくのは別物）。
        - IO-aware は attention 以外にも広げる余地があるが本論文ではやっていない。
        - 単一 GPU では定数項まで含めて最適だが、multi-GPU の IO（GPU 間通信）は未対応。
    - 評価の主軸が A100。3090 や T4 の図はあるが、TPU や AMD GPU など別アーキでの効果は議論されない。SRAM/HBM 比が違うハードでは利得は変わるはず。
    - head dim $d$ が大きくなる（例えば $d \ge 256$）と $d^2 / M$ が 1 に近づき、理論的優位が縮む。本論文の実験は $d=64$ 中心で、その点の感度分析は手薄。
    - block-sparse の sparsity pattern として butterfly を採用しているが、pattern の選び方への感度（content-based sparsity と比べての劣化）は本文では追っていない。
    - Path-X 61.4% は確かに better-than-chance だが S4 系の 88% 等には遠い。「Transformer でも解けた」事実の重みはあるが、ベンチマークの SOTA としては弱い（本論文の主目的ではないので妥当だが）。
    - Long Document の MIMIC-III は seq 16K で 57.1、seq 8K で 56.4、seq 1024 で逆に 50.7 と非単調。本文も「distribution shift の可能性」と認めているが、長系列＝常に良いというストーリーをやや弱める。
    - 公平比較として、近似 attention は標準的なリファレンス実装で測られている。もし近似手法も同じくらい IO-aware に書き直されたら？ block-sparse FlashAttention 自身がその一例だが、Linformer/Performer の IO-aware 実装との比較はない。
- **次に試したいこと**:
    - head dim $d \in \{64, 128, 256\}$ で speedup を測って $d^2/M$ がボトルネックになる点を実測する。
    - online softmax の数値安定性を fp16/bf16 で詳しく測る（特に長系列で $m_i$ の更新誤差が積もるかどうか）。
    - block-sparse の mask を learned routing（MoE 的 / Mixture-of-Depths 的）にして dynamic sparsity に拡張、IO 複雑度との折り合いを見る。
    - decode 時（KV cache 増分、$Q$ が 1 行）の IO 解析。本論文は training 中心で inference の incremental case はあまり扱っていない。
    - multi-GPU IO-aware の attention 実装（GPU 間 ring + tiling）を考えると、後に出る Ring Attention の前段として整理し直せる。

## Notes / Quotes

- "We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM." (abstract)
- A100 spec: 40–80 GB HBM @ 1.5–2.0 TB/s, **192 KB SRAM per SM × 108 SMs @ ~19 TB/s**（§Background）。SRAM は HBM より 1桁速いが何桁も小さい、というのが設計の前提。
- Algorithm 1 (Algorithm 0 is standard): outer loop は $K, V$ ブロック、inner loop は $Q, O$ ブロック。ブロックサイズ $B_c = \lceil M/(4d) \rceil$, $B_r = \min(B_c, d)$。
- 統計量更新: $m_i^{\mathrm{new}} = \max(m_i, \tilde m_{ij})$, $\ell_i^{\mathrm{new}} = e^{m_i - m_i^{\mathrm{new}}} \ell_i + e^{\tilde m_{ij} - m_i^{\mathrm{new}}} \tilde \ell_{ij}$、$O_i$ も同じ再正規化で更新。これが「online softmax」の本体。
- Theorem 2: standard $\Theta(Nd + N^2)$ vs FlashAttention $\Theta(N^2 d^2 M^{-1})$。
- Proposition 3 (lower bound): "no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes $M \in [d, Nd]$"。
- 著者自身の限界宣言: "Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention implementation... Implementations may also not be transferrable across GPU architectures." (§Discussion)
- Path-X / Path-256 はそれぞれ 128×128 / 256×256 の白黒画像中の2点が経路で結ばれているかを判定するタスクで、画像をピクセル単位で系列入力する。Path-256 の方が path が短いので Path-X より易しい、と著者は注釈している。

## Related Papers

- Milakov & Gimelshein 2018, "Online normalizer calculation for softmax" — online softmax の元ネタ。
- Rabe & Staats 2021, "Self-attention does not need $O(n^2)$ memory" — sequential な online softmax + checkpointing で attention を low memory に。FlashAttention は IO 視点でこれを再構成し速度も出す。
- Aggarwal & Vitter 1988, "Input/output complexity of sorting" — IO 複雑度モデルの古典。
- Williams, Waterman, Patterson 2009, "Roofline" — memory-bound / compute-bound の判定。
- Kitaev et al. 2020 Reformer, Wang et al. 2020 Linformer, Choromanski et al. 2020 Performer, Beltagy et al. 2020 Longformer, Zaheer et al. 2020 BigBird, Daras et al. 2020 SMYRF, Katharopoulos et al. 2020 Linear Attention — LRA / Path-X で並ぶ近似 attention 群。
- Child et al. 2019, "Sparse Transformer"; Dao et al. 2021 Pixelated Butterfly — butterfly sparsity の出典。
- Griewank & Walther 2008; Chen et al. 2016 — gradient checkpointing の出典。
- Tay et al. 2020, Long Range Arena; Gu et al. 2022, S4 — Path-X benchmark とそこを先に解いていた非 Transformer モデル。
- Devlin et al. 2018 BERT; Radford et al. 2019 GPT-2; Shoeybi et al. 2019 Megatron-LM; Wolf et al. 2020 HuggingFace Transformers; MLPerf 1.1 — 速度比較 baseline。
