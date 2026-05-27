# Attention Is All You Need（self-attention のみで系列変換を行う Transformer の提案）

- arXiv: https://arxiv.org/abs/1706.03762
- 一次ソース: ../papers/arXiv-1706.03762v7/
- 正規ノート: ../notes/arXiv-1706.03762v7.md

---

## 一言で言うと

RNN や convolution を使わず、attention mechanism だけで encoder-decoder 型の sequence transduction model を作れるかを問う論文である。提案する Transformer は、WMT 2014 English-to-German で 28.4 BLEU、WMT 2014 English-to-French で 41.8 BLEU を報告し、著者は「more parallelizable」「significantly less time to train」と主張する（`ms.tex` abstract, Table `tab:wmt-results`）。

## 何を議論する論文か

- **問題設定**: 入力系列を出力系列へ変換する sequence transduction、特に machine translation を対象に、既存の encoder-decoder モデルで中心的だった recurrent layer や convolutional layer の代替を検討する。
- **対象範囲 / 仮定**: 入力 token と出力 token を learned embedding に変換し、decoder は previously generated symbols を追加入力として使う auto-regressive model である。系列順序は recurrence/convolution から得られないため、positional encoding を embedding に足す。
- **既存研究との差分**: 既存の有力モデルは RNN/LSTM/GRU、または Extended Neural GPU、ByteNet、ConvS2S のような convolutional model を使う。attention は多くの場合 recurrent network と併用されていたが、この論文は「entirely on self-attention」「without using sequence-aligned RNNs or convolution」として Transformer を位置づける（`background.tex`）。
- **この論文で答えたい問い**: recurrence と convolution を捨てても、self-attention と position-wise feed-forward layer だけで翻訳品質を保ち、かつ訓練の並列化と長距離依存の扱いを改善できるか。

## 背景と前提

- RNN 系モデルでは位置 $t$ の hidden state $h_t$ が $h_{t-1}$ と入力位置 $t$ に依存するため、training example 内の系列方向の並列化が難しい。Introduction はこの「inherently sequential nature」が長い系列で問題になると述べる。
- Convolutional model は全位置の hidden representation を並列に計算できるが、任意の 2 位置を関係づける操作数は ConvS2S で距離に対して線形、ByteNet で対数的に増えると説明される（`background.tex`）。
- Self-attention は、1 つの系列内の異なる位置を関係づけ、sequence representation を計算する attention mechanism である。論文は reading comprehension、abstractive summarization、textual entailment、sentence representation での先行利用に触れるが、sequence transduction model 全体を self-attention のみにした点を新規性としている。
- Why Self-Attention 節は、layer type を比較する軸として、per-layer complexity、minimum number of sequential operations、maximum path length を置く。Table `tab:op_complexities` では Self-Attention が $O(n^2 \cdot d)$、$O(1)$、$O(1)$、Recurrent が $O(n \cdot d^2)$、$O(n)$、$O(n)$ と整理される。
- 著者は self-attention layer が recurrent layer より computational complexity 上速い条件を、sequence length $n$ が representation dimensionality $d$ より小さい場合と述べる。これは word-piece や byte-pair representations を使う state-of-the-art machine translation で多い、と説明される（`why_self_attention.tex`）。

## 提案手法

### コアアイデア

Transformer は encoder-decoder 構造を保つが、encoder と decoder の各層を stacked self-attention と point-wise, fully connected layer で構成する（Figure `fig:model-arch`）。Encoder は入力系列 $(x_1,\ldots,x_n)$ を continuous representations $\mathbf{z}=(z_1,\ldots,z_n)$ に写し、decoder は $\mathbf{z}$ と既知の出力 token を使って $(y_1,\ldots,y_m)$ を 1 要素ずつ生成する。

Encoder は $N=6$ identical layers からなり、各 layer は multi-head self-attention と position-wise feed-forward network の 2 sub-layer を持つ。Decoder も $N=6$ layers だが、masked self-attention、encoder-decoder attention、position-wise feed-forward network の 3 sub-layer を持つ。各 sub-layer の周囲には residual connection があり、その後に layer normalization を置く。TeX の表記では $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$ で、全 sub-layer と embedding layer の出力次元は $d_{\text{model}}=512$ である。

Attention は query、key、value の対応から value の weighted sum を返す関数として定義される。Transformer ではこれを multi-head 化し、encoder self-attention、decoder masked self-attention、encoder-decoder attention の 3 つの用途で使う。Decoder の masked self-attention では、auto-regressive property を保つため、softmax 入力の illegal connections を $-\infty$ にする（`model_architecture.tex`）。

### 重要な定義・数式

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**式の意味**: Scaled Dot-Product Attention の定義である。Query と key の dot product を $\sqrt{d_k}$ で割り、softmax によって value への重みを作り、その重み付き和を出力する。

**記号の定義**:
- $Q$ ... queries をまとめた行列
- $K$ ... keys をまとめた行列
- $V$ ... values をまとめた行列
- $d_k$ ... query と key の次元
- $QK^T$ ... query と key の compatibility score

**この論文での役割**: Transformer の attention sub-layer の基本演算である。著者は、$q$ と $k$ の各成分が平均 0、分散 1 の独立確率変数なら $q \cdot k$ の分散が $d_k$ になるため、large $d_k$ で softmax が小さい勾配領域に入ることを避ける目的で $1/\sqrt{d_k}$ scaling を入れる、と説明する（`model_architecture.tex` footnote）。

$$
\begin{aligned}
\mathrm{MultiHead}(Q,K,V) &= \mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O \\
\mathrm{head}_i &= \mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\end{aligned}
$$

**式の意味**: Query、key、value を head ごとに learned linear projections で低次元へ写し、それぞれで attention を並列に計算し、結果を結合して再射影する。

**記号の定義**:
- $h$ ... attention head の数。この論文の base model では $h=8$
- $W_i^Q,W_i^K,W_i^V$ ... head $i$ 用の projection matrices
- $W^O$ ... concatenated heads を $d_{\text{model}}$ 次元へ戻す projection matrix
- $\mathrm{head}_i$ ... $i$ 番目の Scaled Dot-Product Attention の出力

**この論文での役割**: 単一 head の weighted average だけでは異なる位置・表現部分空間への同時注意が制限されるため、著者は multi-head attention によって「different representation subspaces at different positions」へ同時に attend できると主張する。Base model では $h=8$、$d_k=d_v=d_{\text{model}}/h=64$ で、計算量は full dimensionality の single-head attention と同程度とされる。

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

**式の意味**: 各位置に同一に適用される position-wise feed-forward network の定義である。2 つの線形変換の間に ReLU activation を挟む。

**記号の定義**:
- $x$ ... ある位置の representation
- $W_1,W_2,b_1,b_2$ ... feed-forward network の learned parameters
- $\max(0,\cdot)$ ... ReLU activation
- $d_{\text{model}}$ ... 入出力次元。この論文では 512
- $d_{ff}$ ... inner-layer 次元。この論文では 2048

**この論文での役割**: Attention が位置間の情報交換を担う一方、FFN は各位置で同じ非線形変換を行う。Encoder と decoder の各 layer に含まれる基本 sub-layer であり、kernel size 1 の 2 つの convolution とも説明される。

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin(pos / 10000^{2i/d_{\text{model}}}) \\
PE_{(pos,2i+1)} &= \cos(pos / 10000^{2i/d_{\text{model}}})
\end{aligned}
$$

**式の意味**: Sinusoidal positional encoding の定義である。偶数次元には sine、奇数次元には cosine を使い、各次元が異なる周波数の sinusoid になる。

**記号の定義**:
- $pos$ ... token の系列内位置
- $i$ ... positional encoding の dimension index
- $d_{\text{model}}$ ... embedding と positional encoding の次元
- $PE_{(pos,2i)}$, $PE_{(pos,2i+1)}$ ... 位置 $pos$ に対応する encoding の各成分

**この論文での役割**: Transformer は recurrence も convolution も持たないため、token order を明示的に注入する必要がある。著者は、任意の fixed offset $k$ について $PE_{pos+k}$ が $PE_{pos}$ の linear function として表せるので relative positions を学びやすい、という仮説から sinusoidal 版を選ぶ。Table `tab:variations` row (E) では learned positional embeddings と nearly identical results と報告される。

$$
lrate = d_{\text{model}}^{-0.5}\cdot
\min({step\_num}^{-0.5},\ {step\_num}\cdot {warmup\_steps}^{-1.5})
$$

**式の意味**: Adam optimizer と組み合わせて使う learning rate schedule である。最初は線形に増加し、その後は step number の inverse square root に比例して減少する。

**記号の定義**:
- $lrate$ ... learning rate
- $d_{\text{model}}$ ... model representation dimension
- $step\_num$ ... training step number
- $warmup\_steps$ ... warmup の長さ。この論文では 4000

**この論文での役割**: 実験結果を支える training regime の一部である。論文は Adam の $\beta_1=0.9$、$\beta_2=0.98$、$\epsilon=10^{-9}$ とともにこの schedule を使う（`training.tex`）。

### 実装 / アルゴリズム上の要点

- 入力 token と出力 token は learned embeddings に変換され、embedding layer では重みを $\sqrt{d_{\text{model}}}$ 倍する。入力 embedding、出力 embedding、pre-softmax linear transformation は同じ weight matrix を共有する。
- Encoder は 6 層で、各層は multi-head self-attention と position-wise FFN からなる。各 sub-layer は residual connection と layer normalization に包まれる。
- Decoder は 6 層で、masked decoder self-attention、encoder-decoder attention、position-wise FFN を持つ。Masked self-attention は future position を参照しないように softmax 入力で $-\infty$ を入れる。
- Attention の 3 用途は、encoder-decoder attention、encoder self-attention、decoder self-attention である。Encoder-decoder attention では query が previous decoder layer、key/value が encoder output から来る。
- Base model は $N=6$、$d_{\text{model}}=512$、$d_{ff}=2048$、$h=8$、$d_k=d_v=64$、$P_{drop}=0.1$、$\epsilon_{ls}=0.1$、100K steps、65M parameters である（Table `tab:variations`）。
- Big model は $N=6$、$d_{\text{model}}=1024$、$d_{ff}=4096$、$h=16$、$d_k=d_v=64$、$P_{drop}=0.3$、$\epsilon_{ls}=0.1$、300K steps、213M parameters である（Table `tab:variations`）。Big 行の $d_k,d_v,\epsilon_{ls}$ 欄は空欄だが、caption の「Unlisted values are identical to those of the base model」に従う。なお WMT 2014 English-to-French の Transformer (big) は $P_{drop}=0.1$ を使い、0.3 ではないと `results.tex` が明記する。
- Training は 8 NVIDIA P100 GPUs の 1 machine で行う。Base model は 1 step 約 0.4 秒で 100,000 steps または 12 hours、big model は 1 step 1.0 秒で 300,000 steps または 3.5 days と書かれている。
- Training batch は approximate sequence length でまとめられ、各 batch は約 25000 source tokens と約 25000 target tokens を含む。推論では base は last 5 checkpoints、big は last 20 checkpoints を averaging し、beam size 4、length penalty $\alpha=0.6$、maximum output length は input length + 50 を使う。

## 実験・結果

- **データセット / ベンチマーク**: WMT 2014 English-German は約 4.5M sentence pairs、shared source-target vocabulary 約 37000 tokens の byte-pair encoding。WMT 2014 English-French は 36M sentences、32000 word-piece vocabulary。追加評価として Penn Treebank の WSJ portion を使った English constituency parsing を扱い、WSJ only は約 40K training sentences、semi-supervised は high-confidence and BerkleyParser corpora の約 17M sentences を用いる。
- **比較対象 / baseline**: Machine translation では ByteNet、Deep-Att + PosUnk、GNMT + RL、ConvS2S、MoE、および Deep-Att + PosUnk Ensemble、GNMT + RL Ensemble、ConvS2S Ensemble と比較する（Table `tab:wmt-results`）。Constituency parsing では Vinyals & Kaiser et al.、Petrov et al.、Zhu et al.、Dyer et al.、Huang & Harper、McClosky et al.、Luong et al. などと比較する（Table `tab:parsing-results`）。
- **指標**: Translation quality は BLEU、training cost は FLOPs。Ablation は English-to-German development set newstest2013 上の PPL(dev)、BLEU(dev)、params。Parsing は WSJ Section 23 F1。
- **主な結果**: WMT 2014 EN-DE では Transformer (base model) が 27.3 BLEU、Transformer (big) が 28.4 BLEU。Table `tab:wmt-results` 上の既存最高 EN-DE は ConvS2S Ensemble 26.36 BLEU で、本文は big が best previously reported models including ensembles を more than 2.0 BLEU 上回ると述べる。
- **主な結果**: WMT 2014 EN-FR では Transformer (base model) が 38.1 BLEU、Table `tab:wmt-results` と abstract の Transformer (big) が 41.8 BLEU とする。Table 上の既存 single model 最高は MoE 40.56 BLEU、ensemble 最高は ConvS2S Ensemble 41.29 BLEU。EN-FR の big model は dropout rate $P_{drop}=0.1$ を使う。一方、`results.tex` 本文には big 41.0 BLEU とも書かれており TeX 内に不一致がある。このノートでは表と abstract の 41.8 を主値として扱い、不一致を明記する。
- **主な結果**: Training Cost は Transformer base が $3.3\cdot10^{18}$ FLOPs、Transformer big が $2.3\cdot10^{19}$ FLOPs。ConvS2S は EN-DE $9.6\cdot10^{18}$、MoE は EN-DE $2.0\cdot10^{19}$、GNMT + RL Ensemble は EN-DE $1.8\cdot10^{20}$ と表に示される。
- **主な結果**: Ablation では base が PPL 4.92、BLEU 25.8、65M params。Head 数の row (A) は $h=1$ で BLEU 24.9、$h=4$ で 25.5、$h=16$ で 25.8、$h=32$ で 25.4。Row (B) では $d_k=16$ が BLEU 25.1、$d_k=32$ が 25.4 で、著者は attention key size を小さくすると品質が落ちると述べる。
- **主な結果**: English constituency parsing では Transformer (4 layers) が WSJ only, discriminative で WSJ 23 F1 = 91.3、semi-supervised で 92.7。WSJ only では Dyer et al. (2016) の 91.7 に次ぎ、semi-supervised では表中の semi-supervised baseline（最高 92.1）を上回る。ただし Table `tab:parsing-results` には multi-task の Luong et al. (2015) 93.0 と generative の Dyer et al. (2016) 93.3 も載っており、訓練設定別に読む必要がある。
- **著者が主張する貢献**: Abstract と Conclusion は、Transformer を recurrence と convolution を完全に捨てた attention-based architecture として提示し、翻訳で new state of the art、訓練時間の短縮、English constituency parsing への generalization を主張する。

## 妥当性と限界

- **この主張を支える根拠**: Table `tab:op_complexities` は self-attention の sequential operations と maximum path length が $O(1)$ であることを、recurrent の $O(n)$ と対比する。Introduction と Results はこの並列化上の利点が訓練時間の短縮につながるという筋立てで、12 hours on eight P100 GPUs（base）と 3.5 days（big）を示す。
- **この主張を支える根拠**: Table `tab:wmt-results` は WMT 2014 EN-DE/EN-FR での BLEU と FLOPs を既存モデルと並べる。EN-DE では big 28.4 BLEU が既存 ensemble を含む表中の値より高く、base 27.3 BLEU も既存 published models and ensembles を上回ると本文が述べる。
- **この主張を支える根拠**: Table `tab:variations` は head 数、$d_k$、layer 数、$d_{\text{model}}$、$d_{ff}$、dropout、label smoothing、positional encoding の変更が PPL/BLEU に与える影響を示す。ただし多くは base model から 1 軸ずつ変える ablation である。
- **著者が認めている limitations / future work**: Why Self-Attention 節は、very long sequences では self-attention を neighborhood size $r$ に制限することで計算性能を改善できるかもしれないと述べ、これを future work とする。Conclusion でも images、audio、video のような large inputs and outputs に対して local, restricted attention mechanisms を調べる予定と書く。
- **著者が認めている limitations / future work**: Conclusion は「Making generation less sequential」も research goal として挙げる。これは decoder が auto-regressive に 1 要素ずつ出力する設計であることと対応する。
- **読者として注意すべき点**: TeX にある実験は二つの machine translation task と English constituency parsing に限られる。より広い NLP タスクや非テキスト modality での実証は、この論文内では future work として扱われる。
- **読者として注意すべき点**: Attention visualization は layer 5 of 6 の long-distance dependency、anaphora resolution、sentence structure に関する例を示すが、qualitative evidence である。Head が「clearly learned to perform different tasks」と本文は述べるものの、定量的な head 機能分類までは TeX 中にない。
- **追加で確認したい実験 / 疑問**: Sinusoidal positional encoding は learned positional embeddings と nearly identical results で、著者は長い系列への extrapolation 可能性を理由に選ぶが、訓練時より長い系列での比較実験は TeX 中には明示されていない。
- **追加で確認したい実験 / 疑問**: Restricted self-attention の neighborhood size $r$ を変えたときの BLEU、FLOPs、maximum path length の実測は、この論文の表にはない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Sequence transduction**: 入力 token 系列を出力 token 系列に写す問題群。本文では language modeling や machine translation の文脈で導入され、実験の中心は WMT 2014 translation である。
- **Transformer**: Recurrence と convolution を使わず、multi-headed self-attention と position-wise FFN によって encoder-decoder を構成するモデル名。
- **Encoder-decoder structure**: Encoder が $(x_1,\ldots,x_n)$ を $\mathbf{z}=(z_1,\ldots,z_n)$ に写し、decoder が $\mathbf{z}$ と既知の出力を使って $(y_1,\ldots,y_m)$ を生成する構造。
- **Auto-regressive**: Decoder が次の token を出す際に、previously generated symbols を追加入力として消費する性質。Transformer decoder は future positions を mask してこれを保つ。
- **Self-attention / intra-attention**: 1 つの系列内の異なる位置を関係づけ、系列の representation を計算する attention mechanism。この論文では encoder と decoder の主要な位置間通信として使われる。
- **Encoder-decoder attention**: Decoder 側の query が encoder output の key/value に attend する attention。従来の sequence-to-sequence attention に相当する役割を担う。
- **Query, key, value**: Attention function の入力。Query と key から compatibility を計算し、その重みによって value を weighted sum する。
- **Scaled Dot-Product Attention**: Dot-product attention に $1/\sqrt{d_k}$ の scaling factor を入れたもの。大きい $d_k$ で softmax が極端になり勾配が小さくなる問題を抑えるためと説明される。
- **Multi-head attention**: $Q,K,V$ を head ごとに異なる learned projections へ写して attention を並列実行し、結果を concatenate する機構。Base model では 8 heads。
- **Position-wise Feed-Forward Network**: 各位置に別々かつ同一に適用される 2 層の fully connected network。入力・出力次元は $d_{\text{model}}=512$、inner-layer は $d_{ff}=2048$。
- **Positional encoding**: Recurrence/convolution がない Transformer に順序情報を入れるため、encoder/decoder stack の bottom で embedding に加える encoding。この論文は sinusoidal 版を採用する。
- **Byte-pair encoding / word-piece**: 翻訳データを subword token に分割する方法。EN-DE では shared source-target vocabulary 約 37000 tokens の byte-pair encoding、EN-FR では 32000 word-piece vocabulary。
- **BLEU**: Translation quality の評価指標。Table `tab:wmt-results` の主指標。
- **PPL(dev)**: Model variations で使われる development set perplexity。Caption は byte-pair encoding による per-wordpiece perplexities であり、per-word perplexities と比較すべきでないと注意する。
- **F1**: English constituency parsing の WSJ Section 23 評価指標。
- **Restricted self-attention**: 入力系列全体ではなく neighborhood size $r$ のみを見る self-attention。Table `tab:op_complexities` では $O(r\cdot n\cdot d)$、maximum path length $O(n/r)$ とされ、future work として扱われる。

## 読む順番の提案

- まず `ms.tex` の abstract と Conclusion を読み、著者が Transformer を「based solely on attention mechanisms」「first sequence transduction model based entirely on attention」と位置づける主張を押さえる。正規ノートでは Summary の問題設定と貢献に対応する。
- 次に `introduction.tex` と `background.tex` を読み、RNN の sequential nature、ConvS2S/ByteNet の距離に応じた操作数、self-attention の先行利用を確認する。正規ノートでは Takeaway の「並列化できることが主要な動機」に対応する。
- `model_architecture.tex` は Figure `fig:model-arch`、Scaled Dot-Product Attention、Multi-Head Attention、Position-wise FFN、Embeddings and Softmax、Positional Encoding の順に読む。正規ノートの手法箇条書きと数式の根拠はほぼここにある。
- `why_self_attention.tex` の Table `tab:op_complexities` を読む。Self-Attention、Recurrent、Convolutional、restricted Self-Attention の complexity、sequential operations、maximum path length が、この論文の妥当性説明の中心である。
- `training.tex` でデータセット、batching、8 NVIDIA P100 GPUs、Adam、learning rate schedule、dropout、label smoothing を確認する。正規ノートの訓練レシピに対応する。
- `results.tex` は Table `tab:wmt-results`、Table `tab:variations`、Table `tab:parsing-results` を優先して読む。EN-FR の 41.8 と 41.0 の TeX 内不一致もここで確認する。
- 最後に `visualizations.tex` を読み、attention head の long-distance dependency、anaphora resolution、sentence structure に関する qualitative evidence を確認する。これは主張の補助であり、定量実験とは分けて読む。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1706.03762v7/`
- 正規ノート: `notes/arXiv-1706.03762v7.md`
