# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（Vision Transformer による大規模画像認識）

- arXiv: https://arxiv.org/abs/2010.11929
- 一次ソース: ../papers/arXiv-2010.11929v2/
- 正規ノート: ../notes/arXiv-2010.11929v2.md

---

## 一言で言うと

画像を固定サイズの patch 列として扱い、標準的な Transformer encoder をほぼそのまま画像分類に適用する Vision Transformer（ViT）を提案する論文。ViT は CNN より画像固有の inductive bias が少ないため小さな事前学習データでは弱いが、ImageNet-21k や JFT-300M のような大規模データで事前学習すると複数の画像認識ベンチマークで state-of-the-art CNN に並ぶか上回り、事前学習 compute も少ないと著者は主張する。

## 何を議論する論文か

- **問題設定**: NLP で成功した Transformer を、CNN を主役にせず画像分類へ直接使えるかを調べる。具体的には、画像を patch の sequence に変換し、supervised pre-training と downstream fine-tuning で表現学習能力を評価する。
- **対象範囲 / 仮定**: 主な対象は image classification と transfer learning。事前学習には ImageNet、ImageNet-21k、JFT-300M を使い、下流評価には ImageNet、ImageNet-ReaL、CIFAR-10/100、Oxford-IIIT Pets、Oxford Flowers-102、VTAB などを使う。detection や segmentation は結論で future work として挙げられており、本文の主張範囲ではない。
- **既存研究との差分**: 既存の vision self-attention は、CNN に attention を併用する、畳み込みの一部を specialized attention pattern で置き換える、pixel や小 patch に attention を使う、という方向が多かった。この論文は「fewest possible modifications」で標準 Transformer を画像 patch 列へ適用し、大規模事前学習で SOTA CNN と比較する点が差分である。
- **この論文で答えたい問い**: 純粋な Transformer は画像分類で CNN 依存を減らせるのか。CNN の locality や translation equivariance という inductive bias が少ないことは、データ規模を増やせば補えるのか。性能と pre-training compute の trade-off は ResNet や hybrid と比べてどうか。

## 背景と前提

- Transformer は 1D sequence の token embeddings を入力にし、multiheaded self-attention（MSA）と MLP block を重ねる。NLP では大規模 corpus で pre-train し、task-specific dataset で fine-tune する流れが標準になっている、という前提から論文は始まる。
- CNN は locality、two-dimensional neighborhood structure、translation equivariance を各層に持つ。ViT はこの画像固有の inductive bias をほとんど持たず、2D 構造を手で入れる箇所は patch extraction と fine-tuning 時の position embedding の 2D interpolation に限られる、と `03_method.tex` の "Inductive bias" 段落で説明されている。
- 本論文での patch は、画像 $\mathbf{x}\in\mathbb{R}^{H\times W\times C}$ を $P\times P$ に切ったものを flatten した単位である。例えば $P=16$ のモデルは `ViT-L/16` のように表記され、patch size が小さいほど sequence length が増え、計算量が大きくなる。
- 主な CNN baseline は Big Transfer（BiT-L, ResNet152x4）と Noisy Student（EfficientNet-L2）。論文内の ResNet baseline は Batch Normalization を Group Normalization に置き換え、standardized convolutions を使う "ResNet (BiT)" として扱われる。
- 評価は fine-tuning accuracy が中心だが、一部の大規模比較では計算コストを抑えるため、frozen representation に対する regularized least-squares regression による linear few-shot accuracy も使う。

## 提案手法

### コアアイデア

ViT は画像を 2D grid として畳み込むのではなく、flattened patches の 1D sequence として Transformer encoder に渡す。入力画像を $N=HW/P^2$ 個の patch に分け、各 patch を trainable linear projection $\mathbf{E}$ で $D$ 次元に写す。BERT の `[class]` token と同様に学習可能な classification token を先頭に付け、learnable 1D position embeddings を加算して、pre-LayerNorm 型の Transformer encoder に入力する。

分類には最終層の classification token の状態 $\mathbf{z}_L^0$ を使う。pre-training 時の classification head は hidden layer を 1 つ持つ MLP、fine-tuning 時は single linear layer である。下流データセットへ移すときは pre-trained prediction head を外し、zero-initialized $D\times K$ feedforward layer を付ける。ここで $K$ は downstream classes の数である。

モデルサイズは Table 1（`tbl:models`）で定義される。ViT-Base は 12 layers、hidden size $D=768$、MLP size 3072、12 heads、86M params。ViT-Large は 24 layers、$D=1024$、MLP size 4096、16 heads、307M params。ViT-Huge は 32 layers、$D=1280$、MLP size 5120、16 heads、632M params。

### 重要な定義・数式

$$
\mathbf{z}_0 = [ \mathbf{x}_\text{class}; \, \mathbf{x}^1_p \mathbf{E}; \, \mathbf{x}^2_p \mathbf{E}; \cdots; \, \mathbf{x}^{N}_p \mathbf{E} ] + \mathbf{E}_{pos},
\qquad
\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D},\,
\mathbf{E}_{pos} \in \mathbb{R}^{(N + 1) \times D}
$$

**式の意味**: `03_method.tex` の Eq. (1) で、画像 patch を Transformer に入れる初期 sequence $\mathbf{z}_0$ として組み立てる式である。flatten した各 patch を同じ線形射影 $\mathbf{E}$ で $D$ 次元へ写し、先頭に classification token を付け、position embeddings を加える。

**記号の定義**:
- $\mathbf{x}_\text{class}$ ... 学習可能な classification token。出力側の状態 $\mathbf{z}_L^0$ が画像表現として使われる。
- $\mathbf{x}^i_p$ ... $i$ 番目の flattened 2D image patch。
- $\mathbf{E}$ ... flattened patch を $D$ 次元へ写す trainable linear projection。
- $P$ ... patch の一辺の pixel 数。`ViT-L/16` なら $P=16$。
- $H,W$ ... 元画像の解像度。
- $C$ ... 入力画像の channel 数。
- $D$ ... Transformer の latent vector size。
- $N=HW/P^2$ ... patch 数であり、Transformer の有効な input sequence length。
- $\mathbf{E}_{pos}$ ... learnable 1D position embeddings。

**この論文での役割**: この式が ViT の最小限の画像化である。CNN の畳み込み特徴ではなく raw image patches を token として扱うため、論文の「pure transformer applied directly to sequences of image patches」という主張の中心になる。

$$
\begin{aligned}
\mathbf{z^\prime}_\ell &= \operatorname{MSA}(\operatorname{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, && \ell=1\ldots L \\
\mathbf{z}_\ell &= \operatorname{MLP}(\operatorname{LN}(\mathbf{z^\prime}_{\ell})) + \mathbf{z^\prime}_{\ell}, && \ell=1\ldots L \\
\mathbf{y} &= \operatorname{LN}(\mathbf{z}_L^0)
\end{aligned}
$$

**式の意味**: `03_method.tex` の Eq. (2) から Eq. (4) で、Transformer encoder block と最終画像表現を定義する。各層は LayerNorm の後に MSA、次に LayerNorm の後に MLP を置き、それぞれ residual connection を加える。

**記号の定義**:
- $\ell$ ... Transformer encoder の層番号。
- $L$ ... encoder layer 数。Table 1 では Base 12、Large 24、Huge 32。
- $\mathbf{z}_{\ell-1}$ ... 1 つ前の層から来る sequence representation。
- $\mathbf{z^\prime}_{\ell}$ ... MSA block と residual connection の後の中間表現。
- $\mathbf{z}_{\ell}$ ... MLP block と residual connection の後の層出力。
- $\mathbf{z}_L^0$ ... 最終層の classification token の状態。
- $\operatorname{LN}$ ... LayerNorm。各 block の前に適用される。
- $\operatorname{MSA}$ ... multiheaded self-attention。
- $\operatorname{MLP}$ ... 2 層の MLP block。GELU non-linearity を含む。
- $\mathbf{y}$ ... final classification token から得る画像表現。

**この論文での役割**: ViT が画像用に新しい encoder を作ったのではなく、standard Transformer encoder を使っていることを示す。残差接続と pre-LayerNorm を含むこの構成が、ViT-Base/Large/Huge の共通骨格になる。

$$
\begin{aligned}
[\mathbf{q}, \mathbf{k}, \mathbf{v}] &= \mathbf{z}\mathbf{U}_{qkv},
& \mathbf{U}_{qkv} &\in \mathbb{R}^{D \times 3D_h} \\
A &= \operatorname{softmax}\left(\mathbf{q}\mathbf{k}^{\top}/\sqrt{D_h}\right),
& A &\in \mathbb{R}^{N \times N} \\
\operatorname{SA}(\mathbf{z}) &= A\mathbf{v}
\end{aligned}
$$

**式の意味**: Appendix A（`sec:self_attention`）の Eq. (5) から Eq. (7) で、single-head self-attention を定義する。入力 sequence の各要素について query、key、value を作り、query-key similarity から attention weights $A$ を計算し、value の重み付き和を返す。

**記号の定義**:
- $\mathbf{z}\in\mathbb{R}^{N\times D}$ ... sequence 長 $N$、特徴次元 $D$ の入力。
- $N$ ... Appendix A での一般的な sequence length。ViT の初期入力では patch 数 $N$ に classification token が加わるため、Eq. (1) の $\mathbf{E}_{pos}$ は $(N+1)\times D$ になる。
- $\mathbf{q},\mathbf{k},\mathbf{v}$ ... query、key、value 表現。
- $\mathbf{U}_{qkv}$ ... 入力 $\mathbf{z}$ から query、key、value を作る projection matrix。
- $D_h$ ... 1 head あたりの次元。
- $A$ ... attention weight matrix。
- $A_{ij}$ ... sequence 内の $i$ 番目の要素が $j$ 番目の要素をどれだけ参照するかを表す attention weight。
- $\operatorname{SA}$ ... self-attention。

**この論文での役割**: ViT で patch 間の関係を global に統合する部分である。CNN の local convolution と異なり、self-attention は低層から画像全体の patch 関係を扱えるため、`04_experiments.tex` の attention distance 解析にもつながる。

$$
\operatorname{MSA}(\mathbf{z}) =
[\operatorname{SA}_1(z); \operatorname{SA}_2(z); \cdots ; \operatorname{SA}_k(z)] \, \mathbf{U}_{msa},
\qquad
\mathbf{U}_{msa} \in \mathbb{R}^{k \cdot D_h \times D}
$$

**式の意味**: Appendix A の Eq. (8) で、$k$ 個の self-attention heads を並列に走らせ、出力を concatenate してから線形射影する。

**記号の定義**:
- $\mathbf{z}$ ... MSA に入力される sequence representation。
- $k$ ... attention head 数。Table 1 では Base 12、Large/Huge 16。
- $D_h$ ... 1 head あたりの次元。
- $\operatorname{SA}_i$ ... $i$ 番目の self-attention head。
- $\mathbf{U}_{msa}$ ... concatenated heads を $D$ 次元に戻す射影行列。

**この論文での役割**: MSA は ViT-Base/Large/Huge に共通する Transformer block の self-attention 部分であり、Table 1 の heads と hidden size はこの block の model shape を決める値である。Appendix の Transformer shape ablation では depth、width、patch size などを動かし、性能との関係を調べている。

### 実装 / アルゴリズム上の要点

- step1: 入力画像を $P\times P$ の fixed-size patches に分け、flatten する。
- step2: 各 patch に trainable linear projection $\mathbf{E}$ を適用して patch embeddings にする。
- step3: sequence の先頭に learnable classification token を prepend し、learnable 1D position embeddings を加える。
- step4: pre-LayerNorm の Transformer encoder に通す。各層は MSA block と MLP block からなり、各 block の後に residual connection を置く。
- step5: $\mathbf{z}_L^0$ から画像表現 $\mathbf{y}$ を得て、classification head で分類する。
- step6: 大規模データで事前学習した後、downstream dataset ごとに head を付け替えて fine-tune する。fine-tuning で解像度を上げる場合、patch size は固定し、pre-trained position embeddings は original image 上の位置に従って 2D interpolation する。
- step7: Hybrid architecture では raw image patch の代わりに CNN feature map から patch を作る。patch size 1x1 の場合、feature map の spatial dimensions を flatten して Transformer dimension へ射影する。

実験設定として、§4.1 は pre-training に Adam（$\beta_1=0.9,\beta_2=0.999$）、batch size 4096、高い weight decay を使うと説明する。Appendix Table 3（`tbl:hparams-training`）では dataset/model ごとの値が整理され、weight decay は JFT-300M で 0.1、ImageNet-21k で 0.03、ImageNet で 0.3 である。fine-tuning は SGD with momentum 0.9、batch size 512。Appendix Table 3 では training resolution 224、warmup 10k steps とされ、Appendix Table 4（`tbl:hparams-finetuning`）では特記がなければ fine-tuning resolution 384、cosine learning rate decay、no weight decay、global norm 1 の grad clipping とされる。Table 2 の ImageNet 結果では ViT-L/16 を 512、ViT-H/14 を 518 resolution で fine-tune し、Polyak averaging factor 0.9999 を使う。

## 実験・結果

- **データセット / ベンチマーク**: 事前学習には ILSVRC-2012 ImageNet（1k classes、1.3M images）、ImageNet-21k（21k classes、14M images）、JFT（18k classes、303M high-resolution images）を使う。pre-training datasets は downstream test sets に対して de-duplicate される。downstream には ImageNet validation labels、ImageNet-ReaL、CIFAR-10/100、Oxford-IIIT Pets、Oxford Flowers-102、VTAB 19-task suite を使う。VTAB は各 task 1,000 training examples で、Natural、Specialized、Structured の 3 group に分かれる。
- **比較対象 / baseline**: Table 2 の主比較は BiT-L（ResNet152x4）と Noisy Student（EfficientNet-L2）。VTAB breakdown では BiT、VIVI、S4L と比較する。scaling study では 7 個の ResNet、6 個の ViT、5 個の hybrid を JFT-300M transfer で比較する。
- **指標**: downstream fine-tuning accuracy が中心で、Table 2 は 3 回 fine-tuning の mean と standard deviation を報告する。高速な途中評価には frozen representation から $\{-1,1\}^K$ target vectors へ写す regularized least-squares regression による linear few-shot accuracy を使う。compute は TPUv3-core-days、Appendix Table 6 では exaFLOPs も使う。
- **主な結果**: Table 2（`tbl:best_results`）では、JFT-300M で pre-train した ViT-H/14 が ImageNet 88.55、ImageNet-ReaL 90.72、CIFAR-10 99.50、CIFAR-100 94.55、Oxford-IIIT Pets 97.56、Oxford Flowers-102 99.68、VTAB 77.63 を出す。ViT-L/16（JFT）は Oxford Flowers-102 で 99.74 と最良で、ViT-H/14 より高い。

| Model | ImageNet | ImageNet ReaL | CIFAR-10 | CIFAR-100 | Pets | Flowers | VTAB | TPUv3-core-days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ViT-H/14, JFT | 88.55 | 90.72 | 99.50 | 94.55 | 97.56 | 99.68 | 77.63 | 2.5k |
| ViT-L/16, JFT | 87.76 | 90.54 | 99.42 | 93.90 | 97.32 | 99.74 | 76.28 | 0.68k |
| ViT-L/16, ImageNet-21k | 85.30 | 88.62 | 99.15 | 93.25 | 94.67 | 99.61 | 72.72 | 0.23k |
| BiT-L, ResNet152x4 | 87.54 | 90.54 | 99.37 | 93.51 | 96.62 | 99.63 | 76.29 | 9.9k |
| Noisy Student, EfficientNet-L2 | 88.4/88.5* | 90.55 | - | - | - | - | - | 12.3k |

- **データ規模の結果**: §4.2 と Appendix Table 5（`tbl:imagenet_imagenet21k_jft`）では、ImageNet pre-training だけだと ViT-L/16 の ImageNet fine-tuning accuracy は 76.53 で、ViT-B/16 の 77.91 より低い。ImageNet-21k では ViT-B/16 83.97、ViT-L/16 85.15、ViT-H/14 85.13 で差が縮まり、JFT-300M では ViT-B/16 84.15、ViT-L/16 87.12、ViT-H/14 88.04 と大きいモデルが伸びる。Table 5 の ImageNet 値は 384 resolution fine-tuning で、Table 2 の Polyak averaging と 512/518 resolution を使った値とは条件が異なる。
- **JFT subset の結果**: §4.2 では JFT の 9M、30M、90M、300M random subsets で linear few-shot を比較する。本文は、ViT-B/32 は 9M subset で ResNet50 より大きく悪いが 90M+ subsets では良くなり、ResNet152x2 と ViT-L/16 でも同じ傾向だと述べる。
- **scaling study**: §4.3 では、ViT は ResNet より performance/compute trade-off がよく、同じ performance に約 $2-4\times$ 少ない compute で到達すると著者は述べる。hybrid は小さい compute budget では pure ViT を少し上回るが、大きい model では差が消える。Appendix Table 6 では、ViT-L/16 14 epochs が ImageNet 87.12、exaFLOPs 1567、ViT-H/14 14 epochs が ImageNet 88.08、exaFLOPs 4262、ResNet200x3 14 epochs が ImageNet 87.22、exaFLOPs 3306 と報告される。
- **内部表現の観察**: §4.4 では、initial linear embedding の top principal components が patch 内 fine structure の basis functions のように見えること、position embeddings は近い patch ほど cosine similarity が高く row-column structure も現れること、attention distance は低層から global に見る head と local な head が共存し深層で広がることを報告する。hybrid では低層の localized attention が弱いとも述べる。
- **self-supervision**: §4.5 と Appendix B.1.2（`sec:self_supervision`）では masked patch prediction を予備実験として行う。patch embeddings の 50% を corrupt し、その内訳は `[mask]` embedding 80%、random other patch embedding 10%、そのまま 10%。目標は corrupted patch の 3-bit mean color、つまり 512 colors。ViT-B/16 を JFT で 1M steps、約 14 epochs、batch size 4096 で pre-train すると ImageNet 79.9% になり、from scratch より 2% 高いが supervised pre-training より 4% 低い。
- **ObjectNet**: Appendix D.9 では flagship ViT-H/14 を ObjectNet で評価し、top-5 accuracy 82.1%、top-1 accuracy 61.7% と報告する。
- **著者が主張する貢献**: 標準 Transformer を patch sequence に直接適用する単純な構成で、大規模事前学習と組み合わせれば CNN への依存は不要であり、JFT-300M pre-training では複数の分類ベンチマークで ResNet-based baselines を上回り、より少ない事前学習 compute で済む、という主張である。

## 妥当性と限界

- **この主張を支える根拠**: Table 2 は複数の downstream classification datasets に対する fine-tuning accuracy と TPUv3-core-days を並べる。ViT-L/16（JFT）は BiT-L に対して全 task で同等以上で、ViT-H/14（JFT）はさらに ImageNet、CIFAR-100、VTAB などで伸び、Noisy Student と比べても ImageNet と ImageNet-ReaL で同等以上である。§4.2 は ImageNet、ImageNet-21k、JFT-300M と JFT subsets でデータ規模の効果を分けて調べ、§4.3 は ResNet、ViT、hybrid を JFT-300M transfer で compute に対して比較する。Table 2 は 3 fine-tuning runs の平均と標準偏差を報告している。
- **著者が認めている limitations / future work**: Conclusion は、detection や segmentation など他の computer vision tasks への適用、self-supervised pre-training の探索、さらに大きい ViT への scaling を今後の課題として挙げる。§4.5 では self-supervised ViT-B/16 が supervised pre-training より 4% 低いことを明記し、contrastive pre-training は future work とする。§4.2 では ViT の few-shot properties のさらなる分析も future work としている。
- **読者として注意すべき点**: 最高性能の ViT-H/14 は in-house JFT-300M に依存する。public ImageNet-21k で pre-train した ViT-L/16 は Table 2 で ImageNet 85.30、VTAB 72.72 であり、JFT 版 ViT-H/14 の 88.55、77.63 とは差がある。ImageNet のみの pre-training では Large が Base を下回るので、「ViT は常に CNN より強い」とは読めない。
- **読者として注意すべき点**: compute 比較は重要だが、著者自身も pre-training efficiency は architecture choice だけでなく training schedule、optimizer、weight decay などにも影響されうると述べる。そのため §4.3 の controlled study と Table 6 を合わせて読む必要がある。
- **読者として注意すべき点**: VTAB の Specialized group では、Figure 2 の説明で top two models の performance は similar と書かれている。Natural と Structured ほど明確な優位として読まない方がよい。
- **追加で確認したい実験 / 疑問**: JFT-300M なしの public data だけで同じ compute 条件の Pareto curve はどこまで保たれるか。masked patch prediction 以外の self-supervised objective、特に本文が future work とする contrastive pre-training では 4% gap が縮むか。classification 以外の detection、segmentation に移したとき、patch 化と global self-attention の利点は同じように残るか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Vision Transformer（ViT）**: 画像を patch sequence として標準 Transformer encoder に入れるモデル。画像固有の inductive bias を少なくした設計が論文の焦点である。
- **patch**: 入力画像から切り出した $P\times P$ の領域を flatten したもの。ViT では NLP の token に対応する単位として扱う。
- **patch embedding**: flattened patch に線形射影 $\mathbf{E}$ をかけて $D$ 次元にした表現。
- **classification token**: BERT の `[class]` token と同様に sequence 先頭へ追加する学習可能 embedding。最終状態 $\mathbf{z}_L^0$ が画像表現になる。
- **position embedding**: patch の位置情報を保持するために patch embeddings へ加える学習可能 embedding。本文の標準設定は 1D position embeddings。Appendix D.4 の Table 8 では、No Pos. Emb. 0.61382、1-D 0.64206、2-D 0.64001、Rel. 0.64032 とされ、位置情報の有無は効くが種類の差は小さい。
- **inductive bias**: モデル構造にあらかじめ入っている仮定。CNN では locality、two-dimensional neighborhood structure、translation equivariance が各層に組み込まれる。ViT では patch extraction と fine-tuning 時の 2D interpolation 以外では 2D 構造を強く仮定しない。
- **MSA / SA**: self-attention は sequence 内の各要素が他の要素を重み付きで参照する操作。MSA は複数 head の SA を並列に行い、concat 後に射影する。
- **Hybrid architecture**: raw image patches の代わりに CNN feature map から sequence を作り、ViT に入れる構成。scaling study では小さい compute budget で pure ViT より少し良いが、大きいモデルでは差が消える。
- **BiT-L**: Big Transfer の large ResNet baseline。Table 2 では ResNet152x4 として ViT の主要比較対象になる。
- **Noisy Student**: EfficientNet-L2 を使う semi-supervised baseline。Table 2 では ImageNet と ImageNet-ReaL の SOTA 比較に出る。
- **VTAB**: 19 tasks の low-data transfer benchmark。各 task 1,000 training examples を使い、Natural、Specialized、Structured に分かれる。
- **TPUv3-core-days**: pre-training compute の単位。TPU v3 cores の数に training days を掛けたもの。Table 2 の compute 比較で使われる。
- **exaFLOPs**: Appendix Table 6 で使われる pre-training compute 指標。scaling study の詳細値を読むときに出る。
- **masked patch prediction**: BERT の masked language modeling をまねた予備的 self-supervised task。corrupted patch の 3-bit mean color を予測する。

## 読む順番の提案

- まず `main.tex` の abstract と `01_introduction.tex` を読み、"pure transformer applied directly to sequences of image patches" と "large scale training trumps inductive bias" という主張を押さえる。正規ノートでは `Summary（著者の主張）` の問題設定と結果の箇条書きにつながる。
- 次に `03_method.tex` の Figure 1 と Eq. (1) から Eq. (4) を読む。patch embedding、classification token、position embedding、Transformer encoder の流れが分かる。正規ノートでは「手法」の長い bullet と `Notes / Quotes` の Eq.1 に対応する。
- その後 `04_experiments.tex` の §4.1 と Table 1、Table 2 を読む。モデルサイズ、pre-training datasets、baseline、main results、TPUv3-core-days を確認する。正規ノートの `Summary（著者の主張）` の Table 2 数値と照合しやすい。
- 次に §4.2、Figure 3、Figure 4、Appendix Table 5 を読む。ImageNet だけでは ViT-Large が弱く、ImageNet-21k と JFT-300M で大きいモデルが効いてくる、という論文の条件付き主張を確認する。正規ノートでは `Takeaway` の「データ規模を見ずに一般化してはダメ」という点につながる。
- §4.3、Figure 5、Appendix Table 6 で compute と性能の trade-off を読む。正規ノートの「同精度を 2〜4× 少ない compute」に対応する。
- §4.4、Appendix D.4、Appendix D.7 で position embedding と attention distance の分析を読む。正規ノートの内部可視化、position embedding ablation の項目に対応する。
- 最後に §4.5、Appendix B.1.2、`05_conclusion.tex` を読む。self-supervision の 79.9%、+2%、-4% と、detection/segmentation、self-supervised pre-training、further scaling という future work を確認する。正規ノートでは `Critical Thoughts（評価・疑問）` と `Notes / Quotes` の self-supervision 項目に対応する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2010.11929v2/`
- 正規ノート: `notes/arXiv-2010.11929v2.md`
