# Deep Residual Learning for Image Recognition（深い CNN を実際に最適化可能にする残差学習）

- arXiv: https://arxiv.org/abs/1512.03385
- 一次ソース: ../papers/arXiv-1512.03385v1/
- 正規ノート: ../notes/arXiv-1512.03385v1.md

---

## 一言で言うと

この論文は、層を深くすると training error まで悪化するという ``degradation'' 問題を、各ブロックに identity shortcut を入れて残差関数を学習させることで緩和する。ImageNet で最大 152 層、CIFAR-10 で 100 層・1000 層級の ResNet を学習し、深さを増やすと精度が改善すること、さらにその表現が ILSVRC & COCO 2015 の detection / localization / segmentation submissions の基盤になることを示す。

## 何を議論する論文か

- **問題設定**: 深い convolutional neural networks は表現の階層を増やせるが、単に層を積むだけでは、ある深さ以降で accuracy が飽和し、さらに悪化する。著者はこれを ``degradation'' と呼び、Fig.~\ref{fig:teaser} と Fig.~\ref{fig:imagenet} で、深い plain network の方が validation error だけでなく training error も高いことを示す。
- **対象範囲 / 仮定**: 主対象は画像認識用 CNN で、ImageNet 2012 classification、CIFAR-10、PASCAL VOC / MS COCO / ImageNet の detection・localization で評価する。学習は SGD + backpropagation、BN、He initialization を使い、ImageNet / CIFAR-10 の分類実験では dropout を使わない。
- **既存研究との差分**: VGG nets は 3x3 convolution を深く積む方針、GoogLeNet / Inception は深い構造と補助分類器を使う。Highway Networks は gated shortcut を使うが、ResNet の shortcut は parameter-free identity が中心で、gate が閉じて non-residual function になることがない、と著者は対比する。
- **この論文で答えたい問い**: Introduction の問いは ``Is learning better networks as easy as stacking more layers?'' である。著者の答えは、plain に積むだけでは難しいが、残差形式に再定式化すれば 100 層超のネットワークでも最適化しやすく、深さから精度向上を得られる、というもの。

## 背景と前提

- 深い CNN では、低レベル・中レベル・高レベルの特徴と classifier が end-to-end に統合される。論文は、特徴の ``levels'' は stacked layers の数、つまり depth によって豊かになる、という前提から出発する。
- 以前から vanishing / exploding gradients は深層ネット学習の障害だったが、normalized initialization と intermediate normalization layers、特に BN により、数十層のネットワークは SGD で収束を開始できるようになっていた。したがってこの論文の主張は「勾配消失が残っている」ではなく、BN でも plain network に残る optimization difficulty を扱う。
- degradation は overfitting ではない。TeX では、深くしたモデルが higher training error を持つと明記される。もし追加層が identity mapping を実現できるなら、深いモデルは浅いモデルの解を含むはずなので、training error は悪くならないはずである。この「構成上は存在する解を solver が見つけられない」ことが論文の問題設定である。
- shortcut connection 自体は古くから研究されているが、本論文では「数層を飛ばす identity shortcut の出力を、stacked layers の出力に element-wise addition する」ことを residual learning の実装として使う。重要なのは、identity shortcut は追加パラメータと計算量を増やさないため、plain network と residual network をほぼ同じ depth / width / parameter / computational cost で比較できる点である。
- 関連する baseline として、分類では VGG-16, GoogLeNet, PReLU-net, BN-inception、CIFAR-10 では Maxout, NIN, DSN, FitNet, Highway Networks、検出では Faster R-CNN with VGG-16 が使われる。

## 提案手法

### コアアイデア

数層の stacked nonlinear layers に、望ましい写像 $\mathcal{H}(\mathbf{x})$ を直接学習させるのではなく、入力 $\mathbf{x}$ との差である residual function $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$ を学習させる。ブロックの出力は $\mathcal{F}(\mathbf{x})+\mathbf{x}$ になる。

この設計の狙いは、もし identity mapping が最適に近いなら、非線形層の重みを調整して identity を再現するより、残差を 0 に近づける方が solver にとって容易かもしれない、という仮説である。論文はこの仮説を理論的に証明するのではなく、plain network との対照実験、shortcut の ablation、層応答の標準偏差、分類・検出への転用で支持する。

ImageNet 用には、VGG 風の plain network に shortcut を挿入して 18 / 34 層の ResNet を作る。さらに 50 / 101 / 152 層では、各 residual function を $1\times1$, $3\times3$, $1\times1$ convolution からなる bottleneck design に変更する。CIFAR-10 では、より単純な $6n+2$ 層のネットワークを使い、20 / 32 / 44 / 56 / 110 / 1202 層を調べる。

### 重要な定義・数式

$$
\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}, \qquad
\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}
$$

**式の意味**: 望ましい underlying mapping $\mathcal{H}(\mathbf{x})$ を直接学習する代わりに、入力からの差分である residual mapping $\mathcal{F}(\mathbf{x})$ を学習するという再定式化である。Sec.~3.1 と Introduction でこの形が提案手法の動機として導入される。

**記号の定義**:
- $\mathbf{x}$ ... 数層の stacked layers に入る入力ベクトル、または feature map
- $\mathcal{H}(\mathbf{x})$ ... その数層で本来フィットしたい underlying mapping
- $\mathcal{F}(\mathbf{x})$ ... $\mathcal{H}(\mathbf{x})-\mathbf{x}$ として定義される residual function

**この論文での役割**: 論文全体の中心定義である。degradation 問題を「深い層が identity mapping を学習できない」という最適化の困難として捉え、identity を shortcut で直接渡し、残差だけを学習対象にする理由を与える。

$$
\mathbf{y}= \mathcal{F}(\mathbf{x}, \{W_{i}\}) + \mathbf{x}
$$

**式の意味**: Eqn.(1) の residual building block である。stacked layers が出す residual mapping と、shortcut で渡した入力 $\mathbf{x}$ を element-wise addition してブロック出力 $\mathbf{y}$ を作る。

**記号の定義**:
- $\mathbf{x}$ ... building block の入力
- $\mathbf{y}$ ... building block の出力
- $\mathcal{F}(\mathbf{x}, \{W_i\})$ ... 学習される residual mapping
- $\{W_i\}$ ... residual branch 内の layer weights

**この論文での役割**: Fig.~\ref{fig:block} の基本ブロックを表す。identity shortcut は追加パラメータも計算量も増やさないため、plain / residual networks の公平な比較を可能にする。

$$
\mathcal{F}=W_{2}\sigma(W_{1}\mathbf{x})
$$

**式の意味**: Fig.~\ref{fig:block} の 2-layer residual branch の例である。入力に $W_1$ を適用し、ReLU $\sigma$ を通し、さらに $W_2$ を適用する。TeX では notation を簡単にするため biases は省略される。

**記号の定義**:
- $W_1, W_2$ ... residual branch の 2 つの重み
- $\sigma$ ... ReLU activation
- $\mathbf{x}$ ... branch への入力
- $\mathcal{F}$ ... shortcut に足される residual branch の出力

**この論文での役割**: residual branch は特別な solver や目的関数ではなく、通常の convolution / activation の積み重ねで実装できることを示す。実際の ImageNet では 2-layer block と 3-layer bottleneck block の両方が使われる。

$$
\mathbf{y}= \mathcal{F}(\mathbf{x}, \{W_{i}\}) + W_{s}\mathbf{x}
$$

**式の意味**: Eqn.(2) の projection shortcut である。入力 $\mathbf{x}$ と residual branch の出力の次元が一致しない場合に、shortcut 側へ線形射影 $W_s$ を入れて次元を合わせる。

**記号の定義**:
- $W_s$ ... shortcut connection による linear projection。実験では次元増加時に $1\times1$ convolution として使われる
- $\mathbf{x}, \mathbf{y}, \mathcal{F}, \{W_i\}$ ... Eqn.(1) と同じ

**この論文での役割**: shortcut option A/B/C の比較に対応する。A は zero-padding で全 shortcut を parameter-free、B は次元増加時のみ projection、C は全 shortcut を projection とする。Table~\ref{tab:10crop} では error は A > B > C（C が最良）だが差は小さく、著者は projection shortcut が degradation の解決に本質的ではないと解釈する。

$$
\text{depth} = 6n + 2
$$

**式の意味**: CIFAR-10 実験で使う plain / residual network の weighted layers 数である。3 種類の feature map size $\{32,16,8\}$ にそれぞれ $2n$ layers を置き、最初の convolution と最後の fully-connected layer を含めて $6n+2$ になる。

**記号の定義**:
- $n$ ... 各 feature map size ごとの residual block 数に対応する整数
- $6n$ ... 3 つの stage にある $3\times3$ convolution layers の総数
- $+2$ ... 最初の convolution と最後の fully-connected layer

**この論文での役割**: CIFAR-10 で深さだけを体系的に変える実験設計を表す。$n=\{3,5,7,9\}$ が 20 / 32 / 44 / 56 層、$n=18$ が 110 層、$n=200$ が 1202 層に対応する。

### 実装 / アルゴリズム上の要点

- ImageNet の plain baseline は VGG nets の方針に基づき、主に $3\times3$ filters を使う。同じ output feature map size では同じ filter 数、feature map size が半分になると filter 数を 2 倍にし、downsampling は stride 2 の convolution で行う。最後は global average pooling と 1000-way fully-connected layer + softmax である。
- ResNet-18 / 34 は、この plain network に shortcut connections を挿入したもの。input / output dimensions が同じなら identity shortcut、次元が増える場合は option A の zero-padding または option B の $1\times1$ projection を使う。shortcut が feature map size をまたぐ場合は stride 2。
- ResNet-50 / 101 / 152 は bottleneck block を使う。各 residual function は $1\times1$ で次元を減らし、$3\times3$ で処理し、$1\times1$ で次元を戻す。Table~\ref{tab:arch} では ResNet-152 が 11.3 billion FLOPs で、VGG-16 / VGG-19 の 15.3 / 19.6 billion FLOPs より低い複雑度だと述べられる。
- ImageNet training では、shorter side を $[256,480]$ から random sample して resize、224x224 crop と horizontal flip、per-pixel mean subtraction、standard color augmentation を使う。各 convolution の直後・activation の前に BN を置く。mini-batch size は 256、learning rate は 0.1 から開始し plateau で 10 分の 1、最大 $60\times10^4$ iterations、weight decay 0.0001、momentum 0.9、dropout なし。
- ImageNet testing では、比較実験に standard 10-crop testing を使う。best results では fully-convolutional form と multi-scale scoring を使い、shorter side を $\{224,256,384,480,640\}$ に resize して平均する。
- CIFAR-10 では 32x32 images、per-pixel mean subtraction、3x3 convolution、feature map sizes $\{32,16,8\}$、filters $\{16,32,64\}$、global average pooling、10-way fully-connected layer、softmax を使う。weight decay 0.0001、momentum 0.9、He initialization、BN、dropout なし。mini-batch size は 128、標準の learning rate は 0.1 から始め、32k / 48k iterations で 10 分の 1、64k iterations で終了する。110-layer ResNet では 0.1 が開始時にやや大きいため、training error が 80% 未満になるまで 0.01 で warm up してから 0.1 に戻す。

## 実験・結果

- **データセット / ベンチマーク**: ImageNet 2012 classification は 1000 classes、1.28 million training images、50k validation images、100k test images。CIFAR-10 は 50k training images、10k testing images、10 classes。検出では PASCAL VOC 2007 / 2012 と MS COCO、付録では ImageNet Detection と ImageNet Localization も扱う。
- **比較対象 / baseline**: ImageNet classification では plain-18 / plain-34、VGG-16、GoogLeNet、PReLU-net、BN-inception と比較する。CIFAR-10 では Maxout、NIN、DSN、FitNet、Highway Networks と比較する。検出では Faster R-CNN の backbone を VGG-16 から ResNet-101 に置き換える比較を行う。
- **指標**: 分類は top-1 error と top-5 error。CIFAR-10 は classification error。検出は mAP@.5 と COCO standard metric の mAP@[.5,.95]。localization は top-5 localization error を使う。
- **主な結果**: ImageNet validation の 10-crop testing では、plain-34 が top-1 28.54 / top-5 10.02、ResNet-34 A が 25.03 / 7.76、ResNet-50 が 22.85 / 6.71、ResNet-101 が 21.75 / 6.05、ResNet-152 が 21.43 / 5.71（Table~\ref{tab:10crop}）。single-model validation では ResNet-152 が top-1 19.38 / top-5 4.49（Table~\ref{tab:single}）。ensemble は ImageNet test top-5 error 3.57% で、ILSVRC 2015 classification task の 1st place とされる（Table~\ref{tab:ensemble}）。
- **主な結果**: plain / ResNet の直接比較では、ImageNet 10-crop top-1 error が 18 layers で plain 27.94、ResNet 27.88、34 layers で plain 28.54、ResNet 25.03（Table~\ref{tab:plain_vs_shortcut}）。Fig.~\ref{fig:imagenet} では 34-layer plain net の training error が 18-layer plain net より高く、ResNet では 34-layer が 18-layer より良い。
- **主な結果**: shortcut option は ResNet-34 で A 25.03 / 7.76、B 24.52 / 7.46、C 24.19 / 7.40（top-1 / top-5, Table~\ref{tab:10crop}）。著者は、A/B/C の差が小さいことから projection shortcuts は degradation 問題の解決に essential ではないと解釈し、以後は計算量・モデルサイズを抑えるため主に option B を使う。
- **主な結果**: CIFAR-10 では ResNet-20 8.75%、32 7.51%、44 7.17%、56 6.97%、110 6.43%（5 runs の mean±std は 6.61±0.16）、1202 7.93%（Table~\ref{tab:cifar}）。1202-layer network は training error <0.1% に達するが、test error は 110-layer より悪く、著者は overfitting と解釈する。
- **主な結果**: Layer response analysis では、Fig.~\ref{fig:std} で ResNets の layer responses の standard deviations が plain counterparts より一般に小さく、ResNet-20 / 56 / 110 の比較では深いほど個々の層が signal をより小さく変更する傾向が示される。
- **主な結果**: baseline Faster R-CNN で backbone を VGG-16 から ResNet-101 に替えると、PASCAL VOC 2007 test mAP は 73.2 から 76.4、VOC 2012 test mAP は 70.4 から 73.8（Table~\ref{tab:detection_voc}）。COCO validation では mAP@.5 が 41.5 から 48.4、mAP@[.5,.95] が 21.2 から 27.2（Table~\ref{tab:detection_coco}）。著者は同一 detection implementation なので gain は better networks に帰属できると述べる。
- **著者が主張する貢献**: residual learning framework により、以前より substantially deeper なネットワークを学習しやすくしたこと、ImageNet で 152-layer ResNet を評価し VGG nets より 8x deeper かつ lower complexity としたこと、ImageNet test top-5 error 3.57% を達成したこと、COCO object detection で 28% relative improvement を得たこと、ILSVRC & COCO 2015 の複数タスクで 1st place を得たこと。

## 妥当性と限界

- **この主張を支える根拠**: degradation 問題は、ImageNet の 18 / 34-layer plain nets と CIFAR-10 の plain nets で、深い方の training error が高いことで示される。これは validation error だけでなく training error の比較なので、単純な overfitting ではないという主張を支える。
- **この主張を支える根拠**: residual learning の効果は、same depth / width / parameter / computational cost に近い plain vs ResNet 比較で示される。特に option A は追加パラメータなしなので、ResNet-34 A が plain-34 より大きく改善する結果は、projection やパラメータ増ではなく residual formulation の効果を示す対照実験になっている。
- **この主張を支える根拠**: shortcut option A/B/C の比較では C が最良だが差は小さい。著者は C の改善を extra parameters によるものとみなし、projection shortcut は essential ではないと述べる。この解釈は Table~\ref{tab:10crop} の ablation に基づく。
- **この主張を支える根拠**: CIFAR-10 の 1202-layer network が training error <0.1% まで下がることは、少なくともこの設定では 1000 層級 ResNet の optimization difficulty が顕在化していないという根拠になる。ただし test error は 110-layer より悪い。
- **著者が認めている limitations / future work**: 34-layer plain net の optimization difficulty について、BN により forward signals は non-zero variances を持ち、backward gradients も healthy norms で、vanishing gradients が原因とは考えにくいと述べる一方で、原因は future work とされる。著者は exponentially low convergence rates の可能性を conjecture として述べるにとどめる。
- **著者が認めている limitations / future work**: 1202-layer ResNet は training error が低いのに test error が悪く、著者は 19.4M parameters が CIFAR-10 には大きすぎる overfitting と解釈する。maxout や dropout のような strong regularization を組み合わせる可能性は future work とされる。
- **読者として注意すべき点**: この論文は residual learning がなぜ最適化を容易にするかを数学的に証明する論文ではない。中心は、plain / residual の対照、深さスイープ、shortcut ablation、複数タスクでの実験証拠である。
- **読者として注意すべき点**: 検出の後半の改善、たとえば box refinement、context、multi-scale testing、ensemble は ResNet 以外の工夫も含む。backbone 置換だけの効果として読むべき主要比較は、baseline Faster R-CNN の VGG-16 vs ResNet-101 である。
- **追加で確認したい実験 / 疑問**: plain network の degradation が、loss landscape、Jacobian / Hessian の条件、gradient の分散などのどの要因と関係するかは、この TeX では明示的に切り分けられていない（これらの確認観点自体は TeX 中には明示されていない）。
- **追加で確認したい実験 / 疑問**: shortcut option の詳細比較は主に ResNet-34 で示される。bottleneck architecture の深いモデルで option A/B/C を同じ粒度で比較した結果は、この TeX には示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **degradation problem**: 深くすると accuracy が飽和後に悪化し、training error も高くなる現象。著者は overfitting ではなく optimization difficulty として扱う。
- **plain network**: shortcut を持たず、convolutional layers を単純に積んだ baseline。ImageNet では VGG nets の設計思想に基づく 18 / 34-layer network、CIFAR-10 では $6n+2$ 層の network。
- **residual learning**: $\mathcal{H}(\mathbf{x})$ を直接学習する代わりに $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x})-\mathbf{x}$ を学習し、出力を $\mathcal{F}(\mathbf{x})+\mathbf{x}$ にする再定式化。
- **identity shortcut connection**: 入力 $\mathbf{x}$ を residual branch の出力にそのまま足す接続。Eqn.(1) の $+\mathbf{x}$ に対応し、追加パラメータや計算量を増やさない。
- **projection shortcut**: 次元が合わないときに $W_s\mathbf{x}$ を足す shortcut。ImageNet では $1\times1$ convolution で実装され、option B では次元増加時のみ使われる。
- **option A / B / C**: shortcut の次元合わせの ablation。A は zero-padding で全 shortcut が parameter-free、B は次元増加時のみ projection、C は全 shortcut が projection。
- **bottleneck design**: ResNet-50 / 101 / 152 で使う $1\times1$, $3\times3$, $1\times1$ の 3-layer residual branch。1x1 layers が dimensions を reduce / restore し、3x3 layer の計算量を抑える。
- **top-1 / top-5 error**: ImageNet classification の評価指標。top-1 は最上位予測が外れた割合、top-5 は上位 5 予測に正解が含まれない割合。
- **mAP@.5 / mAP@[.5,.95]**: 物体検出の評価指標。mAP@.5 は IoU=0.5、mAP@[.5,.95] は COCO standard metric として複数 IoU thresholds を使う。
- **layer responses の standard deviations**: Fig.~\ref{fig:std} の分析対象。各 $3\times3$ layer の出力を BN 後・nonlinearity 前で見た標準偏差で、ResNet では residual functions の response strength を見るために使われる。

## 読む順番の提案

- まず Abstract と Introduction の ``degradation'' の説明を読む。正規ノートの Summary 冒頭にある「training error が深くしたほうが高くなる」という問題設定につながる。
- 次に Sec.~3.1 と Sec.~3.2、Eqn.(1) / Eqn.(2)、Fig.~\ref{fig:block} を読む。ここが正規ノートの「$\mathcal{H}(x)$ ではなく $\mathcal{F}(x):=\mathcal{H}(x)-x$ を学習する」という手法説明に対応する。
- ImageNet の Table~\ref{tab:plain_vs_shortcut} と Fig.~\ref{fig:imagenet} を先に見る。plain-34 の degradation と ResNet-34 の改善が、この論文で一番重要な対照実験である。
- その後 Table~\ref{tab:10crop} と Table~\ref{tab:single} / Table~\ref{tab:ensemble} を読む。shortcut option、50 / 101 / 152 層への拡張、single-model と ensemble の結果がまとまっている。
- CIFAR-10 では Sec.~4.2、Fig.~\ref{fig:cifar}、Table~\ref{tab:cifar}、Fig.~\ref{fig:std} を読む。正規ノートの「1202 層は training error <0.1% だが test は 110 層に劣る」という評価・限界につながる。
- 検出や位置推定の主張を確認したい場合は Sec.~4.3 と Appendix の detection / localization tables を読む。正規ノートの Critical Thoughts にある「純粋な backbone 置換ゲインと追加技のゲインを分けて読む」という注意点に対応する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-1512.03385v1/`
- 正規ノート: `notes/arXiv-1512.03385v1.md`
