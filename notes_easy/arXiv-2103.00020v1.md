# Learning Transferable Visual Models From Natural Language Supervision（自然言語監督による汎用視覚モデル CLIP の大規模事前学習）

- arXiv: https://arxiv.org/abs/2103.00020
- 一次ソース: ../papers/arXiv-2103.00020v1/
- 正規ノート: ../notes/arXiv-2103.00020v1.md

---

## 一言で言うと

固定されたラベル集合を予測する従来の画像分類を、Web 上の 400 million 個の `(image, text)` ペアから学ぶ `natural language supervision` に置き換え、自然言語でクラスを指定できる zero-shot 画像分類器 CLIP を作る論文。著者は、CLIP が ImageNet zero-shot で 76.2% を達成して元の ResNet-50 に並び、over 30 different existing computer vision datasets で転移・表現学習・頑健性を評価したと主張する（`clip_paper.tex` abstract, Table `zeroshot_table`）。

## 何を議論する論文か

- **問題設定**: State-of-the-art computer vision systems は、ImageNet の 1000 クラスや JFT-300M の 18291 クラスのような `fixed set of predetermined object categories` を予測するように訓練される。そのため、別の視覚概念を扱うには追加の labeled data が必要になる（`clip_paper.tex` abstract, Introduction）。
- **対象範囲 / 仮定**: 画像と自然言語テキストがペアになった大規模 Web データから、画像エンコーダとテキストエンコーダの共有埋め込み空間を学習する。論文での zero-shot は、通常の「未見カテゴリ」だけでなく、`generalization to unseen datasets` という広い意味で使われる（Experiments, §Zero-Shot Transfer）。
- **既存研究との差分**: Visual N-Grams は ImageNet zero-shot で 11.5% だった。VirTex, ICMLM, ConVIRT は画像と言語の表現学習を示したが、1-2 hundred thousand images 程度の規模で、著者は scale が決定的に違うと述べる。CLIP は 400 million `(image, text)` pairs で、caption 生成ではなく「どのテキストがどの画像に対応するか」を当てる contrastive objective を使う（Introduction, Approach）。
- **この論文で答えたい問い**: NLP の task-agnostic web-scale pre-training の成功を computer vision に移せるか。自然言語を使えば、タスクごとの分類ヘッドやデータセット固有の再学習なしに、既存の画像分類・OCR・動画行動認識・地理位置推定などへ転移できるか。さらに、その性能・スケーリング・頑健性・社会的影響はどう評価されるべきか。

## 背景と前提

- この論文を読むには、画像分類、事前学習、zero-shot transfer、対照学習、softmax / cross entropy、埋め込みベクトル、linear probe の基本を押さえておくとよい。
- `natural language supervision` は、画像に付随する自然言語を訓練信号として使うという広い概念である。論文は、unsupervised / self-supervised / weakly supervised / supervised という呼び名の違いよりも、自然言語が training signal になる点を重視する（Approach, §Natural Language Supervision）。
- 既存の crowd-labeled dataset は機械学習向けの `1-of-N majority vote "gold label"` を提供するが、自然言語はより広い視覚概念を表現できる。CLIP の狙いは、表現を学ぶだけでなく、その表現を言語に接続し、zero-shot transfer を可能にすることにある。
- 先行研究との関係では、CLIP の objective は `multi-class N-pair loss`、InfoNCE、ConVIRT の contrastive `(text, image)` representation learning と近い。著者は ConVIRT の単純化版を大規模に訓練したものとして CLIP を位置づける（Approach, §Selecting an Efficient Pre-Training Method）。
- 評価面では、zero-shot classifier だけでなく、同じ画像特徴に logistic regression を載せる linear probe、natural distribution shift、Oxford-IIIT Pets での人間比較、data overlap、FairFace / surveillance の broader impacts まで扱う。

## 提案手法

### コアアイデア

CLIP は、画像そのもののカテゴリラベルを固定的に学習するのではなく、`which caption goes with which image` を当てる事前学習を行う。具体的には、batch 内の N 個の正しい `(image, text)` ペアに対して、N x N 個の全組合せのうち実際に同時に現れた N ペアを高く、残りの N^2 - N ペアを低くスコアする。

データは `WIT` = WebImageText と呼ばれる 400 million `(image, text)` pairs で、variety of publicly available sources on the Internet から集められる。構築時には 500,000 queries を使い、各 query について up to 20,000 pairs を入れることで概略的に class balance する。query list は、英語版 Wikipedia で 100 回以上出る単語、high pointwise mutual information の bi-grams、一定以上の search volume を持つ Wikipedia articles 名、WordNet synsets などから作る（Approach, §Creating a Sufficiently Large Dataset）。

学習後、テキストエンコーダにクラス名や説明文を入力して、そのクラスの重みベクトルを生成する。画像エンコーダの出力と各テキスト埋め込みの cosine similarity を比較し、最も確率が高いクラスを予測する。著者はこの zero-shot prediction layer を、L2-normalized inputs, L2-normalized weights, no bias, temperature scaling を持つ multinomial logistic regression classifier と解釈し、text encoder を linear classifier の重みを生成する `hypernetwork` と見なす（Experiments, §Using CLIP for Zero-Shot Transfer）。

### 重要な定義・数式

TeX 本文には閉じた形の数式は多くない。中核は Figure `pseudocode` の Numpy-like pseudocode と、zero-shot 節の softmax 説明にある。

$$
I_e = \operatorname{l2\_normalize}(I_f W_i), \qquad
T_e = \operatorname{l2\_normalize}(T_f W_t)
$$

**式の意味**: 画像エンコーダとテキストエンコーダの出力を、それぞれ learned projection で同じ multimodal embedding space に写し、L2 正規化する。Figure `pseudocode` の `I_e = l2_normalize(np.dot(I_f, W_i), axis=1)` と `T_e = ...` に対応する。

**記号の定義**:
- $I_f$ ... `image_encoder(I)` が出す画像特徴。Figure `pseudocode` では形状 `[n, d_i]`。
- $T_f$ ... `text_encoder(T)` が出すテキスト特徴。形状 `[n, d_t]`。
- $W_i$ ... image feature を embedding dimension に写す learned projection。
- $W_t$ ... text feature を embedding dimension に写す learned projection。
- $I_e, T_e$ ... L2-normalized された画像・テキストの joint multimodal embedding。

**この論文での役割**: CLIP が画像とテキストを同じ空間で比較できるようにする部分である。dot product が cosine similarity として働くため、次の pairwise logits と対照学習の損失に直接つながる。

$$
\mathrm{logits} = I_e T_e^\top \exp(t)
$$

**式の意味**: batch 内のすべての画像埋め込みとテキスト埋め込みの組合せについて、scaled pairwise cosine similarities を作る。Figure `pseudocode` では `logits = np.dot(I_e, T_e.T) * np.exp(t)` と書かれる。

**記号の定義**:
- $I_e T_e^\top$ ... N 個の画像と N 個のテキストの全ペアの dot product。L2 正規化済みなので cosine similarity として読める。
- $t$ ... Figure `pseudocode` の learned temperature parameter。本文では softmax の logits の範囲を制御する温度 $\tau$ として説明され、log-parameterized multiplicative scalar として最適化される。
- $\exp(t)$ ... logits のスケール。Training 節では、logits を 100 より大きく scale しないよう clipped したとある。

**この論文での役割**: N x N の対応スコア行列を作る。対角成分が正しいペア、非対角成分が負例として扱われ、contrastive pre-training の中心になる。

$$
\mathrm{loss}_i = \mathrm{cross\_entropy\_loss}(\mathrm{logits}, \mathrm{labels}, \mathrm{axis}=0), \qquad
\mathrm{loss}_t = \mathrm{cross\_entropy\_loss}(\mathrm{logits}, \mathrm{labels}, \mathrm{axis}=1), \qquad
\mathrm{loss} = \frac{\mathrm{loss}_i + \mathrm{loss}_t}{2}
$$

**式の意味**: 画像から正しいテキストを選ぶ方向と、テキストから正しい画像を選ぶ方向の両方で cross entropy loss を計算し、平均する。本文の `symmetric cross entropy loss` と Figure `pseudocode` の `loss = (loss_i + loss_t)/2` に対応する。

**記号の定義**:
- $\mathrm{labels}$ ... Figure `pseudocode` の `np.arange(n)`。batch の i 番目の画像は i 番目のテキストと対応する。
- $\mathrm{loss}_i$ ... one direction の cross entropy。`axis=0` は pseudocode の表記。
- $\mathrm{loss}_t$ ... reverse direction の cross entropy。`axis=1` は pseudocode の表記。
- $n$ ... minibatch size。Training 節では 32,768。

**この論文での役割**: caption を逐語的に生成するのではなく、batch 内の正しい対応を当てる objective にすることで、Figure `compare_objective_fig` に示される効率改善の根拠になる。

$$
p(y=k \mid I) =
\frac{\exp\left(\exp(t)\, I_e^\top T_{e,k}\right)}
{\sum_{j=1}^{K} \exp\left(\exp(t)\, I_e^\top T_{e,j}\right)}
$$

**式の意味**: zero-shot 分類時に、画像 $I$ と各クラス名・説明文のテキスト埋め込みを比較し、softmax でクラス確率に正規化する。TeX 本文はこの閉じた式を直接は書かず、`cosine similarity ... scaled by a temperature parameter τ, and normalized into a probability distribution via a softmax` と説明しているため、ここでは Figure `pseudocode` のスケーリング表記と本文説明を合わせて式として整理している。

**記号の定義**:
- $I$ ... 分類したい入力画像。
- $T_{e,k}$ ... k 番目のクラス名または説明文を text encoder と projection に通した L2-normalized embedding。
- $K$ ... zero-shot classifier に含める候補クラス数。ImageNet では 1000 クラス。
- $p(y=k \mid I)$ ... 画像が k 番目のクラスである確率。

**この論文での役割**: 自然言語で指定したクラス集合から、追加訓練なしに分類器を作る部分である。prompt engineering と prompt ensembling は、この $T_{e,k}$ の作り方を改善する操作として読める。

### 実装 / アルゴリズム上の要点

- step1: 400 million `(image, text)` pairs の WIT を作る。500,000 queries、各 query up to 20,000 pairs、public Internet sources という構成が TeX に明記されている。
- step2: image encoder と text encoder を最初から同時に訓練する。ImageNet weights や pre-trained text weights で初期化しない。
- step3: image encoder は ResNet 系と Vision Transformer 系を試す。ResNet は ResNet-D improvements、antialiased rect-2 blur pooling、global average pooling の代わりの attention pooling を使う。ViT は patch/position embedding 後に追加 layer normalization を入れるなど小変更を行う。
- step4: text encoder は Transformer。base size は 63M-parameter、12-layer、512-wide、8 attention heads。テキストは lower-cased BPE 表現で、max sequence length は 76、`[EOS]` token の最上位層 activation を text feature とする。
- step5: 5 ResNets と 3 Vision Transformers を 32 epochs 訓練する。ResNet 系は RN50, RN101, RN50x4, RN50x16, RN50x64、ViT 系は ViT-B/32, ViT-B/16, ViT-L/14。さらに ViT-L/14 は 336 pixel resolution で 1 additional epoch 事前学習し、ViT-L/14@336px と呼ぶ。
- step6: optimization は Adam、decoupled weight decay、cosine schedule、batch size 32,768。temperature は 0.07 相当で初期化し、logits scaling が 100 を超えないよう clipped する。mixed precision、gradient checkpointing、half-precision Adam statistics、similarity 計算の sharding も使う。
- step7: 訓練コストは最大 ResNet の RN50x64 が 592 V100 GPUs で 18 days、最大 Vision Transformer が 256 V100 GPUs で 12 days。特に断りがない限り、論文中の `CLIP` 結果は最良の ViT-L/14@336px を指す。
- step8: zero-shot 評価では、クラス名だけでなく prompt を使う。`A photo of a {label}.` は ImageNet accuracy を 1.3% 改善し、ImageNet で 80 different context prompts を ensemble するとさらに 3.5% 改善する。合わせて almost 5% の改善とされる（Figure `prompt_engineering`）。

## 実験・結果

- **データセット / ベンチマーク**: zero-shot transfer は over 30 datasets を扱い、主要な 27 dataset suite には Food-101, CIFAR-10/100, Birdsnap, SUN397, Stanford Cars, FGVC Aircraft, Pascal VOC 2007, DTD, Oxford-IIIT Pets, Caltech-101, Oxford Flowers 102, MNIST, FER2013, STL-10, EuroSAT, RESISC45, GTSRB, KITTI, Country211, PatchCamelyon, UCF101, Kinetics700, CLEVR Counts, Hateful Memes, Rendered SST2, ImageNet が含まれる（Appendix Table `dataset_table`）。
- **比較対象 / baseline**: zero-shot では Visual N-Grams と比較する。ただし著者は、CLIP は 10x larger dataset、nearly 100x more compute per prediction、likely over 1000x training compute など多くの違いがあるため、これは direct methods comparison ではなく contextualizing performance だと明記する。linear probe では ResNet, EfficientNet / Noisy Student EfficientNet-L2, Instagram-pretrained ResNeXt, BiT, ViT, SimCLRv2, BYOL, MoCo, VirTex など 66 models を評価する。
- **指標**: 多くは accuracy だが、FGVC Aircraft / Oxford-IIIT Pets / Caltech-101 / Oxford Flowers 102 は mean per class、Pascal VOC 2007 は 11-point mAP、Hateful Memes は ROC AUC、Kinetics700 は mean(top1, top5) を使う（Table `dataset_table`）。
- **主な結果**: Table `zeroshot_table` では、ImageNet zero-shot が Visual N-Grams 11.5% から CLIP 76.2% に上がる。aYahoo は 72.4 から 98.4、SUN は 23.0 から 58.5。ImageNet top-5 は 95% で Inception-V4 に並ぶとされる。
- **Zero-shot vs supervised baseline**: Figure `zeroshot_vs_supervised` では、zero-shot CLIP が 27 datasets 中 16 datasets で ResNet-50 features 上の supervised logistic regression を上回る。Kinetics700 では ResNet-50 比 +14.5%、UCF101 では +7.7%。STL10 は 99.3% で、著者は new state of the art と述べる。一方、EuroSAT, RESISC45, PatchCamelyon, CLEVRCounts, GTSRB, KITTI Distance などの specialized / complex / abstract tasks では弱い。
- **Few-shot / data efficiency**: Figure `zeroshot_data_efficiency` では、zero-shot transfer の有効データ効率を、同じ CLIP feature space 上の logistic regression が何 examples per class 必要かで推定する。範囲は less than 1 から 184、中央値 5.4、平均 20.8。ImageNet では 16-shot linear classifier と同等とされる。
- **Zero-shot と linear probe の関係**: Figure `zeroshot_vs_linear_probe` では、zero-shot performance と fully supervised linear probe performance の相関は 0.82、p-value < 10^-6。ただし多くの dataset で zero-shot は fully supervised より 10% から 25% 低く、zero-shot が linear probe に近づくのは STL10, CIFAR10, Food101, OxfordPets, Caltech101 の 5 datasets だけとされる。
- **Scaling**: Figure `zeroshot_scaling` では、5 ResNet CLIP models、39 evaluations、36 datasets、44x range of compute にわたり、average zero-shot error が log-log linear trend でよくモデル化される。ただし個別タスクの性能は noisy で、単調かどうかは不確実と述べる。
- **Representation learning**: linear probe では、best CLIP model が Kornblith et al. の 12 dataset suite で best existing model を平均 2.6% 上回り、broader 27 dataset suite では previous systems より平均 5% 改善する。Figure `linear-probe-clip-vs-enet` は、CLIP が Noisy Student EfficientNet-L2 を 27 datasets 中 21 datasets で上回ると示す。Appendix Results でも、ViT-L/14 336px の CLIP が 99.5% Clopper-Pearson confidence interval 基準で 21 of 27 datasets の state of the art に入ったと書かれる。
- **Robustness**: natural distribution shift は ImageNetV2, ImageNet Sketch, Youtube-BB, ImageNet-Vid, ObjectNet, ImageNet Adversarial, ImageNet Rendition の 7 datasets。zero-shot CLIP は effective robustness を大きく改善し、ImageNet accuracy と distribution shift accuracy の gap を up to 75% 縮める（Figure `robust_main_fig`）。一方、CLIP features に ImageNet training set で L2 regularized logistic regression を合わせると ImageNet accuracy は +9.2% で 85.4% になるが、average accuracy under distribution shift は slightly decreases とされる（Figure `robustness_interventions`）。
- **人間比較**: Oxford-IIIT Pets の 37 breeds、test split 3669 images で 5 humans と比較する。Table `human-performance-on-pets` では、zero-shot human 53.7%、zero-shot CLIP 93.5%、one-shot human 75.7%、two-shot human 75.7%。著者は、人間は 0-shot から 1-shot で大きく改善するが、この論文の few-shot CLIP 評価はその prior knowledge の使い方を再現していないと述べる。
- **Data overlap**: overlap analysis では 35 datasets のうち 9 datasets に検出 overlap がない。median overlap 2.2%、average overlap 3.2%。overall accuracy gain は多くの場合 0.1% 以下で、最大 estimated increase は Birdsnap の 0.6%。Country211 は overlap 21.5% だが accuracy increase は 0.2% と報告される（Figure `overlap_fig`）。
- **Broader impacts**: FairFace では zero-shot / linear probe CLIP の race, gender, age 分類と denigration harm probes を調べる。10,000 FairFace images で non-human classes への誤分類は 4.9%、`Black` images は approximately 14% とされる。crime-related classes への誤分類は male images 16.5%、female images 9.8%。`child` category を追加すると 0-20 歳の crime-related / non-human category への誤分類が大きく減る（Tables `racial_bias_table`, `age_bias_table`）。
- **Surveillance**: CCTV では 515 surveillance images、12 video sequences、hand-captioned coarse classes を使い、初期 coarse classification は top-1 91.8%。`close` distractor を含める stress test では 51.1% に下がり、`close` answer を 40.7% 選ぶ。CelebA zero-shot identity recognition では CLIP L/14 が 100 classes で 59.2%、1k classes で 43.3%、2k classes で 42.2%（Table `celeba_table`）。
- **著者が主張する貢献**: 400 million `(image, text)` pairs から contrastive pre-training で画像表現を学ぶ CLIP を示し、自然言語 prompt による zero-shot transfer を over 30 datasets で評価したこと。さらに、linear probe、robustness、人間比較、data overlap、bias / surveillance まで含めて、task-agnostic web-scale pre-training を computer vision に移す可能性と限界を示したこと。

## 妥当性と限界

- **この主張を支える根拠**: 提案手法の妥当性は、(1) Figure `compare_objective_fig` の効率比較、(2) Table `zeroshot_table` の Visual N-Grams 比較、(3) Figure `zeroshot_vs_supervised` の 27 datasets zero-shot 評価、(4) Figure `fig:linear-probe-graph` と Table `tab:linear-probe-big-table` の linear probe、(5) Figure `robust_main_fig` / Table `table:robustness` の natural distribution shift によって支えられている。
- **著者が認めている limitations / future work**: zero-shot CLIP は平均的には ResNet-50 feature 上の supervised baseline と競合するが、全体の SOTA には届かない。著者は overall SOTA に到達するには around a 1000x increase in compute が必要で、current hardware では infeasible と推定する。fine-grained classification、counting、KITTI Distance のような novel tasks では弱く、MNIST zero-shot は 88% で raw pixels logistic regression に負ける。
- **著者が認めている limitations / future work**: CLIP は zero-shot classifier に含めた概念からしか選べず、image captioning のように novel outputs を生成できない。著者は contrastive objective と generative objective の joint training を `simple idea worth trying` と述べる。また、few-shot へ移ると性能が落ちる counter-intuitive behavior があり、strong zero-shot performance と efficient few-shot learning を組み合わせる方法が future work とされる。
- **著者が認めている limitations / future work**: 方法論上も、CLIP の開発では full validation sets を繰り返し問い合わせており、true zero-shot scenario とは違う。さらに main results の 27 dataset collection は `haphazardly assembled` で、CLIP の development and capabilities と co-adapted していると明記される。
- **読者として注意すべき点**: Visual N-Grams との比較は、データ量、計算量、モデル世代が揃っていない。著者自身が direct methods comparison ではないと明記するため、「CLIP の objective だけで 11.5% から 76.2% に上がった」と読むのは不適切である。
- **読者として注意すべき点**: robustness では、zero-shot CLIP が ImageNet-trained models より高い effective robustness を示すが、これが zero-shot だからなのか、大規模で多様な WIT や自然言語監督のためなのかは切り分け切れていない。著者は end-to-end finetuning でも同じ挙動が成り立つか confident answers はないと述べる。
- **読者として注意すべき点**: Broader Impacts の bias probes は initial efforts で limited in scope とされる。FairFace の race / gender categories 自体にも問題があると footnote で認めており、benchmark accuracy は fairness の十分条件ではないと明記する。
- **追加で確認したい実験 / 疑問**: CLIP の zero-shot robustness を保ったまま dataset-specific performance を上げる方法。validation set への co-adaptation が少ない新しい broad zero-shot benchmark で同じ傾向が再現するか。contrastive + generative objective が CLIP の効率と caption model の柔軟性を本当に両立するか。few-shot で zero-shot より悪化する原因が、linear probe の訓練法なのか、prompt-derived classifier との統合不足なのか。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **CLIP**: Contrastive Language-Image Pre-training。画像エンコーダとテキストエンコーダを contrastive objective で同時に訓練し、自然言語で zero-shot classifier を作る方法。
- **natural language supervision**: 画像に付随する自然言語テキストを training signal として使う考え方。論文は、従来の crowd-labeled `gold label` より広い視覚概念を表現できる点を強調する。
- **WIT / WebImageText**: CLIP のために構築した 400 million `(image, text)` pairs のデータセット。500,000 queries と up to 20,000 pairs per query により概略的に class balance する。
- **zero-shot transfer**: この論文では「未見カテゴリ」だけでなく、`generalization to unseen datasets` として広く使う。新しい dataset の training examples を使わず、クラス名や説明文を text encoder に渡して分類する。
- **multi-modal embedding space**: 画像とテキストを同じ次元のベクトル空間に写したもの。L2 正規化後の dot product が cosine similarity として使われる。
- **symmetric cross entropy loss**: 画像からテキストを当てる方向と、テキストから画像を当てる方向の両方で cross entropy を取り、平均する損失。Figure `pseudocode` の中心。
- **temperature / `t` / $\tau$**: softmax logits のスケールを制御するパラメータ。Figure `pseudocode` は `t` と `exp(t)` を使い、本文は temperature parameter $\tau$ と呼ぶ。
- **prompt engineering**: クラス名をそのまま入れるのではなく、`A photo of a {label}.` や `A photo of a {label}, a type of pet.` のように文脈を足す操作。polysemy と pre-training text distribution との gap を扱うために使う。
- **prompt ensembling**: 複数の prompt から作った zero-shot classifiers を ensemble する方法。論文では probability space ではなく embedding space で平均し、ImageNet では 80 prompts を使う。
- **linear probe**: 事前学習モデルの画像特徴を固定し、その上に logistic regression classifier を訓練する評価。CLIP の zero-shot classifier も linear classifier として読めるため、比較しやすい。
- **effective robustness / relative robustness**: Taori et al. の枠組み。effective robustness は ImageNet accuracy から予測される distribution shift accuracy をどれだけ上回るか、relative robustness は out-of-distribution accuracy 自体の改善を表す。
- **natural distribution shift**: ImageNetV2, ImageNet Sketch, Youtube-BB, ImageNet-Vid, ObjectNet, ImageNet-A, ImageNet-R のように、既存画像を人工的に壊した synthetic shift ではなく、新しく収集された自然画像分布の違い。
- **data overlap**: pre-training data と evaluation data の重複。論文は duplicate detector で Overlap / Clean subsets を作り、reported zero-shot accuracy がどれだけ inflated されたかを推定する。
- **class design**: Broader Impacts で使われる語。どのクラスを候補に含めるか、どんな言葉で表すかが model performance と unwanted biases / behaviour を変えるという論点。

## 読む順番の提案

- まず `clip_paper.tex` abstract と Figure `main_fig` を読む。ここで「固定分類器」から「text encoder が zero-shot linear classifier を合成する」構図を掴む。正規ノートの Summary の「問題」「手法」に対応する。
- 次に Introduction の Visual N-Grams, ImageNet / JFT-300M, VirTex / ICMLM / ConVIRT の段落を読む。CLIP が何を新規と主張しているか、特に scale と natural language supervision の位置づけが分かる。
- Method は §Creating a Sufficiently Large Dataset, §Selecting an Efficient Pre-Training Method, Figure `pseudocode`, §Training を順に読む。数値は 400 million pairs、500,000 queries、32 epochs、batch size 32,768、temperature clipping、RN50x64 / ViT training cost を確認する。
- Experiments は §Using CLIP for Zero-Shot Transfer, Table `zeroshot_table`, Figure `prompt_engineering`, Figure `zeroshot_vs_supervised`, Figure `zeroshot_data_efficiency`, Figure `zeroshot_scaling` を先に読む。正規ノートの Takeaway の zero-shot / prompt / scaling に対応する。
- Representation learning と robustness は Figure `fig:linear-probe-graph`, Figure `linear-probe-clip-vs-enet`, Figure `robust_main_fig`, Figure `robustness_interventions`, Table `table:robustness` を見る。正規ノートの Critical Thoughts の robustness 論点につながる。
- Limitations は必読。1000x compute、MNIST 88%、validation set への繰り返し問い合わせ、27 datasets の co-adaptation、few-shot での性能低下が、論文の主張をどう限定しているかを確認する。
- Broader Impacts は Bias と Surveillance を読む。FairFace の Tables `racial_bias_table`, `age_bias_table`、CelebA の Table `celeba_table` は、正規ノートの bias / surveillance メモと直接対応する。
- 最後に Appendix の Table `dataset_table`, Table `tab:zero-shot-big-table`, Table `tab:linear-probe-big-table`, Table `table:retrieval`, Table `table:ocr`, Table `table:action` を必要に応じて確認する。正規ノートの数値を検算する場所として使う。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2103.00020v1/`
- 正規ノート: `notes/arXiv-2103.00020v1.md`
