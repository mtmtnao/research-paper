# DINOv2: Learning Robust Visual Features without Supervision（大規模 curated data 上の自己教師あり視覚基盤モデル）

- arXiv: https://arxiv.org/abs/2304.07193
- 一次ソース: ../papers/arXiv-2304.07193v2/
- 正規ノート: ../notes/arXiv-2304.07193v2.md

---

## 一言で言うと

DINOv2 は、画像とテキストのペアではなく画像のみを用いた self-supervised learning で、fine-tuning なしに image-level と pixel-level の広い下流タスクへ転移する汎用視覚特徴を作れるかを問う論文である。著者は、大規模 curated dataset `LVD-142M`、DINO+iBOT 系の学習 recipe、1.1B parameter の `ViT-g/14` と distillation を組み合わせ、OpenCLIP などの weakly-supervised feature と多くのベンチマークで同等または上回ると主張する（`abstract.tex`, `intro.tex`, `experiments.tex`）。

## 何を議論する論文か

- **問題設定**: NLP では raw text から学習した foundation model の feature を「そのまま」使える一方、computer vision では caption や image-text aligned corpus に依存する text-guided pretraining が中心である。本論文は、画像だけから学ぶ self-supervised pretraining で、classification, segmentation, depth, retrieval などにそのまま使える `general-purpose visual features` を得られるかを検証する。
- **対象範囲 / 仮定**: 学習対象は ViT image encoder であり、中心的な評価は frozen backbone の上に linear probe, kNN, simple decoder, DPT decoder などを載せる設定である。pretraining loss はラベルやテキストを使わないが、データ構築では ImageNet-22k, ImageNet-1k train, Google Landmarks, fine-grained / segmentation / depth / retrieval datasets などの curated image datasets を retrieval query として使う（`data.tex`, `supp-data.tex`）。
- **既存研究との差分**: CLIP/OpenCLIP/SWAG/EVA-CLIP は text supervision または weak supervision を使う。MAE は fine-tuning 後に強いが frozen features は本論文の比較では弱い。DINO/iBOT などの discriminative SSL は ImageNet-1k や ImageNet-22k で発展してきたが、大規模化では uncurated data の品質低下が問題だったと著者は述べる（`intro.tex`, `related.tex`）。
- **この論文で答えたい問い**: 「十分に大きく多様で curated な画像集合を使えば、self-supervised pretraining alone で weakly-supervised model と競合する frozen visual features を作れるか」である。論文末では、性能要因を training recipe, model scale, dataset scale, distillation の複合として整理している（`tmlr.tex` Future work and Discussion）。

## 背景と前提

- **self-supervised learning (SSL)** は、人手ラベルや caption を目的変数にせず、入力画像から作った pretext objective によって特徴を学ぶ。DINOv2 では DINO の image-level objective と iBOT の patch-level masked prediction を組み合わせる（`approach.tex`）。
- **Vision Transformer (ViT)** は画像を patch token 列として扱う。本論文では image-level task では class token、dense task では patch token の質が重要になる。
- **teacher-student / EMA** は DINO/iBOT 系の基本構成である。student を最適化し、teacher は student の過去パラメータの exponential moving average として更新する（`approach.tex`, `supp-implem.tex`）。
- **frozen feature 評価**では backbone を固定し、linear classifier, kNN, retrieval, simple segmentation head などで特徴の可読性を測る。これは、著者の `finetuning is optional` という主張を支える評価設計である（`experiments.tex` Table `tab:lin-inet1k`, `tab:ft-inet1k-alone`）。
- **text-guided pretraining との関係**: caption は画像の情報を近似するだけなので、pixel-level information が表面化しにくい可能性がある、というのが著者の問題意識である（`intro.tex`）。ただし、本論文は text-guided model が不要だと一般に断じるのではなく、carbon section では text encoder を再利用する場合は text-guided training に意味があるとも述べる（`carbon.tex`）。

## 提案手法

### コアアイデア

DINOv2 は、単一の新しい目的関数を提案するというより、既存の discriminative SSL を大規模データ・大規模 ViT で安定に動かす recipe を作る論文である。pretraining objective は DINO の class-token 間 cross-entropy、iBOT の masked patch prediction、SwAV 由来の Sinkhorn-Knopp centering、KoLeo regularizer、終盤の high-resolution adaptation から成る（`approach.tex`）。

データ面では、uncurated web images から、curated image datasets に近い画像を retrieval して `LVD-142M` を作る。本文では post-processing 後に `1.2B unique images` と書かれ、補足では uncurated source `1.3B` から self-deduplication で `1.1B`、relative deduplication で `744M` へ減らす手順が説明されている。最終的な `LVD-142M` は `142,109,386` images である（`data.tex`, `supp-data.tex` Table `tab:lavida-details`）。

モデル面では、`ViT-g/14` を scratch から学習し、その後 `ViT-S/14`, `ViT-B/14`, `ViT-L/14` を `ViT-g/14` から distill する。`ViT-g/14` は embedding dimension `1536`, heads `24`, blocks `40`, `1.1B` parameters であり、head dimension を 64 にするため Zhai et al. の構成から変更している（`approach.tex`, `supp-implem.tex` Table `tab:vit-hparams`）。

### 重要な定義・数式

$$
m(s, r) = \text{cosine-similarity}\left(f\left(s\right),f\left(r\right)\right) = \frac{f(s)\cdot{}f(r)}{\lVert f(s)\rVert_2\lVert f(r)\rVert_2}
$$

**式の意味**: 2 枚の画像 `s`, `r` の feature embedding がどれだけ近いかを cosine similarity で測る式である。`supp-data.tex` の `Image similarity` で、deduplication と retrieval の基礎として定義されている。

**記号の定義**:
- $s, r$ ... 比較する画像のペア
- $f$ ... 画像から feature を生成するモデル
- $f(s), f(r)$ ... それぞれの画像 embedding
- $\lVert\cdot\rVert_2$ ... L2 norm

**この論文での役割**: `LVD-142M` を作るとき、uncurated pool から curated datasets に近い画像を探すための距離尺度である。データ curation が本論文の性能主張の中心なので、この式は pretraining data の構成根拠に直接関わる。

$$
{\mathcal L}_{DINO} = - \sum p_t \log p_s
$$

**式の意味**: 同じ画像の異なる crop から得た teacher と student の class-token 出力を、prototype score の softmax 後の分布として比較する cross-entropy loss である（`approach.tex` の `Image-level objective`）。

**記号の定義**:
- ${\mathcal L}_{DINO}$ ... DINO image-level objective
- $p_t$ ... teacher DINO head から得た prototype score を softmax と centering 後にした分布
- $p_s$ ... student DINO head から得た prototype score を softmax した分布
- $\sum$ ... prototype 次元にわたる和

**この論文での役割**: class token が画像全体の情報を保持するようにする主要 loss である。ImageNet classification や kNN/linear probing のような image-level 評価を支える学習信号として使われる。

$$
{\mathcal L}_{iBOT} = - \sum_i p_{ti} \log p_{si}
$$

**式の意味**: student では一部 patch を mask し、teacher では対応する visible patch token を使って、masked patch index ごとに teacher 分布と student 分布を合わせる loss である（`approach.tex` の `Patch-level objective`）。

**記号の定義**:
- ${\mathcal L}_{iBOT}$ ... iBOT patch-level masked prediction objective
- $i$ ... student で mask された patch token の index
- $p_{ti}$ ... teacher iBOT head から得た、patch $i$ に対応する分布
- $p_{si}$ ... student iBOT head から得た、masked patch $i$ の分布

**この論文での役割**: patch token に局所情報を持たせるための loss である。ablation では MIM objective を外すと ADE-20k linear segmentation が `44.2` から `47.1` へ改善する差分を失うため、dense prediction で重要だと著者は解釈している（`ablation.tex` Table `tab:ibot`）。

$$
{\mathcal L}_{\mathrm{koleo}} = - \frac{1}{n} \sum_{i=1}^n \log( d_{n, i}), \quad d_{n, i} = \min_{j \neq i} \| x_i - x_j \|
$$

**式の意味**: batch 内の feature が互いに近づきすぎないように、各 feature から最も近い別 feature までの距離を大きくする正則化である。論文では Kozachenko-Leonenko differential entropy estimator に由来すると説明される（`approach.tex`）。

**記号の定義**:
- ${\mathcal L}_{\mathrm{koleo}}$ ... KoLeo regularizer
- $n$ ... batch 内の vector 数
- $x_i$ ... $i$ 番目の feature vector
- $d_{n,i}$ ... $x_i$ から batch 内の最も近い別 vector までの距離

**この論文での役割**: feature 空間を広く使わせる正則化である。ablation では KoLeo ありで Oxford-M retrieval が `55.6` から `63.9` に上がり、著者は nearest-neighbor search task に効くと述べる（`ablation.tex` Table `tab:koleo`）。

### 実装 / アルゴリズム上の要点

- **データ構築**: curated datasets と uncurated data source を embedding 化し、uncurated data を deduplicate してから curated images に match する（`data.tex` Fig. `fig:retrieval-system`）。retrieval では ImageNet-22k で self-supervised pretraining した `ViT-H/16` の embedding と cosine similarity を使う。大規模 dataset には sample-based retrieval を使い、Google Landmarks v2 と ImageNet-22k は `k=4`、ImageNet-1k train は `k=32` とする。小規模 dataset には `100,000` clusters の cluster-based retrieval を使い、各 dataset から最大 `1M` images を入れる（`supp-data.tex`）。
- **学習 recipe**: DINO/iBOT heads は iBOT 原論文と異なり separate heads にする。teacher 側の centering は Sinkhorn-Knopp を `3` iterations 実行し、student は softmax normalization を使う。KoLeo は first global crop の class tokens に重み `0.1` で適用する（`approach.tex`, `supp-implem.tex`）。
- **高解像 adaptation**: pixel-level task で小物体が低解像度で消える問題に対し、pretraining 終盤の短期間だけ `518 x 518` に上げる（`approach.tex`）。resolution ablation では `224 -> 416` を `10k` iterations だけ行う設定が、全期間 `416` に近い結果をより低コストで出すと説明される（`ablation.tex` Fig. `fig:res`）。
- **効率化**: custom FlashAttention、sequence packing、efficient stochastic depth、FSDP mixed precision を使う。iBOT implementation と比べて `2x` faster、memory `1/3` と著者は報告する（`approach.tex`）。stochastic depth は drop rate `d=40%` で dropped residual の計算自体を skip する。
- **distillation**: 小モデルは frozen `ViT-g/14` を teacher とし、masking と stochastic depth を外し、iBOT loss を two global crops に適用する。最終モデルは student EMA を使う。ViT-L/14 は scratch より distill が全 12 benchmark で良いと報告される（`approach.tex`, `ablation.tex` Fig. `fig:distillation`）。

## 実験・結果

- **データセット / ベンチマーク**: ImageNet-1k/ReaL/V2, ImageNet-A/R/C/Sketch, iNaturalist 2018/2021, Places205, SimCLR 由来の 12 transfer classification benchmarks, Kinetics-400/UCF-101/SSv2, Oxford/Paris/Met/AmsterTime, ADE20k/CityScapes/Pascal VOC, NYU Depth V2/KITTI/SUN RGB-D, Dollar Street, Casual Conversations を評価に使う（`experiments.tex`, `fairness.tex`, `supp-eval-datasets.tex`）。
- **比較対象 / baseline**: SSL baseline は MAE, DINO, SEERv2, MSN, EsViT, Mugs, iBOT。weakly-supervised baseline は CLIP, OpenCLIP, SWAG, EVA-CLIP で、ImageNet 以外では OpenCLIP-G を代表比較として使う（`experiments.tex`）。
- **指標**: classification は Top-1 accuracy, kNN, linear probing。robustness では ImageNet-C のみ lower-is-better の corruption metric。retrieval は mAP/GAP/ACC。semantic segmentation は mIoU。depth は RMSE。fairness は income / region / group ごとの classifier performance と label association を見る。
- **主な結果**: ImageNet-1k linear evaluation では `DINOv2 ViT-g/14` が kNN `83.5`, val `86.5`, ReaL `89.6`, V2 `78.4` で、iBOT ViT-L/16 の val `82.3` より `+4.2%`、OpenCLIP ViT-G/14 の `86.2` より `+0.3%`、EVA-CLIP ViT-g/14 の `86.4` より `+0.1%` と本文は述べる（`experiments.tex` Table `tab:lin-inet1k`）。V2 について本文は EVA-CLIP 比 `+1.1%` と書くが、表値は `78.4` と `77.4` である。
- **主な結果**: fine-tuning sanity check では ViT-g/14 が resolution `224` で linear `86.5` から finetuned `88.5`、resolution `448` で `86.7` から `88.9` へ上がる。著者は、frozen でも強く fine-tuning でも崩れないため `finetuning is optional` と解釈する（Table `tab:ft-inet1k-alone`）。
- **主な結果**: robustness では ViT-g/14 が Im-A `75.9`, Im-R `78.8`, Im-C `28.2` lower, Sketch `62.5`。OpenCLIP-G より Im-A と Im-C では良いが、Im-R と Sketch では OpenCLIP-G が上である（Table `tab:robustness`）。
- **主な結果**: iNaturalist 2018/2021 では ViT-g/14 が `81.6` / `85.7` で OpenCLIP-G より `+8.6%` / `+9.7%`。Places205 は `67.5` で OpenCLIP-G の `69.8` より `-2.3%`。video では K400 `78.4`, UCF-101 `91.2`, SSv2 `38.3` で、SSv2 は iBOT ViT-L/16 の `38.7` がわずかに上である（Table `tab:finegrained_video`）。
- **主な結果**: 12 transfer classification benchmarks の平均は ViT-g/14 が `92.1`、OpenCLIP-G が `91.9`。ただし SUN は `78.7` vs `84.0`、Cars は `91.4` vs `96.1` で OpenCLIP-G が上であり、本文も `SUN (-5.3%)` と `Cars (-4.7%)` を例外として挙げる（Table `tab:finegrained`）。
- **主な結果**: instance recognition では Oxford-Hard mAP が ViT-L/14 で `54.0`、OpenCLIP-G は `19.7`、iBOT は `12.7`。本文は Oxford-Hard で SSL より `+41%` mAP、weakly-supervised より `+34%` mAP と述べる（Table `tab:retrieval`）。
- **主な結果**: semantic segmentation では ViT-g/14 が ADE20k `49.0` linear / `53.0` +ms、CityScapes `71.3` / `81.0`、Pascal VOC `83.0` / `86.2`。ViT-Adapter + Mask2Former に frozen ViT-g/14 を入れると ADE20k `60.2` mIoU で、InternImage の `62.9` に近いと著者は述べる（Table `tab:semseg`）。
- **主な結果**: depth estimation では ViT-g/14 + DPT が NYUd `0.279`, KITTI `2.11`, NYUd -> SUN RGB-D `0.338` RMSE。表上の reference SoTA はそれぞれ `0.330`, `2.10`, `0.421` なので、NYUd と SUN transfer では上回り、KITTI はほぼ同水準でわずかに劣る（Table `tab:depth`）。
- **著者が主張する貢献**: discussion では、DINOv2 は wide range of benchmarks で weakly-supervised alternatives との gap を fine-tuning なしに閉じる最初の SSL work だと主張する。また、性能要因を improved recipe, larger model scale, larger dataset, distillation に分ける（`tmlr.tex`）。

## 妥当性と限界

- **この主張を支える根拠**: 評価範囲が image-level, instance-level, pixel-level, video, fairness にまたがり、主に frozen features を使っているため、「backbone に情報がすぐ読める形で入っている」という主張と評価設計が対応している。さらに、training recipe, data source, loss components, distillation, resolution の ablation があり、最終性能を支える要素を部分的に分解している（`ablation.tex`）。
- **この主張を支える根拠**: データ ablation では `LVD-142M` が ImageNet-22k に対して INet-1k は `85.8` vs `85.9` と同程度だが、Im-A `73.9` vs `73.5`, ADE-20k `47.7` vs `46.6`, Oxford-M `64.6` vs `62.5`, iNat2018 `82.3` vs `81.1`, iNat2021 `86.4` vs `85.6`, Places205 `67.6` vs `67.0` で上回る。uncurated data は INet-1k `83.3`, Im-A `59.4`, Oxford-M `54.3` などで劣り、著者の curation 仮説を支えている（Table `tab:ablation-data`）。
- **この主張を支える根拠**: loss ablation では KoLeo が Oxford-M `55.6 -> 63.9` に効き、MIM/iBOT objective が ADE-20k `44.2 -> 47.1` に効く。distillation では ViT-L/14 scratch が INet-1k `84.5`、distill が `86.3` であり、全 12 benchmarks で distill が scratch を上回る（Tables `tab:koleo`, `tab:ibot`, Fig. `fig:distillation`）。
- **著者が認めている limitations / future work**: Dollar Street では DINOv2 ViT-g/14 が Europe `89.7` に対し Africa `74.0`、high income `90.5` に対し low income `67.4` で、著者は Western countries と wealthy households への bias が残ると述べる。なお本文は Europe 比の Africa drop を `25.7%`、income buckets の差を `31.7%` と書くが、Table `tab:dollar` の単純な percentage-point 差は Europe-Africa が `15.7`、high-low が `23.1` である（`fairness.tex` Table `tab:dollar`）。
- **著者が認めている limitations / future work**: Casual Conversations では Non-Human は `0.0`、Crime は最大 `0.2` の小さい値に留まるが、Possibly-Human が男性で多く trigger され、Beard class の影響だと説明される。著者は、より徹底した bias evaluation で flaws が見つかる可能性を認めている（`fairness.tex` Table `tab:gsa`）。
- **著者が認めている limitations / future work**: future work では、visual features を word tokens のように処理する language-enabled AI system に使う計画が述べられる。また、より大きな model/data scale で object parts や scene geometry のような property がさらに emergent になる可能性を期待しているが、これは将来計画であり実証済み結果ではない（`tmlr.tex`）。
- **読者として注意すべき点**: `pretraining with no supervision` は loss がラベルやテキストを使わないという意味であり、データ curation は curated datasets を query として使う。したがって、完全に「任意の raw web data だけ」から自律的にデータ分布を決めているわけではない。
- **読者として注意すべき点**: OpenCLIP-G/EVA-CLIP との差は ImageNet linear では小さい。さらに OpenCLIP-G は Im-R, Sketch, Places205, SUN, Cars で上回るため、「多くのタスクで同等または上回る」は、タスクごとの勝敗を確認して読むべき主張である。
- **読者として注意すべき点**: compute cost は大きい。`DINOv2-g` の再学習は A100-40GB `22,016` GPU-hours、`9.7 MWh`, `3.7` tCO2eq と見積もられ、project 全体は `0.5k` から `1k` tCO2eq、約 `200k` GPU-days と著者は報告する（`carbon.tex`）。
- **追加で確認したい実験 / 疑問**: curation query から ImageNet 系を外した場合にも同じ generality が出るか、text-guided model に KoLeo や同様の patch-level signal を加えた場合に retrieval/depth の差が縮むか、という点は TeX 中では実験されていない読者側の疑問である。

## 用語メモ

- **DINOv2**: 本論文で提案される pretrained image encoder family。`ViT-S/14`, `ViT-B/14`, `ViT-L/14`, `ViT-g/14` を含む。
- **LVD-142M**: curated datasets を query とする image retrieval で作った `142,109,386` images の pretraining dataset。論文中の `\LaViDa`。
- **general-purpose visual features**: image distributions と tasks をまたいで fine-tuning なしに使える特徴、という意味で使われる。
- **frozen features**: backbone weights を固定し、linear classifier や decoder だけを学習する評価設定。特徴自体がどれだけ「readily available」かを見る。
- **text-guided pretraining**: caption や aligned text-image corpus を使って視覚特徴を学ぶ方法。CLIP/OpenCLIP が代表比較対象。
- **weakly-supervised model**: 本論文では CLIP, OpenCLIP, SWAG, EVA-CLIP など、ラベルやテキスト由来の弱い教師を使う比較対象を指す。
- **DINO loss**: class token に対する teacher-student cross-entropy。image-level feature の主な学習信号。
- **iBOT loss / MIM objective**: masked patch token に対する teacher-student prediction。patch-level feature と dense task に効く要素。
- **prototype scores**: DINO/iBOT head が出す score vector。softmax により $p_s$, $p_t$ の分布になる。
- **Sinkhorn-Knopp centering**: teacher 側の分布を batch normalization 的に調整する SwAV 由来の手続き。本論文では `3` iterations。
- **KoLeo regularizer**: batch 内特徴を広げる正則化。Oxford-M retrieval の改善で重要性が示される。
- **sequence packing**: large crop と small crop から生じる長さの異なる token sequences を concatenate し、block-diagonal attention mask で相互干渉を防ぐ高速化。
- **efficient stochastic depth**: dropped residual を mask するだけでなく計算自体を skip する実装。drop rate `40%` が使われる。
- **FSDP**: student, teacher, AdamW moments などで必要な巨大な model replicas を GPU 間で shard する仕組み。
- **distillation**: scratch で小モデルを学習する代わりに、frozen ViT-g teacher の出力を小モデルに再現させる学習。
- **+ms**: segmentation evaluation の boosted linear setup。last 4 layers の patch tokens、resolution `640`、multiscale test-time augmentations を使う。
- **DPT**: depth estimation で frozen ViT patch tokens の上に置く decoder。Table `tab:depth` の主要設定。

## 読む順番の提案

- まず `tmlr.tex` の `\input` 順を見て、本文が `abstract -> intro -> related -> data -> approach -> ablation -> experiments -> fairness -> carbon -> Future work and Discussion` で構成されることを確認する。正規ノートの Summary はこの順序に沿っている。
- 次に `abstract.tex` と `intro.tex` を読み、`general-purpose visual features`, `without finetuning`, `self-supervised pretraining alone` という主張の範囲を押さえる。正規ノートの「問題」と「貢献」に対応する。
- 手法は `data.tex` と `supp-data.tex` Table `tab:lavida-details` を先に読むとよい。`LVD-142M` が何から作られ、どこで `ImageNet-1k train`, `ImageNet-22k`, `Google Landmarks`, fine-grained / segmentation / depth / retrieval datasets を使うかが分かる。
- その後 `approach.tex` を読み、DINO loss, iBOT loss, Sinkhorn-Knopp, KoLeo, high-resolution adaptation, efficient implementation, distillation を確認する。正規ノートの「DINO+iBOT+SwAV+KoLeo」「学習効率」「distillation」に対応する。
- 結果は `experiments.tex` の Tables `tab:lin-inet1k`, `tab:robustness`, `tab:finegrained_video`, `tab:finegrained`, `tab:retrieval`, `tab:semseg`, `tab:depth` を順に見る。正規ノートの Takeaway にある数値はここで裏取りできる。
- 主張の妥当性は `ablation.tex` の Tables `tab:ibot-dino`, `tab:ablation-data`, `tab:koleo`, `tab:ibot` と Fig. `fig:distillation`, `fig:res` で確認する。正規ノートの「強い視覚 backbone を作るレシピ」という読み方につながる。
- 最後に `fairness.tex`, `carbon.tex`, `tmlr.tex` の Future work and Discussion を読む。正規ノートの Critical Thoughts にある bias, carbon, future work の根拠はここにある。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2304.07193v2/`
- 正規ノート: `notes/arXiv-2304.07193v2.md`
