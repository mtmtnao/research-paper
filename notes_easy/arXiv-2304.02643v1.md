# Segment Anything（画像セグメンテーションを promptable な foundation model として扱う研究）

- arXiv: https://arxiv.org/abs/2304.02643
- 一次ソース: ../papers/arXiv-2304.02643v1/
- 正規ノート: ../notes/arXiv-2304.02643v1.md

---

## 一言で言うと

この論文は、画像セグメンテーションに対して「任意の prompt から valid な mask を返す」promptable segmentation task を定義し、それを解く Segment Anything Model (SAM) と、SAM を使った data engine で作った SA-1B dataset を提案する。著者は、SAM が 23 個の多様な segmentation dataset と edge detection / object proposals / instance segmentation / text-to-mask で zero-shot transfer でき、11M images / 1.1B masks の SA-1B が foundation model 研究の基盤になると主張している。

## 何を議論する論文か

- **問題設定**: NLP の large language model や CLIP のように、prompt engineering で新しいデータ分布・下流タスクへ zero-shot / few-shot generalization する基盤モデルを、image segmentation でも作れるかを問う。著者の目標は `a foundation model for image segmentation` である。
- **対象範囲 / 仮定**: 対象は pixel-level の segmentation mask 生成で、入力 prompt は foreground / background points、rough box、mask、free-form text など「何を segment するか」を示す情報として扱う。特に single point prompt は曖昧であり、ground truth mask が可能な valid masks をすべて列挙していないことを前提に評価設計が組まれている。
- **既存研究との差分**: 従来の interactive segmentation は人が十分な点を追加して目的 mask に近づけることが主目的で、multi-task segmentation は固定された task set を学習・評価する。SAM は promptable segmentation task により、学習時と異なる下流タスクでも prompt engineering と composition によって使える component を目指す点が違う。
- **この論文で答えたい問い**: 著者は Introduction で三つの問いを明示する。1. zero-shot generalization を可能にする task は何か、2. その model architecture は何か、3. その task/model を支える data は何か、である。

## 背景と前提

- **Foundation model**: TeX では Bommasani et al. の定義を引き、`trained on broad data at scale and are adaptable to a wide range of downstream tasks` という見方を参照する。ただし Discussion では、SAM は MAE で初期化されるものの、能力の大部分は large-scale supervised training から来ると述べ、self-supervised learning だけを前提にしていない。
- **Prompt engineering / composition**: CLIP が DALL·E などの component として使われる例を背景に、SAM も detector、gaze point、text encoder など別 component と組み合わせられる segmentation interface を目指す。
- **Segmentation の広さ**: TeX は interactive segmentation、edge detection、super pixelization、object proposal generation、foreground segmentation、semantic segmentation、instance segmentation、panoptic segmentation を関連 task として挙げる。SAM はこれらすべてを直接 supervised training する multi-task model ではなく、prompt を通じて多くの task に適応する model として位置づけられる。
- **Baseline との関係**: single-point 評価では RITM が主 baseline で、付録では FocalClick と SimpleClick も使う。object proposals では ViTDet-H detector を強い baseline とし、instance segmentation では ViTDet-H の box を prompt として SAM に渡す。

## 提案手法

### コアアイデア

論文の中心は task / model / data の三点を同時に設計することにある。

1. **Promptable segmentation task**: 任意の segmentation prompt から valid segmentation mask を返す task。prompt が曖昧な場合でも、少なくとも一つの合理的な object / part / subpart mask を返せば valid とする。
2. **Segment Anything Model (SAM)**: image encoder、prompt encoder、lightweight mask decoder からなる。image encoder は重いが画像ごとに一回だけ計算し、prompt encoder + mask decoder は precomputed image embedding に対して browser CPU で約 50ms で動く。
3. **Segment Anything Data Engine**: assisted-manual、semi-automatic、fully automatic の三段階で mask を集める model-in-the-loop data engine。最終的に released SA-1B は 11M licensed and privacy-preserving images と 1.1B automatically generated masks からなる。

### 重要な定義・数式

TeX 中には損失関数や IoU の内部を展開した中核的な明示式は少ない。以下では、TeX に書かれている定義・演算・閾値を、読解用に式の形で整理する。IoU、focal loss、dice loss 自体の標準的な内部式は TeX では展開されていない。

$$
\text{promptable segmentation task: given any segmentation prompt, return a valid segmentation mask}
$$

**式の意味**: `Segment Anything Task` 節の `return a valid segmentation mask given any prompt` という文章定義を読解用に式風に置いたもの。prompt は point / box / mask / free-form text など、画像中で何を segment するかを示す情報であり、出力は valid mask である。

**記号の定義**:
- `prompt` ... foreground / background points、rough box、mask、free-form text など、segment 対象を示す入力
- `valid segmentation mask` ... 曖昧な prompt の場合も、可能な対象の少なくとも一つに対する合理的な mask

**この論文での役割**: SAM の pre-training objective と zero-shot transfer の interface の両方を定義する。単なる interactive segmentation ではなく、曖昧な single prompt にも valid output を返すことが data engine と evaluation の前提になる。

$$
L_{\mathrm{mask}} = \min_i L_i,\quad i \in \{1,2,3\}
$$

**式の意味**: `Making the model ambiguity-aware` では、single prompt に対して三つの mask を同時に予測し、training では ground truth と各 mask の loss を計算して、最も小さい loss だけを backpropagate すると説明される。この式はその `only backpropagate from the lowest loss` を読解用に書いたもの。

**記号の定義**:
- $L_i$ ... $i$ 番目の predicted mask と ground truth mask の mask loss
- $i \in \{1,2,3\}$ ... default で出す三つの mask。TeX では nested masks の典型として whole / part / subpart が挙げられる
- $L_{\mathrm{mask}}$ ... ambiguity-aware training で実際に勾配を流す mask loss

**この論文での役割**: single point prompt が shirt / person のように曖昧な場合、単一出力 model が平均的な mask を学習する問題を避ける。object proposal でも multi-mask output が recall に効くという仮説が置かれ、`single out.` ablation と比較される。

$$
L = 20 L_{\mathrm{focal}} + L_{\mathrm{dice}} + 1.0 L_{\mathrm{IoU\ head}}
$$

**式の意味**: `Losses` では mask prediction を focal loss と dice loss の linear combination で supervision し、比率は focal : dice = 20 : 1 とされる。また IoU prediction head は predicted IoU と predicted mask の ground-truth IoU の mean-square-error loss で学習し、constant scaling factor 1.0 で mask loss に加える。

**記号の定義**:
- $L_{\mathrm{focal}}$ ... pixel-level の foreground/background prediction に使われる focal loss
- $L_{\mathrm{dice}}$ ... mask overlap を見る dice loss
- $L_{\mathrm{IoU\ head}}$ ... IoU prediction head の mean-square-error loss
- $20$ と $1.0$ ... TeX の `20:1 ratio` と `constant scaling factor of 1.0`

**この論文での役割**: SAM は mask 自体の品質と、複数 mask の ranking に使う estimated IoU の両方を学習する。fully automatic stage では predicted IoU score が confident mask の filter にも使われる。

$$
N=\{1,2,3,5,9\}
$$

**式の意味**: 付録 `Zero-Shot Single Point Valid Mask Evaluation` で、$N$ point prompts 後の prediction と ground truth mask の IoU を測るときの $N$ の集合として明示されている。

**記号の定義**:
- $N$ ... iterative point prompts の数
- $\{1,2,3,5,9\}$ ... 評価で報告する prompt 数

**この論文での役割**: SAM は少数 prompt、特に 1 point prompt に焦点を当てる。従来 interactive segmentation のように最大 20 points で目標 IoU に達するまで測る protocol とは異なり、real-time prompt processing と ambiguous prompt の能力を評価する。

$$
\mathrm{IoU}(M_{-1}, M_{+1}) \ge 95.0
$$

**式の意味**: 付録 `Automatic Mask Generation Details` の stable mask filter を式にしたもの。同じ soft mask から logits -1 と +1 で threshold した二つの binary masks の IoU が 95.0 以上なら stable として残す。本文側では probability map を $0.5-\delta$ と $0.5+\delta$ で thresholding した結果が似ている mask を stable と説明している。

**記号の定義**:
- $M_{-1}$ ... soft mask logits を -1 で threshold して得る binary mask
- $M_{+1}$ ... soft mask logits を +1 で threshold して得る binary mask
- $\mathrm{IoU}$ ... intersection-over-union。TeX では名称・指標として使われ、内部式は展開されていない
- $95.0$ ... stable mask として残す threshold

**この論文での役割**: SA-1B の fully automatic masks の品質管理に使われる。confident filter は predicted IoU score threshold 88.0、NMS threshold は 0.7、画像の 95%以上を覆う mask は除外される。

### 実装 / アルゴリズム上の要点

- **Image encoder**: MAE-pretrained ViT-H/16 を使い、1024 x 1024 input から 16x downscaled embedding を得る。出力は 64 x 64 で、1 x 1 conv と 3 x 3 conv により 256 channels へ落とす。ViT は 14 x 14 windowed attention と four equally-spaced global attention blocks を使う。
- **Prompt encoder**: sparse prompts は 256-dimensional vector embeddings にする。point は positional encoding と foreground/background learned embedding の和、box は top-left / bottom-right corner の positional encoding と learned embedding の pair、text は CLIP text encoder。dense prompt である mask は conv で埋め込み、image embedding と element-wise に足す。
- **Mask decoder**: two-layer decoder。各 layer は token self-attention、token-to-image cross-attention、point-wise MLP、image-to-token cross-attention の順で token と image embedding を更新する。最後に image embedding を 4x upsample し、output token から 3-layer MLP で dynamic linear classifier を作って mask を予測する。
- **Ambiguity-aware output**: single prompt では三つの output masks を返す。multiple prompts では曖昧性が少ないため、fourth output token から単一 mask を返す。
- **Training algorithm**: 初期 prompt は foreground point か bounding box を等確率で選ぶ。以降の点は previous prediction と ground truth の error region から選び、前回の unthresholded mask logits も prompt として渡す。合計 11 iterations は initial prompt 1 回、iteratively sampled points 8 回、new external information なしの refinement 2 回。
- **Training recipe**: AdamW、$\beta_1=0.9$、$\beta_2=0.999$、warmup 250 iterations、初期 learning rate $8\mathrm{e}^{-4}$、90k iterations（約 2 SA-1B epochs）、60k / 86666 iterations で learning rate を 10 分の 1 に decay、batch size 256 images、weight decay 0.1、drop path 0.4、layer-wise lr decay 0.8、data augmentation なし。1024 x 1024 input と大きい image encoder のため 256 GPUs で学習する。
- **Data engine**: assisted-manual stage は 120k images から 4.3M masks、平均 annotation time は 34 秒から 14 秒へ低下。semi-automatic stage は additional 5.9M masks / 180k images で total 10.2M masks。fully automatic stage は 11M images 全体に 32 x 32 grid prompts と zoomed-in crops を使い、1.1B masks を生成する。

## 実験・結果

- **データセット / ベンチマーク**: zero-shot single point valid mask evaluation では 23 datasets を使う。ADE20K、BBBC038v1、Cityscapes、DOORS、DRAM、EgoHOS、GTEA、Hypersim、IBD、iShape、LVIS、NDD20、NDISPark、OVIS、PPDLS、Plittersdorf、STREETS、TimberSeg、TrashCan、VISOR、WoodScape、PIDRay、ZeroWaste-f が Table `app:tab:datasets_all` と Figure `fig:benchmark_examples` に並ぶ。
- **比較対象 / baseline**: single point では RITM を主 baseline とし、追加で SimpleClick と FocalClick も評価する。object proposals では cascade ViTDet-H / DMP、instance segmentation では fully-supervised ViTDet-H を比較対象にする。edge detection では HED、EDETR、Sobel filter、Canny、Felz-Hutt が Table `tab:edges` にある。
- **指標**: single point は mIoU と human rating 1-10。edge detection は ODS、OIS、AP、R50。object proposals は LVIS v1 の mask AR@1000。instance segmentation は COCO / LVIS v1 の mask AP。RAI では perceived gender presentation / age group / skin tone ごとの mIoU at 1 point / 3 points。
- **主な結果**: single point mIoU では SAM が 23 datasets 中 16 datasets で RITM を上回り、差は最大約 47 IoU。oracle で三つの mask のうち ground truth に最も合うものを選ぶと、23 datasets すべてで RITM を上回る。human study では SAM の mean ratings が 7-9 に入り、RITM と single-output SAM より高い。
- **Edge detection**: BSDS500 の zero-shot edge detection で SAM は ODS .768、OIS .786、AP .794、R50 .928。EDETR は ODS .840、OIS .858、AP .896、R50 .930、HED は ODS .788、OIS .808、AP .840、R50 .923 であり、SAM は R50 では HED を上回るが ODS/OIS/AP では HED と EDETR に届かない。
- **Object proposals**: LVIS v1 mask AR@1000 は ViTDet-H が all 63.0、SAM が all 59.3、SAM single out. が all 54.9。内訳では SAM は medium 81.6 vs ViTDet-H 80.8、common 63.9 vs 63.3、rare 65.8 vs 58.3 で上回るが、large は 86.9 vs 87.0 でわずかに下回る。TeX 本文には medium and large / rare and common で上回ると書かれているが、表の large 値はほぼ同等で ViTDet-H が 0.1 高い。
- **Instance segmentation**: ViTDet boxes を SAM に渡す zero-shot segmentation module として評価する。COCO は SAM AP 46.5 vs ViTDet-H 51.0、LVIS v1 は SAM AP 44.7 vs ViTDet-H 46.6。AP では ViTDet-H が上だが、Figure `fig:humanstudy:inst` の human ratings では SAM が高く、著者は ViTDet が COCO / LVIS の annotation biases を利用している可能性を述べる。
- **Text-to-mask**: CLIP image embeddings で training prompt を作り、inference では CLIP text encoder の text embeddings を prompt として使う proof-of-concept。`a wheel` や `beaver tooth grille` の qualitative examples があり、失敗時は追加 point prompt が助けになると説明される。著者自身も Discussion で exploratory and not entirely robust とする。
- **Ablation**: data engine stages は各段階で mIoU を上げる。automatic masks のみで学習しても all stages 使用との差は約 0.5 mIoU。SA-1B を 11M images から 1M images（約 10%、約 100M masks）へ減らしても full dataset と comparable。image encoder scaling は ViT-H が ViT-B より大きく改善するが、ViT-L から ViT-H は marginal gains で、著者は `Further image encoder scaling does not appear fruitful at this time.` と書く。
- **著者が主張する貢献**: conclusion では principal contributions を promptable segmentation task、SAM、SA-1B として整理している。加えて TeX は、SAM / SA-1B の release、multiple zero-shot transfer tasks による評価、RAI analysis を示す。Release については、SA-1B は research purposes、SAM は Apache 2.0 の permissive open license と書く。

## 妥当性と限界

- **この主張を支える根拠**: task/model/data を一体で示し、SA-1B の mask quality は 500 images / 約 50k masks の human-corrected masks との比較で 94% の pairs が IoU > 90%、97% が IoU > 75% と報告する。23 datasets の zero-shot single-point 評価、human study、edge / proposal / instance / text-to-mask、ablation、RAI analysis が、多面的に「promptable segmentation model として使えるか」を支える。
- **著者が認めている limitations / future work**: SAM は fine structures を見落とす、小さな disconnected components を hallucinate することがある、zoom-in 系の計算量の大きい method ほど boundary が crisp ではない。多数の points が与えられる interactive segmentation では dedicated methods が上回ると予想される。prompt 処理は real-time でも、heavy image encoder を含む全体は real-time ではない。text-to-mask は exploratory で robust ではない。semantic / panoptic segmentation を簡潔な prompt で実装する方法は不明で、domain-specific tools は各 domain で SAM を上回る可能性がある。
- **読者として注意すべき点**: `valid mask` は ground truth と同一であることを意味しない。single point prompt では評価対象の ground truth が一つしかないため、mIoU が低くても別の valid object を返している可能性がある。そこで oracle 評価と human study が必要になる。一方、human study は 23 datasets 全体ではなく 7 datasets subset と LVIS instance segmentation に限られる。
- **RAI 上の注意**: Table `tab:region` では SA-1B は Europe と Asia & Oceania、middle income countries の比率が高く、Africa、Latin America & Caribbean、low income countries は全 dataset で underrepresented とされる。person segmentation の Table `tab:rai_person` では confidence intervals は older vs middle 以外重なるが、clothing segmentation の Appendix Table `app:tab:rai_clothing` では perceived gender presentation の intervals が disjoint で、masculine の mIoU が高い bias が報告される。
- **追加で確認したい実験 / 疑問**: COCO / LVIS で SAM を fine-tune した場合に AP と human rating の乖離がどう変わるか、domain-specific tools に負けるとされる領域でどの程度の fine-tuning data が必要か、text-to-mask の image-embedding training から text-embedding inference への distribution mismatch がどれほど大きいかは TeX 中には十分に定量化されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Segment Anything (SA) project**: task、model、dataset の三点を同時に導入する研究プロジェクト名。abstract では `a new task, model, and dataset for image segmentation` と説明される。
- **SAM / Segment Anything Model**: promptable segmentation task を解く model。image encoder、prompt encoder、mask decoder の三部品からなり、precomputed image embedding に対して prompt ごとに mask を高速に返す。
- **SA-1B**: 11M licensed and privacy-preserving images と 1.1B high-quality segmentation masks からなる dataset。released dataset は automatically generated masks のみを含み、99.1% が fully automatic stage 由来。
- **Prompt**: segment したい対象を示す入力。点、box、mask、free-form text など。NLP の text prompt だけでなく、spatial information も prompt と呼ぶ。
- **Valid mask**: 曖昧な prompt でも、候補となる対象の少なくとも一つに対する reasonable mask。ground truth annotation と一致することだけを意味しない。
- **Ambiguity-aware**: single prompt に対して三つの mask を出し、minimum loss だけを backpropagate する設計。whole / part / subpart の nested masks を扱うための中心的な工夫。
- **Estimated IoU / IoU prediction head**: predicted mask の quality ranking に使う confidence score。automatic mask generation では confident masks の filtering にも使う。
- **Data engine**: model-in-the-loop で annotation と model improvement を繰り返す仕組み。assisted-manual、semi-automatic、fully automatic の三段階。
- **Zero-shot transfer**: SAM の training task そのものではない task や未見の image distributions に、追加学習なしで prompt engineering により適用すること。CLIP の用語法に従うと TeX に明記されている。
- **Oracle evaluation**: SAM の三つの predicted masks のうち ground truth と最も合うものを選ぶ評価。single point prompt の曖昧性が automatic metric に与える影響を見るために使われる。
- **AR@1000**: object proposal generation の average recall。LVIS v1 で最大 1000 masks/proposals を使って評価する。
- **ODS / OIS / AP / R50**: BSDS500 edge detection の標準指標。R50 は recall at 50% precision。

## 読む順番の提案

- まず `segany.tex` の abstract と Introduction を読み、論文が task / model / data の三点を同時に問題化していることを確認する。正規ノートでは `Summary（著者の主張）` の最初の三つの bullet に対応する。
- 次に `Segment Anything Task` を読み、`valid mask` と ambiguous prompt の定義を押さえる。ここを読まないと、single point mIoU と human study の差が理解しにくい。
- その後 `Segment Anything Model` と Appendix `Segment Anything Model and Task Details` を読む。特に image encoder、prompt encoder、mask decoder、`Making the model ambiguity-aware`、`Losses`、`Training algorithm`、`Training recipe` が正規ノートの architecture / training 記述につながる。
- Data engine と Dataset の節では、4.3M masks / 120k images、additional 5.9M masks / 180k images、11M images / 1.1B masks、99.1% automatically generated、94% IoU > 90% を確認する。Appendix `Automatic Mask Generation Details` で 32 x 32 grid、zoomed-in crops、IoU threshold 88.0、stability 95.0、NMS 0.7 を裏取りする。
- 実験は `Zero-Shot Single Point Valid Mask Evaluation` から読む。Figure `fig:benchmark_exps` と Appendix Table `app:tab:datasets_all` が、正規ノートの 23 datasets / RITM / human study 記述に対応する。
- 最後に Tables `tab:edges`、`tab:proposals`、`tab:instance_segmentation`、Figures `fig:humanstudy:inst`、`fig:ablations`、Discussion `Limitations` を確認すると、性能主張・弱点・ablation のバランスが分かる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2304.02643v1/`
- 正規ノート: `notes/arXiv-2304.02643v1.md`
