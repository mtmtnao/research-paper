# Learning Transferable Visual Models From Natural Language Supervision

- arXiv: https://arxiv.org/abs/2103.00020
- source: ../papers/arXiv-2103.00020v1/
- authors: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (OpenAI)
- venue / year: ICML 2021（TeX は icml2020.sty + `[accepted]` オプション; 会議名 ICML 2021 自体は TeX 中には明示なし）／ arXiv 2021-03
- tags: [contrastive-learning, vision-language, zero-shot, CLIP, representation-learning]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 既存のコンピュータビジョンは事前に固定されたカテゴリ（ImageNet の 1000 クラス、JFT-300M の 18291 クラスなど）を予測するように訓練されており、別の概念を扱うには追加の labeled data が必要で汎化性に乏しい。NLP では生テキストからの大規模事前学習（GPT-3 等）が task-agnostic な転移を可能にしているのに、CV では crowd-labeled dataset への依存が続いている。先行の自然言語監督手法（Visual N-Grams など）は ImageNet で 11.5% 程度しか出せず実用に遠かった。
- **手法**: インターネットから収集した 400M (image, text) ペア（WIT = WebImageText、500,000 クエリで概略クラスバランスし 1 クエリあたり最大 20,000 ペア）に対して、image encoder と text encoder を「バッチ内で正しいペアを当てる」対照学習で同時に訓練する CLIP（Contrastive Language-Image Pre-training）を提案。N×N の組合せから N 個の正ペアの cosine 類似度を最大化、それ以外（N²−N）を最小化する対称交差エントロピー損失。温度 τ は log-parameterized で学習。pretraining 後は class 名を text encoder に通して zero-shot 線形分類器を合成する。image encoder は ResNet（ResNet-D + antialiased pool + attention pooling）5 モデルと ViT 3 モデルの計 8 種、最良は ViT-L/14 を 336px で 1 epoch fine-tune した ViT-L/14@336px。32 epochs、batch 32,768、Adam + cosine schedule、mixed precision、gradient checkpointing。RN50x64 は 592 V100 で 18 日、ViT 最大は 256 V100 で 12 日。
- **結果**:
  - **Zero-shot ImageNet**: Visual N-Grams 11.5% → CLIP 76.2% で ResNet-50 と同等、top-5 で 95%（Inception-V4 並）。aYahoo 72.4→98.4、SUN 23.0→58.5（Table 1）。
  - **30+ データセットの zero-shot 評価**: 27 データセット中 16 で ResNet-50 の線形分類器を上回る。Kinetics700 で ResNet-50 比 +14.5%、UCF101 で +7.7%、STL10 で 99.3%（SOTA）。一方 EuroSAT、PatchCamelyon、CLEVRCounts、GTSRB、KITTI Distance などの専門・抽象タスクでは弱い。
  - **Zero-shot vs few-shot**: zero-shot CLIP は同じ feature space 上の 4-shot logistic regression と一致、16-shot にもほぼ匹敵。データ効率の中央値は 5.4 examples/class、平均 20.8。
  - **Linear probe**: 27 データセットで Noisy Student EfficientNet-L2 を 21/27 で上回り、平均 +5%。ViT は ResNet より約 3 倍計算効率良し。
  - **Robustness (natural distribution shift)**: 7 つの natural shift（ImageNetV2/Sketch/A/R, ObjectNet, Youtube-BB, ImageNet-Vid）で zero-shot CLIP は ImageNet モデルに比べ effective robustness gap を最大 75% 縮める。逆に ImageNet に linear probe で適合させると ImageNet 精度は +9.2% (85.4%) になるが平均 robustness は微減する。
  - **人間との比較**: Oxford-IIIT Pets 37 クラスで zero-shot human 53.7%、CLIP zero-shot 93.5%。人間は 0→1-shot で 54%→76% に大きく伸びるが、CLIP の few-shot 化はむしろ精度を下げる場合がある。
  - **Scaling**: 5 つの ResNet CLIP モデルにおいて zero-shot エラーは compute に対し log-log linear に減少（44× 範囲）。
- **貢献**: (1) 400M ペアで学習した contrastive image-text pretraining 法 CLIP を提案、(2) 30+ データセットで zero-shot / linear probe / robustness / 人間比較を網羅的に評価、(3) 自然言語ベースで「動的に分類器を生成する」インターフェースが画像分類のタスク汎化を大きく前進させることを示した、(4) Broader Impacts として FairFace でのバイアス検証、surveillance（CCTV、CelebA）の能力評価、データ重複（duplicate）の影響分析を提示、(5) コードと事前学習重みを公開（github.com/OpenAI/CLIP）。

## Takeaway（自分にとっての要点）

- 「caption の単語を当てる」から「画像とキャプションの対応を当てる」に変えるだけで zero-shot ImageNet の学習効率は 4x 上がる（Figure 2 のアブレーション）。さらに BoW 予測ベースラインに対して transformer 言語モデルは 3x 遅い。**生成的目的より対照的目的のほうが representation 学習として効率的**という主張の補強。
- text encoder は「class 名 → 線形分類器の重み」を生成する hypernetwork として解釈できる。これにより**カテゴリを増減・変更しても再学習不要**で、推論時に文を埋め込んでキャッシュするだけで amortize できる。
- prompt engineering と 80 種の prompt ensembling だけで ImageNet 5% 改善。安価なため必ず併用する価値がある。`"A photo of a {label}."` だけでも +1.3%。
- 任意 dataset への ImageNet 適合は精度を上げるが robustness は若干下がる。**「pretraining → 固定 zero-shot 評価」のほうが OOD 適用には強い**。supervised fine-tune の dataset-specific な spurious correlation を避けられる、という解釈。
- MNIST で 88%、画素 logistic regression に負ける。「データを大規模化すれば全て in-distribution になる」という思想の脆弱性を著者自身が認めている。**スケールでカバーできない gap が確実にある**ことの実例として覚えておく。
- zero-shot CLIP が ImageNet で 76.2% を出せたのは Visual N-Grams 比で 10× データ・100× per-prediction compute・1000× 訓練 compute・transformer の組合せ（著者自身が "this should not be interpreted as a direct methods comparison" と注釈）。比較は手法そのものよりスケールの効果という側面が強い。
- bias: FairFace で `Black` 画像の 14% が non-human（animal/chimpanzee/gorilla/orangutan）に誤分類、`child` カテゴリ追加で 0-20 歳の crime/non-human への誤分類が大幅減 → **class design 自体が「介入可能なバイアス源」**になるという、定量的にもインパクトある観察。
- 1000x compute 増で SOTA に届くと推定しているが「現行 HW で不可能」と認めている。CLIP のスケーリング曲線はそのまま「open-vocabulary CV はまだ計算上限の内側」ことを示す。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 単一の事前学習で 30+ タスクの zero-shot 性能を一気に押し上げた、CV における GPT モーメント的な成果。インターフェース（自然言語）と訓練目的（対照）と規模（400M）が同時に揃ったときに何が起きるかの実証として記念碑的。
  - robustness（natural distribution shift）と zero-shot 性能を同時に評価し、effective robustness 概念で先行研究（Taori et al.）と直接比較できる枠組みで議論している点が誠実。supervised adaptation がむしろ effective robustness を下げるという反直観的な結果は実用上の含意が大きい。
  - bias / surveillance / data overlap の節を「ablation 並みの厚さ」で書いた点。特に class design（`child` を入れると 0-20 歳の denigration 誤分類が激減）の実験は、技術論文でここまで踏み込んだ介入実験は珍しい。
  - linear probe で 66 モデル × 27 dataset = 1782 評価という規模で baseline と比較しており、結論が cherry-pick に見えない。
- **弱み / 疑問**:
  - データセット WIT が非公開で、再現性に大きな影響。400M ペアの正確な分布・収集方法・フィルタリングは「500K queries・各 20K cap・WordNet 補強」とだけ書かれており検証が困難。
  - 著者自身 limitations で認めているが、**評価データセットの選択が CLIP の開発と co-adapted**（"haphazardly assembled collection of 27 datasets"）。真の zero-shot ではない（dev 中に val を何度も見ている）。
  - 1000x compute で SOTA という主張は単純な log-log 外挿で、実際にその領域でも線形に伸びるかは未検証。task ごとの大きな variance（zs-scaling Figure）を見ると単一の trend line で説明できているか怪しい。
  - few-shot 性能が zero-shot より下がる現象（特に 1-shot, 2-shot 領域）は限界として認めているが、深掘りされていない。L2 prior on zero-shot weights を試して「regularizer が大きすぎて結局 zero-shot」になったというのは消化不良。
  - Visual N-Grams との比較が控えめに「contextualize 用」と書かれてはいるが、結局論文の冒頭で 11.5%→76.2% を強調しており、メソッド比較ではなくほぼスケール比較である事実が読み流されやすい。
  - bias 評価は FairFace の 7 race × 2 gender カテゴリに依存しており、それ自体の問題（footnote で自覚あり）。WIT 由来のバイアスの体系的な評価には届いていない。
  - surveillance 節で「CelebA 100 classes 59.2%、1k classes 43.3%」を「production レベルに劣る」としつつ "noteworthy" と評価しているが、評価尺度が緩い（accuracy のみ、誤同定の害の議論は薄い）。
- **次に試したいこと**:
  - CLIP の zero-shot 性能の各 dataset での誤りを WIT の retrieval で説明する分析。MNIST が崩れる ↔ 訓練データに手書き数字がほぼ無い、というのと同じ整合性チェックを他 dataset でも。
  - prompt ensembling の 80 prompts を学習可能 prefix（prompt tuning）に置き換えたとき、80 prompts ensemble に対する精度差。
  - linear probe での dataset-specific 適合が robustness を下げる現象を fine-tuning（end-to-end）でも検証し、effective robustness が「凍結 zero-shot」固有か「自然言語監督そのもの」由来かを切り分ける。著者自身が future work として言及している箇所。
  - contrastive と generative の joint training（著者が limitation で示唆）— captioning の柔軟性と CLIP の効率の両立。後発の BLIP/CoCa 系で実際に追究されたテーマ。
  - WIT を再現可能な公開コーパス（LAION 等）に置き換えたときの 30+ task scoreboard の再現。実際に OpenCLIP / LAION-400M / LAION-2B でなされたが、scoreboard の各 dataset で何が差を作るかをより細かく分解したい。

## Notes / Quotes

- "the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs" (abstract)
- "swapping the predictive objective for the contrastive objective of CLIP further improves efficiency another 4x" (Figure 2 caption, Approach §2.3)
- "every step of CLIP pre-training can be viewed as optimizing the performance of a randomly created proxy to a computer vision dataset which contains 1 example per class and has 32,768 total classes defined via natural language descriptions" (§Zero-Shot Transfer)
- "Zero-shot CLIP outperforms few-shot linear probes" — 4-shot LR と同等、16-shot LR にも近い（Figure 6）
- "All zero-shot CLIP models improve effective robustness by a large amount and reduce the size of the gap between ImageNet accuracy and accuracy under distribution shift by up to 75%" (§Robustness)
- "adapting CLIP to the ImageNet distribution increases its ImageNet accuracy by 9.2% to 85.4% overall ... average accuracy under distribution shift slightly decreases" (§Robustness)
- "Our methodology has several significant limitations. Despite our focus on zero-shot transfer, we repeatedly queried performance on full validation sets to guide the development of CLIP." (§Limitations) — 自己批判。
- "we estimate around a 1000x increase in compute is required for zero-shot CLIP to reach overall state-of-the-art performance. This is infeasible to train with current hardware." (§Limitations)
- "CLIP only achieves 88% accuracy on the handwritten digits of MNIST. An embarrassingly simple baseline of logistic regression on raw pixels outperforms zero-shot CLIP." (§Limitations)
- FairFace で `child` カテゴリを加えると 0-20 歳の crime/non-human 誤分類が大幅減（Table 8）— class design がバイアスに直接効く実証。
- CCTV 画像で coarse zero-shot 91.8%、`close' distractor を入れた stress test では 51.1% に低下（§Surveillance）。
- 8 モデル: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14（+ @336px）。
- (verified 2026-05-20) メタ情報の "arXiv 2021-02" を "arXiv 2021-03" に修正（arXiv ID 2103.00020 が 2021-03 を示す）。venue を ICML 2021 と明示する根拠は TeX 中に直接無いため、その旨を併記（clip_paper.tex L19 `\usepackage[accepted]{icml2020}`）。Summary/Takeaway/Critical Thoughts の数値・固有名詞は clip_paper.tex (abstract, §Approach, §Experiments, §Robustness, §Limitations, §Broader Impacts, Table 1/8, Figure 2/6) で逐一裏取り済み。

## Related Papers

- Joulin+ 2016（YFCC100M に対する BoW 予測）, Li+ 2017（Visual N-Grams、ImageNet zero-shot 11.5%）— CLIP の直接の前史。
- VirTex (Desai+ 2020), ICMLM (Bulent+ 2020), ConVIRT (Zhang+ 2020) — text からの representation 学習の近接研究。CLIP は ConVIRT の簡略化版と自己定義。
- Mahajan+ 2018（Instagram hashtag, ResNeXt-32x48d）, Kolesnikov+ 2019 (BiT, JFT-300M), Dosovitskiy+ 2020 (ViT) — 大規模 weakly-supervised / supervised pretraining baseline。
- He+ 2016 (ResNet), Tan & Le 2019 (EfficientNet), Xie+ 2020 (Noisy Student EfficientNet-L2) — 比較対象の image encoder/SOTA。
- Sohn 2016 (N-pair loss), van den Oord+ 2018 (InfoNCE), Chen+ 2020 (SimCLR), Bachman+ 2019 — contrastive objective の系譜。
- Radford+ 2018 (GPT-1), Radford+ 2019 (GPT-2), Brown+ 2020 (GPT-3) — task-agnostic pretraining + zero-shot transfer の思想的源流。
- Taori+ 2020 (Measuring Robustness to Natural Distribution Shifts), Recht+ 2019 (ImageNetV2), Hendrycks+ 2019/2020 (ImageNet-A/R, Many Faces of Robustness), Barbu+ 2019 (ObjectNet), Wang+ 2019 (ImageNet Sketch) — robustness 評価セット。
- Buolamwini & Gebru 2018 (Gender Shades), Karkkainen & Joo 2019 (FairFace), Schwemmer+ 2020 — bias 評価の枠組み。
- Kornblith+ 2019 (Better ImageNet → Better Transfer?), Zhai+ 2019 (VTAB) — transfer evaluation の標準。
