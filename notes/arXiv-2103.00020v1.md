# Learning Transferable Visual Models From Natural Language Supervision

- arXiv: https://arxiv.org/abs/2103.00020
- source: ../papers/arXiv-2103.00020v1/
- authors: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (OpenAI)
- venue / year: TeX 中には明示なし（`icml2020.sty` を `[accepted]` オプションで使用）
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
- **貢献**: (1) 400M ペアで学習した contrastive image-text pretraining 法 CLIP を提案、(2) 30+ データセットで zero-shot / linear probe / robustness / 人間比較を評価、(3) text encoder が class 名・説明から zero-shot 線形分類器を合成する方法を示した、(4) Broader Impacts として FairFace でのバイアス検証、surveillance（CCTV、CelebA）の能力評価、データ重複（duplicate）の影響分析を提示、(5) コードと事前学習重みを公開（github.com/OpenAI/CLIP）。

## Takeaway（自分にとっての要点）

- caption prediction / BoW prediction ではなく「画像とテキスト全体の対応を当てる」contrastive objective に替えると、zero-shot ImageNet transfer の効率がさらに 4x 改善したと報告されている（Figure `compare_objective_fig`）。同じ箇所で、transformer language model は BoW baseline より 3x 遅いとも書かれている。
- text encoder は「class 名 → 線形分類器の重み」を生成する hypernetwork として解釈できる。これにより**カテゴリを増減・変更しても再学習不要**で、推論時に文を埋め込んでキャッシュするだけで amortize できる。
- prompt engineering と 80 種の prompt ensembling は ImageNet accuracy を almost 5% 改善し、`"A photo of a {label}."` だけでも +1.3% と報告されている。
- ImageNet への logistic regression 適合は ImageNet accuracy を +9.2%（85.4%）に上げる一方、average accuracy under distribution shift は slightly decreases と報告されている。TeX は、この挙動が end-to-end fine-tuning でも成立するかを未解決問題としている。
- MNIST で 88%、画素 logistic regression に負ける。著者は「大規模で多様なデータなら全データが effectively in-distribution になる」という仮定を naive assumption と明記している。
- zero-shot CLIP が ImageNet で 76.2% を出した Visual N-Grams 比較について、著者自身が direct methods comparison ではないと注釈している。TeX 上の差分は 10× larger dataset、nearly 100× compute per prediction、likely over 1000× training compute、transformer-based model。
- bias: FairFace で `Black` 画像の approximately 14% が non-human（animal/chimpanzee/gorilla/orangutan）に誤分類、`child` カテゴリ追加で 0-20 歳の crime/non-human への誤分類が大幅減。著者は class design が model performance と unwanted biases / behaviour を左右しうると述べている。
- 著者は zero-shot CLIP が overall SOTA に達するには around a 1000x increase in compute が必要で、current hardware では infeasible と書いている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 400M (image, text) pairs、contrastive objective、自然言語 prompt による zero-shot classifier synthesis を組み合わせ、over 30 existing datasets で評価している。
  - robustness（natural distribution shift）を Taori et al. の effective / relative robustness の枠組みで議論し、zero-shot CLIP が robustness gap を最大 75% 縮める一方で ImageNet への supervised adaptation が average distribution-shift accuracy を改善しないことを示している。
  - Broader Impacts で FairFace、denigration harms、class design、surveillance、CelebA identity recognition を具体的な数値つきで扱っている。
  - linear probe で 66 モデル × 27 dataset = 1782 評価を行ったと明記し、個別スコアを Table `tab:linear-probe-big-table` に出している。
- **弱み / 疑問**:
  - WIT の構築説明は、400M pairs、variety of publicly available sources、500,000 queries、up to 20,000 pairs per query、query list の作り方に限られている。TeX 中にデータセット公開や完全な分布表の記述は見当たらない。
  - 著者自身 limitations で認めているが、**評価データセットの選択が CLIP の開発と co-adapted**（"haphazardly assembled collection of 27 datasets"）。真の zero-shot ではない（dev 中に val を何度も見ている）。
  - 1000x compute で SOTA に届くという記述は著者の estimate であり、TeX 中で実験された範囲は 5 ResNet CLIP models の 44x compute range。
  - few-shot で zero-shot から性能が下がる問題は limitations に明記されている。L2 penalty toward generated weights も試したが、hyperparameter optimization が大きな regularizer を選び、結果が「just the zero-shot classifier」になったと書かれている。
  - Visual N-Grams との比較は Table `zeroshot_table` と本文で強調されるが、著者は direct methods comparison ではなく contextualizing performance だと明記している。
  - bias 評価は FairFace の race / gender categories を使うが、著者は footnote でそれらのカテゴリ自体の問題を明記している。Broader Impacts も「initial efforts」「limited in scope」としている。
  - surveillance 節では CelebA 100 classes 59.2%、1k classes 43.3% を示し、production level models とは competitive でないと明記する一方、zero-shot identification based on names inferred from pre-training data である点を noteworthy としている。
- **次に試したいこと**:
  - linear probe での dataset-specific 適合が robustness を下げる現象を fine-tuning（end-to-end）でも検証し、zero-shot / few-shot / fully supervised の差がどこから来るかを切り分ける。著者自身が robustness 節で未解決問題として挙げている。
  - contrastive と generative の joint training。著者は limitation で、CLIP の効率と caption model の柔軟性を組み合わせる simple idea worth trying と述べている。
  - CLIP の strong zero-shot performance と efficient few-shot learning を組み合わせる方法。著者は Future work が必要と明記している。

## Notes / Quotes

- "the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs" (abstract)
- "swapping the predictive objective for the contrastive objective of CLIP further improves efficiency another 4x" (Figure `compare_objective_fig`, Approach §2.3)
- "every step of CLIP pre-training can be viewed as optimizing the performance of a randomly created proxy to a computer vision dataset which contains 1 example per class and has 32,768 total classes defined via natural language descriptions" (§Zero-Shot Transfer)
- "Zero-shot CLIP outperforms few-shot linear probes" — 4-shot LR と同等、16-shot LR にも近い（Figure `zeroshot_vs_fewshot`）
- "All zero-shot CLIP models improve effective robustness by a large amount and reduce the size of the gap between ImageNet accuracy and accuracy under distribution shift by up to 75%" (§Robustness)
- "adapting CLIP to the ImageNet distribution increases its ImageNet accuracy by 9.2% to 85.4% overall ... average accuracy under distribution shift slightly decreases" (§Robustness)
- "Our methodology has several significant limitations. Despite our focus on zero-shot transfer, we repeatedly queried performance on full validation sets to guide the development of CLIP." (§Limitations) — 自己批判。
- "we estimate around a 1000x increase in compute is required for zero-shot CLIP to reach overall state-of-the-art performance. This is infeasible to train with current hardware." (§Limitations)
- "CLIP only achieves 88% accuracy on the handwritten digits of MNIST. An embarrassingly simple baseline of logistic regression on raw pixels outperforms zero-shot CLIP." (§Limitations)
- FairFace で `child` カテゴリを加えると 0-20 歳の crime/non-human 誤分類が大幅減（Table `age_bias_table`）— class design が bias / behaviour に関わるという著者の記述に対応。
- CCTV 画像で coarse zero-shot 91.8%、`close' distractor を入れた stress test では 51.1% に低下（§Surveillance）。
- 8 モデル: RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14（+ @336px）。
- (verified 2026-05-27) venue/year から TeX 中に明示のない "ICML 2021" と "arXiv 2021-03" を削除し、`icml2020.sty` `[accepted]` 使用のみに限定 (clip_paper.tex)。
- (verified 2026-05-27) Takeaway / Critical Thoughts から TeX に無い外部知識・比喩的評価・後発研究名を削除し、Limitations / Robustness / Broader Impacts に根拠がある記述へ修正 (clip_paper.tex)。
- (verified 2026-05-27) WIT、zero-shot / few-shot、linear probe、robustness、FairFace、surveillance の数値を TeX 本文・表と照合 (clip_paper.tex, zero-shot-table.tex, linear-probe-table.tex)。

## Related Papers

- Joulin+ 2016、Li+ 2017（Visual N-Grams）— Introduction と Zero-Shot Transfer で比較される自然言語監督の先行研究。
- VirTex (Desai+ 2020)、ICMLM (Bulent+ 2020)、ConVIRT (Zhang+ 2020) — Introduction で挙げられる image-text representation learning の近接研究。
- Mahajan+ 2018、Kolesnikov+ 2019、Dosovitskiy+ 2020、Xie+ 2020 — Introduction / Representation Learning / Robustness で比較される大規模 pretraining baseline。
- Sohn 2016、van den Oord+ 2018、Zhang+ 2020、Bachman+ 2019、Chen+ 2020 — Approach で contrastive objective / projection / implementation differences の文脈で引用。
- Taori+ 2020、Recht+ 2019、Hendrycks+ 2019/2020、Barbu+ 2019、Wang+ 2019、Shankar+ 2019 — robustness to natural distribution shift の比較対象・評価セット。
- Buolamwini & Gebru 2018、Karkkainen & Joo 2019、Schwemmer+ 2020 — Broader Impacts / Bias の参照元。
- Kornblith+ 2019、Zhai+ 2019 — linear probe / transfer evaluation の参照元。
