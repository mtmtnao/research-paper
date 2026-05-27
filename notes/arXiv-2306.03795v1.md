# AI-Supported Assessment of Load Safety

- arXiv: https://arxiv.org/abs/2306.03795
- source: ../papers/arXiv-2306.03795v1/
- authors: Julius Schöning, Niklas Kruse
- venue / year: TeX 中では `arXiv '2x` / `202x`（所属: Osnabrück University of Applied Sciences, Germany）
- tags: [computer-vision, CNN, image-classification, logistics, industrial-AI, load-safety]
- read_date: 2026-05-13
- rating:

---

## Summary（著者の主張）

- **問題**: 物流会社では出庫前にドライバ/loadmaster がトラック荷台の写真を撮り品質管理部門が積載安全性を目視確認するが、量が多く（独 2020 年の警察検査 11,371 件中 1091 件 = 9.6% が積載安全違反）人手では捌けない。さらに現場で撮られる写真は暗い・ぼけてる・荷台全体が写ってない等で「そもそも判定に使えない画像」が混ざる。
- **手法**: 写真を I) 安全に積載、II) 不安全に積載、III) 使用不能画像 の 3 クラスに分ける。多クラス分類ではなく **2 段の二分木**（Stage1: 使用可/不能、Stage2: 安全/不安全）として解く。CNN は 3 種類使用: 深い InceptionV3, ResNet101 と、AlexNet を簡略化した自前の浅い LogisticNet（第1 Conv の kernel を 11x11 から 3x3 にし最終 Dense を 2 クラスに）。入力解像度は各モデルの native 解像度（227, 244, 299）と、receptive field 計算上の上限の手前でハードウェア都合により 800x800 にキャップした高解像版で比較。Data augmentation は y 軸フリップ・軽い回転・明度/色のランダム調整のみ（クラス意味を壊さない範囲）。300 epoch 走らせて overfitting 開始 epoch を確認し、その手前で再学習する 2 度焼きのプロトコル。
- **結果**:
  - Stage1（使用可 vs 使用不能, Tab.1）: LogisticNet 227x227 で Recall 98% / Precision 95% と最良。ResNet101 244x244 は R94/P91、800x800 で R92/P89。InceptionV3 299x299 は R94/P92、800x800 で R93/P90。**3 アーキテクチャ全てで Precision・Recall とも 90% 超**。
  - Stage2（安全 vs 不安全, Tab.2）: 一気に崩れる。最良は ResNet101 800x800 で F1=0.59, MCC=0.20。InceptionV3 800x800 は F1=0.52, MCC=0.13。低解像版は更に悪い（InceptionV3 299x299 は F1=0.43, MCC=0.09）。confusion matrix（Fig.4）でも InceptionV3 299 では class I を 72/100、class II を 37/100 しか正解できず、800 にしても class I 61, class II 51 と均衡しただけ。ResNet101 低解像は class I 45/100, class II 62/100。
  - 学習中 overfitting は早期（InceptionV3 で epoch 50 前後、ResNet101 で 90 前後、LogisticNet で 49）に出るため早期打ち切りが必須。
- **貢献**: (1) 物流会社の実写真 5712 枚（3456x4608, RGB; I:1813 / II:2355 / III:1544）の real-world データセットでの実証、(2) 使用不能画像のフィルタリングを「現場で導入価値あり」と切り出して二段木で解いた点、(3) deep（InceptionV3/ResNet101）と shallow（LogisticNet）を Stage1 で並べ、shallow でも 95%/98% 出る事を示した比較。

## Takeaway（自分にとっての要点）

- 問題を「3 クラス分類」ではなく「最初に使えない画像を弾く + 中身の判定」に分けたのは現場発想として真っ当。Stage1 だけでも倉庫側で「撮り直し要求」を自動化できれば品管部門のボトルネックは解消するという論調で、**Stage2 が失敗していても production 価値はある**と切り分けて主張している。
- Stage1 で LogisticNet（227x227 の AlexNet 派生）が深い ResNet101/InceptionV3 より良かったのが意外。TeX で確認できる事実は、LogisticNet が Stage1 で Recall 98% / Precision 95% と最良だったこと、および LogisticNet が「simple model でも解けるか」を見るための浅い AlexNet 派生として設計されたこと。
- 解像度を上げる（native → 800x800）と Stage1 はわずかに精度が下がる（ResNet Recall 94→92, Inception Recall 94→93）。一方 Stage2 では InceptionV3 の F1 は 0.43→0.52、ResNet101 の F1 は 0.57→0.59 に上がる。**「画像が使えるか」の判定は大域特徴、「安全かどうか」の判定は局所詳細が効く**という解釈は評者補足。
- 著者は overfitting が早期に起きるため epoch 数を短くして再学習している。データ追加は「significant effort」が必要で、将来作業として "hybrid dataset, a mixture of computer-generated and real-world images" を明記している。
- Stage2 が崩れた理由について、著者は **クラス I と II の交差・boundary cases** を一因として挙げている。MCC が 0.2 を超えない点から、ラベル境界の曖昧さが性能上限に関わる可能性がある、という読みは評者補足。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 物流現場の実画像で評価しており、academic dataset → real world のドメインギャップを最初から回避している（著者も Wuest2016, Lee2020 を引いて強調）。
  - 「Stage1 だけでも価値がある」と切り分けて主張する誠実さ。Stage2 で MCC 0.2 を「うまくいきました」と糊塗していない。
  - Conclusion で AI 単独運用ではなく "still need to be supervised by a human operator" と限界を明示しており、責任ある産業 AI の論調。
- **弱み / 疑問**:
  - 評価が **validation accuracy ベース**で、独立した test set 分割が明記されていない。Recall/Precision の算出母集団が別 test set かどうかは TeX 中に明示されていない。
  - Stage1 の「3 アーキテクチャ全部 90% 超」だけ示しているが、各アーキテクチャの inference cost / latency 比較は TeX 中に無いので、「production で LogisticNet を選ぶべき」という決定的根拠は欠ける（評者補足）。
  - Stage2 で最良の F1=0.59 / MCC=0.20 に留まるため、著者自身も safe / unsafe loaded cargo の認識は satisfactorily に解けなかったと結論している。Stage2 を運用するなら human operator の監督が必要という結論になる。
  - データ拡張が y 軸フリップ + 軽い回転 + 明度/色のみ。TeX は「class を変えない」手法として列挙するが、y 軸フリップが荷台画像の意味を変えないかの個別検討は書かれていない。
  - Stage1 / Stage2 で同じ画像プールを 2 度使う構造だが、Stage2 学習時に使用不能画像をどう扱ったかの記述が薄い（Stage1 で除外された画像で Stage2 を訓練するのか、両方混ぜるのか）。
  - クラスバランスが I:1813 / II:2355 / III:1544 と中程度に偏っているのに、加重損失や resampling の話が無い。State of the Art 節で AbdElrahman2013, Kaur2019 を引いてバランスの重要性を語ってるのに本実験で対策していないのは肩透かし。
  - 著者は Stage2 失敗の原因として「クラス境界の曖昧さ」を挙げているが、その根拠は manual inspection として書かれており、inter-annotator agreement のような定量評価は TeX 中に無い。
  - 表 1 の LogisticNet が「解像度を分けず Recall 98% / Precision 95% だけ書いてある」体裁が他モデルと違い読みにくい（表組みも実は Recall と Precision が左右に並んでいるのか他と意味が違うのか曖昧。TeX には `Recall: \textbf{98\%}` と `Precision: \textbf{95\%}` が 1 行で書かれている）。
- **次に試したいこと**:
  - Stage2 の難しさが「ラベル曖昧」か「モデル能力不足」かを切り分けるため、複数 annotator で再ラベルし inter-annotator agreement を測定 → 上限を可視化（評者補足）。
  - 同じ画像から Stage1 用に低解像、Stage2 用にトラック前後左右の crop を取って Stage2 を局所的特徴の問題に再定義する（評者補足）。
  - LogisticNet を Stage1 専任にして実機 inference 速度を測り、edge デバイス（倉庫の撮影端末）で動かす PoC（評者補足）。
  - Stage2 を classification ではなく **「不安全な箇所を local 検出」する object detection / anomaly localization** に再定義（評者補足）。
  - 著者が将来作業として挙げた合成画像との hybrid データセットを実装し、合成画像比率と Stage2 精度の relationship を出す（hybrid dataset は TeX 根拠あり、比率実験は評者補足）。

## Notes / Quotes

- "In 2020, a total of 11,371 police checks of trucks were carried out, during which 9.6\% (1091) violations against the load safety regulations were detected." (abstract)
- "By visual inspection of the used dataset, the class III) unusable image is easily distinguishable from the classes I) cargo loaded safely and II) unsafely, even for a non-trained human observer." (§Image-Based Load Safety Classification) — 二分木にした論拠。
- データセット: 5712 枚、3456x4608 RGB、I:1813 / II:2355 / III:1544 (§Structure of the Dataset)
- 解像度キャップ理由: "Due to hardware limitations and to allow batch sizes larger than two, the resolution here was capped at 800x800 pixels." (§Architecture Design)
- LogisticNet の AlexNet からの変更点: 初段 Conv kernel を 11x11 → 3x3、最終 Dense を 1000 クラス → 2 クラス、softmax。(§Architecture Design)
- Overfitting epoch: InceptionV3 は予備実験で ~50, 評価では 299 で 58 / 800 で 85, ResNet101 244 で 90 / 800 で 120, LogisticNet 49 (§Evaluation)
- 著者自身の limitation: "recognizing safe and unsafe loaded cargo could not be solved satisfactorily in this work ... an intersection between the classes I) cargo loaded safely and II) cargo loaded unsafely was notable." (§Conclusion)
- 著者自身の運用宣言: "AI can already support the assessment of load safety quite well but still need to be supervised by a human operator." (§Conclusion)
- TeX 中には test set 分割の手順や random seed の記述は明示されていない。
- (verified 2026-05-27) venue/year を TeX で確認できる `arXiv '2x` / `202x` に限定し、arXiv ID 由来の 2023 表記を削除 (AI-SupportedAssessmentLoadSafety.tex)
- (verified 2026-05-27) Takeaway / Critical Thoughts の評者解釈・提案に「評者補足」を明記し、TeX 根拠より強い「4割見逃す」「ほぼランダム」等の表現を削除 (AI-SupportedAssessmentLoadSafety.tex, Table 2, Fig. 4, Conclusion)
- (verified 2026-05-27) Related Papers の Inception/ResNet/AlexNet 関連説明を main.bbl で確認できる範囲に修正 (AI-SupportedAssessmentLoadSafety.bbl)

## Related Papers

- Krizhevsky+ 2017 "ImageNet classification with deep convolutional neural networks" — LogisticNet の土台。
- Szegedy+ 2015 "Going deeper with convolutions" / Szegedy+ 2017 "Inception-v4, inception-ResNet and the impact of residual connections on learning" — InceptionV3 の参照元。
- He+ 2016 "Deep residual learning for image recognition" — ResNet101 の参照元。
- Simonyan+ 2014 VGG — State of the Art 節で参照される CNN 系譜。
- Richter+ 2021/2022 — CNN receptive field 計算で入力解像度上限（ResNet101: 971, InceptionV3: 1311）の根拠。
- Khalifa+ 2021, Taylor+ 2018, Shorten+ 2019 — data augmentation 手法のレビュー、本論文の augmentation 設計の根拠。
- Wuest+ 2016, Lee+ 2020, Chen+ 2021 — academic データを産業応用へ移す難しさの参照。
- Northcutt+ 2021 — ラベルノイズに関する参照、本論文の boundary case 議論の伏線。
- Khan+ 2020, Camgoezlue+ 2021, Novakovic+ 2017 — kernel size を小さくする傾向への参照。
