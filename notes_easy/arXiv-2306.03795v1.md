# AI-Supported Assessment of Load Safety（産業物流における実画像 CNN 分類の適用可能性評価）

- arXiv: https://arxiv.org/abs/2306.03795
- 一次ソース: ../papers/arXiv-2306.03795v1/
- 正規ノート: ../notes/arXiv-2306.03795v1.md

---

## 一言で言うと

物流会社の出庫前チェックで撮られるトラック荷台写真を、I) cargo loaded safely、II) cargo loaded unsafely、III) unusable image に分類できるかを、二段階の CNN 分類として調べる論文である。著者の主張は、使用可能画像と使用不能画像の分離は実用的な水準で可能だが、安全 / 不安全の分離は十分には解けず、人間の監督が必要だというもの。

## 何を議論する論文か

- **問題設定**: 物流会社では、driver または loadmaster が loading process 後にトラック荷台の写真を撮り、quality management department が積載安全を確認する。写真枚数が多く、画像品質も安定しないため、人間による後段確認が時間的ボトルネックになる。
- **対象範囲 / 仮定**: 対象は logistic company の shipping process 中に撮られた rear view の実画像である。画像は三クラス、I) cargo loaded safely、II) cargo loaded unsafely、III) unusable image に手動分類済みで、CNN はこのラベル付き画像分類問題を解くものとして扱われる。
- **既存研究との差分**: 論文の主眼は、InceptionV3、ResNet101 と、AlexNet ベースの浅い LogisticNet を、load safety assessment という現場画像タスクに適用して評価する点にある。Structure of the Dataset では、academic datasets から industrial deployment への移行が難しいことを背景に、target domain に近い real-world dataset を使う意義を置いている。
- **この論文で答えたい問い**: AI は、積載安全評価に使えない画像を自動で弾けるか。さらに、使える画像について安全な積載と不安全な積載を CNN で識別できるか。浅いモデルでもこのタスクを扱えるか。

## 背景と前提

- **load safety assessment** は、積載物が法的・実務的に安全な状態で固定されているかを確認する業務である。Introduction では、2020 年に 11,371 件の truck checks が行われ、そのうち 1091 件、9.6% が load safety regulations 違反だったと述べる。
- **centralized platform** は、トラック、貨物、交通流などの物流データを集約する基盤として説明される。この基盤に、写真収集だけでなく AI による安全確認を組み込めば、quality management と warehouse の負荷を下げられる、という位置づけである。
- **CNN による画像分類** は、画像中の edges of a piece of cargo、framework of the truck、floor of the truck、colors of labels などの特徴を、supervised training により ANN が選ぶという前提で説明される。
- **データセットの偏りと過学習** は、State of the Art で重要な前提として扱われる。著者は、class imbalance が結果に影響しうること、data augmentation が overfitting 対策として使われることを先行研究に基づいて説明する。
- **評価指標** は、Stage1 では Recall と Precision、Stage2 では F1-Score と MCC で報告される。Table 2 の caption は、MCC について「1.00 would represent a perfect classification」と説明しているが、Precision / Recall / F1 / MCC の計算式は TeX 中には明示されていない。

## 提案手法

### コアアイデア

三クラス分類を一度に解くのではなく、"decision tree of two binary classifications" として分ける。第一段階は入力画像が usable か unusable かを判定し、usable の場合だけ第二段階で I) cargo loaded safely と II) cargo loaded unsafely を判定する。

この分割の根拠は、TeX の "By visual inspection" にある。著者は、class III) unusable image は、I) cargo loaded safely と II) unsafely から、non-trained human observer にとっても容易に区別できると述べる。一方で、I と II の識別はより複雑であり、現在は logistics company の quality management department の人間が行っているので、ANN でも可能性がある、という仮説として扱われる。

使うモデルは三つである。InceptionV3 と ResNet101 は deep ANN、LogisticNet は AlexNet ベースの shallow architecture として導入される。LogisticNet は、simple model がこのタスクを解けるかを見るための比較対象であり、Fig. 2 では Conv2D(K_size=3x3, 11x11, 5x5)、Batch Normalization、MaxPool2D(P_Size=3)、Flatten、Dense(4096)、DropOut(0.5)、Dense(2) からなる構成として描かれる。

### 重要な定義・数式

TeX 中には、目的関数、損失関数、更新式、評価指標の計算式は明示されていない。そのため、このノートでは TeX に無い新しい記号を導入した表示数式は置かず、本文中の分類定義と設計上の数値だけを整理する。

- **三つのクラス**: Abstract、Introduction、Image-Based Load Safety Classification、Structure of the Dataset では、画像を I) cargo loaded safely、II) cargo loaded unsafely、III) unusable image の三クラスに分ける。class I は suitable load safety を示す画像、class II は warehouse では安全とされたが subsequent inspection では unsafe と見なされる画像、class III は blurred、too dark、entire cargo hold が写らない等で quality management が cargo security を確認できない画像である。
- **二段階分類**: Image-Based Load Safety Classification は、multi-class classification の代わりに "decision tree of two binary classifications" を使う選択肢を説明する。第一段階で usable / unusable を判定し、usable image の場合だけ cargo loaded safely / unsafely を判定する。
- **二段階化の根拠**: 著者は "By visual inspection of the used dataset" として、class III) unusable image は class I) と II) から non-trained human observer でも容易に区別できると述べる。さらに、first classification step だけでも logistic service provider に considerable value があると位置づける。
- **解像度設定**: Architecture Design では、InceptionV3 の標準入力を 299x299 pixels、ResNet101 を 244x244 pixels、LogisticNet を 227x227 pixels とする。receptive field に基づく上限は ResNet101 が 971x971 pixels、InceptionV3 が 1311x1311 pixels だが、hardware limitations と batch sizes larger than two のため、高解像度条件は 800x800 pixels に cap される。

### 実装 / アルゴリズム上の要点

- **データセット**: 5712 images。多くは 3456x4608 pixels、RGB color space。class I は 1813 images、class II は 2355 images、class III は 1544 images。
- **撮影条件**: 物流会社の shipping process 中に撮られた、truck's rear view の実画像である。外部の学術データではなく、logistic company で評価に使われた real-world dataset と説明される。
- **Stage1**: class I と II を usable images としてまとめ、class III unusable image と区別する。
- **Stage2**: class I cargo loaded safely と class II cargo loaded unsafely を区別する。Result では ResNet101 と InceptionV3 の結果が報告され、LogisticNet の Stage2 結果は TeX 中に示されていない。
- **データ拡張**: random flip on the y axes、slight random rotation、random brightness modification、random color adjustments を使う。著者は、classes of the images を変えない範囲で選んだと説明する。
- **入力解像度**: InceptionV3 は標準 299x299 と 800x800、ResNet101 は標準 244x244 と 800x800、LogisticNet は 227x227 のみで評価される。
- **LogisticNet**: AlexNet ベースの shallow architecture。Fig. 2 では先頭に Conv2D(K_size=3x3) があり、最終 Dense は 2 classes に減らされる。最後の dense layer の activation function は、他モデルと同じく softmax とされる。
- **学習手順**: 全モデルをまず 300 epochs 学習し、overfitting が見える時点を探す。その後、overfitting 前で止める epoch 数を選び、再学習して評価する。
- **最終学習 epoch**: ResNet101 は 244x244 で 87 epochs、800x800 で 117 epochs。InceptionV3 は 299x299 で 55 epochs、800x800 で 82 epochs。LogisticNet は 46 epochs。

## 実験・結果

- **データセット / ベンチマーク**: 物流会社の実写真 5712 枚。class I: 1813、class II: 2355、class III: 1544。画像は truck's rear view で、多くは 3456x4608 pixels、RGB color space。
- **比較対象 / baseline**: deep ANN として InceptionV3 と ResNet101、shallow architecture として LogisticNet。LogisticNet は AlexNet を土台にした著者側の比較モデルである。TeX 中に、従来手法との同一データセット上の比較ベンチマークはない。
- **指標**: Stage1 は Recall と Precision。Stage2 は F1-Score と MCC。Table 2 は MCC について、1.00 が perfect classification を表すと説明する。
- **主な結果**: Stage1 は比較的高い性能で、unusable image の自動フィルタリングには有望である。一方、Stage2 の safe / unsafe 分類は大きく性能が下がり、著者は satisfactory には解けなかったと結論づける。
- **著者が主張する貢献**: AI-supported assessment of load safety は feasible であり、少なくとも usable / unusable images の区別は centralized logistic platform に組み込む価値がある。ただし safe / unsafe の自動判定は human operator による監督が必要である。

Stage1、つまり usable images と unusable images の区別の結果は Table 1 にある。

| Architecture | Resolution | Recall | Precision |
| --- | ---: | ---: | ---: |
| ResNet101 | 244x244px | 94% | 91% |
| ResNet101 | 800x800px | 92% | 89% |
| InceptionV3 | 299x299px | 94% | 92% |
| InceptionV3 | 800x800px | 93% | 90% |
| LogisticNet | 227x227px | 98% | 95% |

Result 本文は "all CNN architecture achieves precision and recall above 90%" と述べるが、Table 1 では ResNet101 800x800px の Precision が 89% である。数値を読むときは表の値を優先し、この一文は厳密には表と一致しない点に注意する。

Stage2、つまり class I) cargo loaded safely と class II) cargo loaded unsafely の区別の結果は Table 2 にある。

| Architecture | Resolution | F1-Score | MCC |
| --- | ---: | ---: | ---: |
| ResNet101 | 244x244px | 0.57 | 0.18 |
| ResNet101 | 800x800px | 0.59 | 0.20 |
| InceptionV3 | 299x299px | 0.43 | 0.09 |
| InceptionV3 | 800x800px | 0.52 | 0.13 |

Fig. 4 は InceptionV3 の confusion matrix を示す。299x299px では class I を 72/100、class II を 37/100 正しく分類する。800x800px では class I を 61/100、class II を 51/100 正しく分類する。ResNet101 について本文は、低解像度で class I が 45/100、class II が 62/100 正解、高解像度ではほぼ同じだが class II の正解が 3% 増えたと説明する。

学習過程では overfitting の扱いも重要である。ResNet101 244x244 は 300 epochs 後に validation accuracy 87%、最高 93% が epoch 242、overfitting は epoch 90 後。ResNet101 800x800 は 300 epochs 後に 94%、最高 94.5% が epoch 292、overfitting は epoch 120 後。InceptionV3 299x299 は 300 epochs 後 95%、最高 96% が epoch 277、overfitting は epoch 58 後。InceptionV3 800x800 については、TeX では 300 epochs 後 97% と書く一方で、最高点を epoch 280 の 96% と書いており、記述に不整合がある。LogisticNet は 300 epochs 後 88%、最高 89% が epoch 254、overfitting は epoch 49 後である。

## 妥当性と限界

- **この主張を支える根拠**: Stage1 では、LogisticNet が Recall 98% / Precision 95%、InceptionV3 と ResNet101 もおおむね 90% 前後の値を出している。class III unusable image は non-trained human observer でも見分けやすいという観察と、Table 1 の結果が著者の「まず使えない画像を弾く」主張を支える。
- **この主張を支える根拠**: 実験は real-world dataset に基づく。著者は、academic datasets から industrial deployment への transferability が challenging であることを背景に、target domain に近いデータを使った点を重視している。
- **著者が認めている limitations / future work**: Conclusion では、safe / unsafe loaded cargo の認識は本研究では satisfactorily には解けなかったと明記する。manual inspection では class I と class II の intersection が見られ、boundary cases が一因になりうると述べる。
- **著者が認めている limitations / future work**: class I と II の intersection を取り除くこと、computer-generated and real-world images を混ぜた hybrid dataset を作ることが今後の方向として書かれている。最終的な結論も、AI は load safety assessment を支援できるが human operator による supervision が必要、という形である。
- **読者として注意すべき点**: TeX 中には、独立した test set の分割方法、random seed、train / validation / test の比率、評価値の分散は明示されていない。"average results" と書かれるが、何回平均かは読み取れない。
- **読者として注意すべき点**: State of the Art では class imbalance の重要性を述べるが、実験で重み付き損失、resampling、stratified split などを使ったとは書かれていない。class I: 1813、class II: 2355、class III: 1544 という偏りが結果にどう影響したかは未検証である。
- **読者として注意すべき点**: Stage2 のラベル境界が曖昧という主張は、manual inspection に基づく説明であり、inter-annotator agreement のような定量的根拠は TeX 中にはない。
- **追加で確認したい実験 / 疑問**: Stage2 で LogisticNet を評価しなかった理由、各モデルの inference time や deployment cost、独立 test set での再評価、複数 annotator による class I / II の一致率、hybrid dataset の効果を確認したい。これらは読者側の疑問であり、TeX 中に実験結果はない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **load safety assessment**: トラックが出庫する前に、荷物の積み方が法規制と安全要求を満たすかを確認する業務。AI の単独判断ではなく、quality management の支援対象として扱われる。
- **driver / loadmaster**: loading process 後に荷台写真を撮る現場側の担当者。loadmaster は違反時の責任にも関わる存在として説明される。
- **quality management department**: 送られてきた写真を確認し、出庫可否に関わるチェックを行う部門。この部門の workload と bottleneck の解消が導入動機である。
- **class I) cargo loaded safely**: suitable load safety を示し、logistics company の観点で望ましい画像。データセットでは 1813 枚。
- **class II) cargo loaded unsafely**: 倉庫では安全とされたが後続検査では不安全とされる画像。legal requirements を満たさず、liability risks を持つクラス。データセットでは 2355 枚。
- **class III) unusable image**: ぼけ、暗さ、cargo hold 全体が写っていない等により、cargo security を判断できない画像。データセットでは 1544 枚。
- **usable / unusable image**: Stage1 の二値分類で使う区別。usable は class I と II をまとめた側、unusable は class III に対応する。
- **decision tree of two binary classifications**: 三クラスを直接分類するのではなく、usable / unusable、safe / unsafe の二段階に分ける設計。
- **InceptionV3**: deep ANN の一つ。標準入力は 299x299px とされ、本論文では 800x800px との比較も行う。
- **ResNet101**: deep ANN の一つ。標準入力は 244x244px とされ、本論文では 800x800px との比較も行う。
- **LogisticNet**: AlexNet ベースの shallow architecture。simple model が解けるかを見るために使われ、227x227px で評価される。
- **receptive field**: CNN が入力画像中のどの範囲を利用できるかに関わる概念。本論文では入力解像度の上限、ResNet101 971x971、InceptionV3 1311x1311 の説明に使われる。
- **data augmentation**: overfitting 対策として画像を人工的に増やす処理。本論文では y axis flip、slight random rotation、random brightness modification、random color adjustments が使われる。
- **overfitting**: training dataset に適合しすぎて general features を認識できなくなる状態として説明される。本論文では、300 epochs の予備学習で発生時点を見てから、短い epoch 数で再学習する。
- **validation accuracy**: 300 epochs の予備学習でモデルの挙動を見るために報告される指標。TeX 中では test set accuracy とは明確に区別されていない。
- **Precision / Recall**: Stage1 の usable / unusable 分類に使われる指標。計算式は TeX 中にはないため、表の値として読む。
- **F1-Score / MCC**: Stage2 の safe / unsafe 分類に使われる指標。MCC は Table 2 caption で、1.00 が perfect classification を表すと説明される。
- **confusion matrix**: Fig. 4 で InceptionV3 の class I / II の正誤数を示す表。299x299px と 800x800px の差を見るために使われる。
- **boundary cases**: class I と class II の交差領域として Conclusion で述べられる、同じ画像が安全とも不安全とも判断されうるケース。
- **hybrid dataset**: future work として提案される、computer-generated images と real-world images の混合データセット。

## 読む順番の提案

- まず Abstract と Introduction を読み、11,371 checks、1091 violations、9.6% という動機と、写真ベースの quality management が bottleneck になる理由を押さえる。正規ノートでは Summary の「問題」に対応する。
- 次に Image-Based Load Safety Classification を読む。三クラス定義、multi-class classification ではなく decision tree of two binary classifications にする理由、class III が non-trained human observer にも見分けやすいという根拠を確認する。正規ノートでは Summary の「手法」に対応する。
- Structure of the Dataset で、5712 images、3456x4608 pixels、RGB、I:1813 / II:2355 / III:1544 を確認する。数値はこの節が根拠になる。
- Applied Data Augmentation と Fig. 1 で、overfitting が epoch 50 付近から見えること、augmentation が y-axis flip、rotation、brightness、color adjustment に限定されることを読む。
- Architecture Design と Fig. 2 で、InceptionV3、ResNet101、LogisticNet、入力解像度、receptive field に基づく 800x800 cap の理由を読む。正規ノートでは Notes / Quotes の LogisticNet と解像度関連に対応する。
- Evaluation、Table 1、Result、Table 2、Fig. 4 をまとめて読む。Stage1 は有望、Stage2 は不十分という結論を、Recall / Precision、F1 / MCC、confusion matrix の値で確認する。
- 最後に Conclusion を読み、著者自身の limitation、boundary cases、hybrid dataset、human operator supervision の記述を確認する。正規ノートでは Critical Thoughts と Notes / Quotes に対応する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2306.03795v1/`
- 正規ノート: `notes/arXiv-2306.03795v1.md`
