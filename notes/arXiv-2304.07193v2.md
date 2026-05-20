# DINOv2: Learning Robust Visual Features without Supervision

- arXiv: https://arxiv.org/abs/2304.07193
- source: ../papers/arXiv-2304.07193v2/
- authors: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski (Meta AI Research / Inria)
- venue / year: TMLR 2024（openreview: a68SUt6zFt）
- tags: [self-supervised, vision, ViT, foundation-model, representation-learning]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: NLP では大量の raw text から task-agnostic な foundation model が確立したのに対し、視覚の foundation model は CLIP 系の text-guided pretraining が主流で、caption が画像情報を近似してしまう／image-text aligned corpus が必要、という制約がある。一方で純粋な self-supervised learning (SSL) は主に ImageNet-1k 上での研究に閉じており、scale させようとした既存研究 (SEER 等) は uncurated データを使い feature 品質を落としていた。「curated な大規模データ上で SSL すれば fine-tune 不要の汎用視覚特徴が得られるか？」が本論文の問い。
- **手法**:
  - **モデル/損失**: iBOT をベースに、(a) DINO の image-level cross-entropy（CLS token 同士、teacher は EMA）、(b) iBOT の patch-level masked prediction、(c) DINO/iBOT head の重みを untie（iBOT 原論文と逆の知見）、(d) SwAV 由来の Sinkhorn-Knopp 中心化を teacher 側に 3 iter、(e) KoLeo 正則化（バッチ内特徴を均一に広げる、Kozachenko-Leonenko 微分エントロピー推定子由来）、(f) 最終短期間だけ 518×518 の高解像 phase を追加（FixRes / FlexiViT 系）。
  - **データ (LVD-142M)**: 公開 web crawl から 1.2B unique images を集め、PCA hash で重複除去・NSFW フィルタ・顔ぼかし後、ImageNet-22k/1k・Google Landmarks・fine-grained datasets を query として SSL ViT-H/16 (ImageNet-22k 事前学習) の embedding で cosine 近傍検索 (N=4) と k-means クラスタからの sampling を組み合わせ、142M の curated set を構築。Faiss + 20 ノード×8 V100 で 2 日以内。
  - **学習効率**: 独自 FlashAttention（head dim 64 倍数を意識し ViT-g は 1536 dim × 24 head, 1.1B params に変更）、sequence packing（large/small crop を連結＋block-diagonal mask）、efficient stochastic depth（drop 率 40% で drop した residual の計算自体を skip）、FSDP で 16GB の 4 replica（student/teacher/AdamW の m, v）を GPU 間 shard、weight は float32 保持・broadcast/grad-reduce は float16。これらで iBOT 実装に対し約 2× 高速・メモリ 1/3。
  - **モデル系列**: ViT-g/14 (1.1B) を本体として学習し、ViT-S/14・B/14・L/14 へ knowledge distillation（同じ loop, frozen ViT-g teacher, mask と stochastic depth を切る、iBOT loss を 2 つの global crop に適用、最終 model は student の EMA）。
- **結果**（frozen features + linear / kNN / 簡易 decoder）:
  - **ImageNet-1k linear**: ViT-g/14 で 86.5%（kNN 83.5%）、ReaL 89.6, V2 78.4。OpenCLIP ViT-G/14 86.2 / EVA-CLIP 86.4 を上回り、V2 では EVA-CLIP に対し +1.1%（paper 本文の主張、Table 1 値は 78.4 vs 77.4）。iBOT ViT-L/16 (82.3%) に対し +4.2%。fine-tune すると 88.5 (224) / 88.9 (448) で、SoTA 91.1 に -2.2%。
  - **Robustness**: Im-A 75.9, Im-R 78.8, Im-C 28.2 (↓), Sketch 62.5。iBOT 比で A +29.6%, R +22.1%, Sketch +23.0%（Table 2）。
  - **fine-grained / video**: iNat2018 81.6 (OpenCLIP-G 比 +8.6), iNat2021 85.7 (+9.7), Places205 67.5 (-2.3)。K400 78.4 (+0.1), UCF-101 91.2 (+0.5), SSv2 38.3 (+2.5)。12 タスク SimCLR-bench 平均 92.1（OpenCLIP-G 91.9）。
  - **Instance recognition** (Oxford-Hard mAP): ViT-L/14 で 54.0、SSL より +41, OpenCLIP-G より +34。
  - **Semantic seg (mIoU)**: ADE20k は ViT-g/14 で linear 49.0 / +ms 53.0（ViT-L/14 +ms 53.1 が最高、SoTA InternImage 62.9）、frozen backbone + ViT-Adapter + Mask2former で ADE20k 60.2 mIoU、Pascal VOC ViT-g/14 +ms 86.2（SoTA 89.0）。
  - **Depth (RMSE)**: NYUd DPT 0.279, KITTI DPT 2.11, NYUd→SUN-RGBD zero-shot 0.338。iBOT-L/16 や OpenCLIP-G を大きく上回り、SoTA BinsFormer (0.330) と同等以上。
  - **ablation**: iBOT→DINOv2 で kNN 72.9→82.0, linear 82.3→84.5（Table 1）。KoLeo は Oxford-M で +8（55.6→63.9）、MIM は ADE20k で +2.9 (44.2→47.1)。LVD-142M は INet-22k に対し iNat/Places で勝つが INet-1k は同等。distill した ViT-L は scratch 比 12/12 benchmark で勝つ。
- **貢献**: (1) SSL のみで fine-tune 不要な汎用 frozen feature を、image 級・pixel 級ともに weakly-supervised SoTA (OpenCLIP/EVA-CLIP) と同等または超える水準まで引き上げた最初の系統。 (2) 自動・メタデータ非依存の image-similarity 駆動 curation pipeline (LVD-142M) を提示。 (3) 1B+ ViT を SSL で安定に学習するための実装スタック（独自 FlashAttention, sequence packing, efficient stochastic depth, FSDP mixed precision）を公開。 (4) ViT-g→ViT-S/B/L の distillation、Sinkhorn-Knopp 中心化、head untie、KoLeo、短期高解像 phase の効果を ablation で示す。 (5) Apache 2.0 でモデルとコードを公開、Dollar Street / Casual Conversations での fairness 分析と carbon コストを開示。

## Takeaway（自分にとっての要点）

- "self-supervised で curated data に scale" すれば、CLIP/EVA-CLIP のようなテキスト監督を使わなくても汎用視覚 backbone が作れる、を初めて系統的に示した。captioning に閉じ込められない pixel-level 情報（深度・instance retrieval）で text-guided model に明確に勝つというのは、SSL の存在意義を再確認させる定量結果。
- 強い視覚 backbone を作るレシピは「個別の新規 loss」ではなく **データ curation + 既存 SSL 法 (DINO+iBOT+SwAV+KoLeo) の組合せ + 学習スタックの徹底的なエンジニアリング + 大→小への distillation** であって、要素ごとに見ると派手さは無いが組み合わさると iBOT から kNN +9.1 / linear +2.2 という効きをする (Table 1)。
- LVD-142M は uncurated より良いだけでなく ImageNet-22k より広い domain（iNat, Places）で良い。「curated データを retrieval で疑似的に拡大する」アイデアはテキストの CCNet 系を画像に持ち込んだ形で、再現可能性が高い。
- 大モデルを scratch から作って小モデルは distill、というレシピは ViT-L で scratch 84.5 → distill 86.3（ViT-g scratch 86.5 とほぼ同等、Fig 5b）と、計算予算配分の指針として効く。
- KoLeo は instance retrieval (Oxford-M +8.3) のような近傍検索系で効くが、分類精度は基本据置き。**「特徴を広げる」型の正則化は分類より検索系で意味が出る**という具体例。
- patch-level 特徴の PCA で、第1成分が前景/背景を分け、以降の成分が「翼」「脚」のような object parts に対応する emerging property。ラベルなしで part-level の対応が出ているのは linear probe 越しの semantic segmentation の強さと整合的。
- carbon: ViT-g 1 回 = 22k GPU-h = 3.7 tCO₂eq、プロジェクト全体 0.5–1k tCO₂eq ≒ 200k GPU-day。同条件で OpenCLIP-G を再現すると 118.9 MWh（DINOv2-g は 9.7 MWh）と 10× 差。SSL は text encoder を捨てる前提で carbon 上も合理的、という主張。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 評価範囲が広い（ImageNet 系・iNat・Places・SimCLR 12-bench・UCF/K400/SSv2・Oxford/Paris/Met/AmsterTime・ADE20k/CityScapes/VOC・NYUd/KITTI/SUN-RGBD・robustness 4 種・fairness 2 種）。ここまで包括的に SSL vs WSL を並べた仕事は珍しく、主張の検証可能性が高い。
  - frozen backbone のまま linear/kNN/単純 decoder で評価しており、特徴の "readily available" 性が定量的に担保されている（finetune で +2% しか伸びない、という Sanity check も明示）。
  - ablation が積上げ式（Table 1）で、どの要素がどれだけ効くかを追える。KoLeo / MIM の単独 ablation も別 Table で切ってあり、検索 vs 密予測の効きの違いがクリーン。
  - 学習スタックを Apache 2.0 で公開し、xFormers にも還元しているので、再現と再利用の障壁が低い。
- **弱み / 疑問**:
  - **著者自身が認める limitations**: (i) Dollar Street で Europe vs Africa の差が依然 -25.7%、low vs high income で -31.7% と「Western/高所得バイアス」が残る (fairness §5.1)。 (ii) Possibly-Human ラベル（Beard 等経由）で男性が多く triggered される (§5.2)。 (iii) intro/related で言及される通り、SSL は「scale すれば質が上がる」が uncurated データに対しては feature 品質が落ちる。 (iv) 性能向上の要因として training recipe / scale / data / distillation を併記しており、どれが支配的かは isolate しきれていない（Future work §で因子列挙）。
  - **「curated データ」が ImageNet-22k 等の人手キュレーション済セットを query とする以上、結局 ImageNet 系の class 分布が bottleneck**。retrieval された web 画像も query 近傍に偏るはずで、本当に "general" な分布なのかは TeX 中の議論だけだと不十分（iNat / Places で勝つ、という間接証拠に頼っている）。
  - **OpenCLIP との比較は ImageNet-1k linear で +0.3、V2 で +1.1 と僅差**。一方 SUN (-5.3) / Cars (-4.7) では負ける。「matches or surpasses on most」は事実だが、テキスト教師付き帰納バイアスが効く domain ではまだ負け得るという解像度は読み取っておいたほうがいい。
  - ViT-g/14 の compute は 22k A100-hours と巨大。LVD-142M の構築自体も 20 ノード×8 V100×2 day。「open recipe」とは言っても academic ラボでの追試はかなり厳しい。
  - **distillation で ViT-L が ViT-g とほぼ並ぶ**のは強みである一方、「結局 1B モデルが必要だったのか？」という疑問を惹起する。同じ計算で複数の中規模 SSL を ensemble する方が良かった可能性は議論されていない。
  - SSv2 では iBOT-L (38.7) が ViT-g (38.3) に僅差で勝つ。temporal motion を要する task では本手法が必ずしも勝たない＝静止画 SSL の限界として正直に出ている。
  - test set 漏洩対策として benchmark validation/test との near-duplicate を除去している記述はあるが、ImageNet-1k を query に curation している以上、評価セットとの "概念的" 漏洩は完全には排除できないはず。
- **次に試したいこと**:
  - DINOv2 features を入力 token として LLM に流す（著者自身が future work で言及）。VLM の vision tower を CLIP から差し替えるとどこで勝ち負けが反転するかは興味深い。
  - LVD-142M の query set を ImageNet 系から外し（例: iNaturalist のみ、Open Images のみ）、curation pipeline 自体が "ImageNet 様の分布" にどれだけ依存しているか測る。
  - KoLeo 正則化を CLIP の image-encoder にも入れたとき instance retrieval が改善するか（テキスト埋め込みとの整合は崩れる可能性あり）。
  - distillation を多段（g→L→B→S）にして、各段で精度がどう劣化するかの curve を引く。1 段 distill が最適という仮定は ablate されていない。
  - "frozen + linear" を保ったまま、ADE20k で Mask2former + ViT-Adapter のように decoder 側だけ強化したときの pareto frontier。本論文の 60.2 mIoU と SoTA 62.9 mIoU の差が decoder 由来か backbone 由来かを isolate したい。

## Notes / Quotes

- "self-supervised pretraining alone is a good candidate for learning transferable frozen features that are competitive with the best openly available weakly-supervised models." (intro.tex)
- "Most of the technical contributions aim at accelerating and stabilizing the training at scale." (abstract.tex)
- LVD-142M: 1.2B → dedup → retrieval (N=4 nearest neighbors per query) で 142M。Faiss + 20 nodes × 8 V100, 2 日以内 (data.tex)。
- ViT-g/14 = 1536 dim × 24 heads = 1.1B params。head dim を 64 倍数に揃えるため Zhai+ の (1408,16) から変更 (approach.tex §Efficient impl)。
- Sequence packing: large/small crops の長さ違い token 列を 1 本に concat し、block-diagonal attention mask で分離 (approach.tex)。
- "we observed that the opposite is true, and we therefore use two separate heads in all our experiments." — iBOT 原論文と逆に DINO/iBOT head を untie した方が良い (approach.tex)。
- "training a self-supervised model is preferable in terms of carbon emission. Training a text-guided model still makes sense when planning to reuse the text encoder." (carbon.tex)
- "we still observe a significant difference between regions, particularly in Africa, where our model performance drops by 25.7% compared to Europe." (fairness.tex)
- "an emerging property -- our model was not trained to parse parts of objects." — PCA 第2以降の成分が object parts に対応 (experiments.tex §Qualitative)。
- "the \OURS{} code runs around 2× faster using only 1/3 of the memory" vs iBOT (approach.tex §Efficient impl)。
- 高解像 phase は学習末期に短期間 518×518。"training at 416 is approximately 3× more compute-intensive than training at 224" (ablation.tex §Impact of Resolution)。
- (verified 2026-05-20) V2 vs EVA-CLIP の差を +1.0% → +1.1% に修正（paper 本文の主張、experiments.tex §"How far are we from weakly-supervised models?"）。Table 1 の値 78.4 vs 77.4 = 1.0 と paper claim の 1.1 がズレている旨も明記。
- (verified 2026-05-20) ADE20k linear 49.0 / +ms 53.1 が ViT-g (49.0) と ViT-L (53.1) を混ぜていたため、ViT-g/14 で linear 49.0 / +ms 53.0、ViT-L/14 +ms 53.1 が最高、と分離（experiments.tex Table tab:semseg）。
- (verified 2026-05-20) Distill ablation の数値 "ViT-L で scratch 86.5→distill 86.3" を "ViT-L scratch 84.5 → distill 86.3, ViT-g scratch 86.5" に修正（ablation.tex Fig 5b distillation table）。

## Related Papers

- Caron+ 2021, *Emerging Properties in Self-Supervised Vision Transformers* (DINO) — image-level loss の直接の元。
- Zhou+ 2021, *iBOT* — patch-level masked prediction の base。本論文は iBOT を起点に scaling した位置付け。
- Caron+ 2020, *SwAV* — Sinkhorn-Knopp 中心化の出典。
- He+ 2021, *Masked Autoencoders Are Scalable Vision Learners* (MAE) — finetune 前提の SSL baseline。frozen 比較で大差負け。
- Radford+ 2021, *CLIP* / Ilharco+ 2021, *OpenCLIP* / Fang+ 2022, *EVA-CLIP* — text-guided weakly-supervised の主要比較対象。
- Goyal+ 2022, *SEERv2* — uncurated 大規模 SSL の代表 baseline。
- Sablayrolles+ 2018 — KoLeo 正則化の出典。
- Dao+ 2022, *FlashAttention* — 独自実装の元。
- Touvron+ 2019, *FixRes* / Beyer+ 2023, *FlexiViT* — 末期高解像 adaptation の思想的源流。
- Hinton+ 2015 / Duval+ 2023 — distillation 手法の元。
- Wenzek+ 2019, *CCNet* — テキスト側の curation pipeline、本論文の image retrieval ベース curation のアナロジー元。
