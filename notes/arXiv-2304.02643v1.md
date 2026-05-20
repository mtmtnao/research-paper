# Segment Anything

- arXiv: https://arxiv.org/abs/2304.02643
- source: ../papers/arXiv-2304.02643v1/
- authors: Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick (Meta AI Research, FAIR)
- venue / year: ICCV 2023 (preprint 2023-04)
- tags: [segmentation, foundation-model, vision, zero-shot, prompt, dataset]
- read_date: 2026-05-13

---

## Summary（著者の主張）

- **問題**: 画像 segmentation には NLP の GPT 系・CLIP に相当する "foundation model" が存在しない。web 上に self-supervised に転用できる大規模マスクデータが無い、汎用的に下流タスクへ転移できる pre-training タスクと、prompt engineering で多様な下流問題を解ける単一モデルの設計が未確立、という三位一体の問題がある。
- **手法**: 三つを同時に提案する。(1) **promptable segmentation task** — 点・bbox・mask・自由テキストといった任意の prompt から「valid な」マスクを返す。曖昧な prompt でも妥当なマスクのうち少なくとも 1 枚を返せれば valid。(2) **SAM** — MAE-pretrained ViT-H/16 を image encoder（1024×1024 入力、64×64×256 embedding、14×14 windowed attention + 4 equally-spaced global attention block）、sparse prompt（点・box・テキスト）は positional encoding + 学習 embedding、dense prompt（mask）は conv で埋め込み、image embedding と要素和。lightweight な 2 層 Transformer mask decoder が prompt token と image embedding の双方向 cross-attention を回し、output token から dynamic linear classifier でマスクを生成。1 つの prompt に対し 3 枚（whole/part/subpart）出力し、loss は min-over-mask、IoU prediction head でランクづけ（曖昧性吸収）。複数 prompt 時は専用の 4 つ目 output token で 1 マスクだけ返す。image encoder は 1 画像 1 回、prompt encoder + mask decoder はブラウザ CPU で ~50ms。Focal + Dice (20:1)、interactive setup を 11 iteration（初期 prompt + 8 反復点 + 2 回マスクのみ refinement）で simulate。AdamW、lr 8e-4、batch 256、90k iter（≈2 epochs SA-1B）、256 GPU。(3) **データエンジン** — model-in-the-loop 3 段階：(a) assisted-manual（SAM 支援で人手アノテ、平均 14 秒/マスク、4.3M masks / 120k images）→ (b) semi-automatic（自動マスクを埋めて annotator は残りを追加、+5.9M masks / 180k images、累計 10.2M masks）→ (c) fully automatic（全画像の 32×32 点グリッド + 2×2/4×4 の overlap zoom-in crops 上の 16×16/8×8 点グリッドで生成、IoU≥88 と stability≥95 で filter、NMS 0.7）、最終 SA-1B = **11M ライセンス済み画像 + 1.1B マスク**（99.1% が完全自動生成）。
- **結果**: 23 個の多様な segmentation データセットで zero-shot 評価。single foreground point → mIoU で 23 のうち **16 でベストの interactive baseline RITM を上回り、差は最大 ~47 IoU**。3 出力中の最良を選ぶ "oracle" 設定では 23 全てで RITM を上回る。Human study（1–10 評価）でも SAM のスコアは 7–9 帯で RITM や ambiguity-unaware ablation を有意に上回る。Edge detection (BSDS500, zero-shot): ODS .768 / OIS .786 / AP .794 / R50 .928（R50 のみ HED の .923 を上回る；ODS/OIS/AP は HED に届かず、SOTA の EDETR (ODS .840 / AP .896) にはさらに届かない。著者は「BSDS500 の bias を学ばないので edge を過剰に出す」と説明）。Object proposals (LVIS v1, AR@1000): SAM 59.3 vs ViTDet-H 63.0。著者は medium/large/rare/common で ViTDet を上回ると主張（Table 値では large は SAM 86.9 vs ViTDet 87.0 とほぼ同等）。Instance segmentation (ViTDet box → SAM): COCO mask AP 46.5 (vs ViTDet 51.0)、LVIS 44.7 (vs 46.6)。AP では負けるが人手評価では SAM のほうが高評価で、ViTDet が COCO/LVIS 固有の annotation bias を学習している示唆。Text-to-mask は CLIP image embedding を訓練時 prompt、推論時に text embedding を流す proof-of-concept。Mask quality: 94% のペアが human-edit との IoU > 90%（inter-annotator consistency 85–91% を上回る）。Ablation: データエンジン 3 段階のうち自動マスクのみで学習しても全データ使用と ~0.5 mIoU 差、SA-1B の 10%（1M 画像）でも全量とほぼ同等、ViT-H は ViT-L 比で gain 飽和。RAI: 知覚 gender / age / skin tone で person segmentation 性能差は概ね CI が overlap（年齢の older vs middle のみ disjoint）、衣服 segmentation では masculine の方が有意に高い bias あり。
- **貢献**: (1) prompt 経由で多様な下流タスクへ転移できる "promptable segmentation" タスクの定式化、(2) ブラウザ実時間動作する ambiguity-aware な SAM 本体、(3) 既存最大の Open Images の **400× のマスク数・11× の画像数** を持つ SA-1B、(4) Apache 2.0 でのモデル公開と研究目的ライセンスでのデータ公開、(5) 23 データセット zero-shot 評価プロトコル一式と RAI 分析。

## Takeaway（自分にとっての要点）

- 「foundation model = 自己教師あり」という Bommasani+ の図式に対し、SAM は **MAE 初期化 + 大規模 supervised** で組む。data engine でラベルがスケールできる領域では supervised の方が有効、という具体例として実用上重要。
- **3 マスク出力 + min-loss + IoU head** だけで曖昧性を吸収しているのが効いている（Single mask 版は human study でも mIoU でも一貫して劣る）。「複数 prompt 時は別の 4 つ目 token に切り替えて 1 マスクだけ返す」という設計まで含めて、ambiguity-aware が本論文の MVP。
- データエンジンの結論として「自動マスクのみで学習しても全データ学習とほぼ同等」「SA-1B の 1/10 でほぼ飽和」が出ているので、再現コミュニティが SA-1B 全量を要するわけではない。**1M 画像 + 100M マスクが実務 sweet spot** として明示されている。
- AP では負けるが human rating では勝つ、という乖離（特に COCO）は重要。COCO/LVIS 系 benchmark の AP は annotation bias を吸えるモデルが有利、という long-standing な不健全さを SAM の "zero-shot だから bias を吸わない" 設定が裏返しに照明している。
- prompt engineering で edge / proposal / instance / text → mask を解いてみせる構成は、後続研究が「SAM をどう composable に組み合わせるか」という流れを生む下敷きになる（実際 MCC や gaze-prompted system が言及）。
- 14×14 windowed attention + 4 global block の ViT-H で 1024² 入力、image encoder は 1 回計算して使い回し、prompt 側は CPU 50ms。**インタラクティブ UX のためのコスト分割が architecture decision の中心**で、これは応用に乗せる時の制約として効く。
- text-to-mask は「image embedding で学習 → 推論時 text embedding」という安価なトリックで動くが、本人たちも exploratory と認めている。

## Critical Thoughts（評価・疑問）

- **強み**:
  - Task / Model / Dataset を三位一体で出した上で、それぞれ単独でも価値がある（SA-1B 公開だけでも segmentation 研究に与えるインパクトは大）。
  - 23 データセット + human study + RAI + 大量の ablation で評価が極めて分厚い。zero-shot の subjective evaluation で自分の弱点（DRAM, IBD で mIoU は劣る）まで human rating で開示しているのは誠実。
  - data engine の 3 段階で「人手 → semi → 全自動」と annotation cost と diversity をトレードオフしていく構成は再現可能で、後続のキュレーション研究の rubric として有用。
  - mask quality を inter-annotator IoU と比較し、自動マスクが人手と同等水準であることを定量化している。
- **弱み / 疑問**:
  - 著者自身が limitations として挙げているもの: (a) fine structure を取りこぼし、小さな disconnected component を hallucinate、boundary が zoom-in 系より粗い、(b) 多点が与えられる interactive setting では専用手法に負ける、(c) image encoder が重く全体は real-time でない、(d) text-to-mask は exploratory で robust でない、(e) semantic / panoptic を prompt で簡潔に書く方法が未解決、(f) ドメイン特化ツールには各々の domain で負ける。
  - SA-1B は「ライセンス画像」のため画像本体の二次配布や商用利用の制約が残り、reproducibility / community auditability に gap がある（geo-information も image provider 要請で release されない）。
  - RAI で「衣服 segmentation は perceived gender presentation で disjoint CI」と認めている bias は、下流で fashion / VTON など実応用に乗せる際に直接効くので軽視できない。
  - text-to-mask の image-embedding → text-embedding tricks は、CLIP の modality gap を考えると distribution mismatch をどこまで吸えているか定量評価がない。
  - mask AP では ViTDet に明確に劣るのに human rating で勝つ、を「ViTDet が bias を学んだから」と説明しているが、では SAM を COCO/LVIS で軽く fine-tune すれば AP も上がるはずで、その実験はしていない。
  - 数値スケーリングは ViT-H で頭打ち（ViT-L とほぼ差なし）、データも 1M で頭打ち、と "scale で殴る foundation model" の物語と内部矛盾がある。SAM はむしろ data curation の研究と読むのが正しい。
  - 計算機資源（256 GPU、2 epoch、batch 256、1024² ViT-H）が要求するコストの記述が control 群との対比で出てこない。
- **次に試したいこと**:
  - SAM を distillation の教師として、軽量 student（モバイル on-device）にどこまで蒸留できるか。image encoder が重いだけで decoder は軽いので image-encoder 蒸留が本筋。
  - "3 マスク（whole / part / subpart）"を semantic 階層と対応づけ、part-aware segmentation や 3D scene parsing への bridge にする。
  - SA-1B の自動マスクで pretrain → 各下流 dataset で短時間 fine-tune した時の mIoU と zero-shot のギャップ計測。「foundation model としての価値」を定量化したい。
  - prompt として gaze / depth / 言語の階層構造を入れた multimodal prompt encoder の拡張、特に panoptic 用 prompt design。
  - 著者が "domain-specific tools には負ける" と認めた医用画像・衛星画像など、SA-1B に該当分布が薄い領域での fine-tune 効率を測る。

## Notes / Quotes

- "Our goal is to build a foundation model for image segmentation." (Introduction)
- "The requirement of a valid output mask means that even when a prompt is ambiguous and could refer to multiple objects ... the output should be a reasonable mask for at least one of those objects." (§Task)
- "We found 3 mask outputs is sufficient to address most common cases (nested masks are often at most three deep: whole, part, and subpart)." (§Model)
- mask decoder と prompt encoder はブラウザ CPU で ~50ms、image encoder は 1 画像 1 回で重い（§Model "Efficiency"）。
- データエンジン 3 stage: 4.3M (120k imgs) → +5.9M = 10.2M (180k imgs) → 全自動で 11M imgs / 1.1B masks。99.1% が fully automatic（§Data Engine, §Dataset）。
- "94% of pairs have greater than 90% IoU (and 97% of pairs have greater than 75% IoU). For comparison, prior work estimates inter-annotator consistency at 85–91% IoU." (§Dataset "Mask quality")
- BSDS500 edge: ODS .768 / OIS .786 / AP .794 / R50 .928（Table 3 = tab:edges, §7.2）。HED は ODS .788 / OIS .808 / AP .840 / R50 .923。
- LVIS proposals: AR@1000 SAM 59.3 vs ViTDet-H 63.0、ablation single-output 54.9（Table 4 = tab:proposals, §7.3）。
- Instance seg: COCO mask AP SAM 46.5 vs ViTDet 51.0、LVIS 44.7 vs 46.6（Table 5 = tab:instance_segmentation, §7.4）。"\sam has higher ratings than ViTDet, suggesting that ViTDet exploits biases in the COCO and LVIS training data." (fig:humanstudy:inst caption)
- Ablation: 自動マスクのみで全データの ~0.5 mIoU 差。1M 画像で全 11M と comparable。ViT-H は ViT-L 比で marginal gain（§7.6, fig:ablations）。
- "Further image encoder scaling does not appear fruitful at this time." (§7.6 Ablations)
- 著者自身の Limitations: 「fine structures を取りこぼす / 小さな disconnected components を hallucinate / boundary が crisp でない / 多点 interactive では専用手法に劣る / image encoder が重く全体は real-time でない / text-to-mask は exploratory / semantic & panoptic 用 prompt の設計法は未解決 / domain-specific tools には負ける」(§Discussion "Limitations")。
- RAI: 知覚 gender / skin tone の person segmentation はすべて 95% CI overlap、知覚 age では older vs middle のみ CI が disjoint（Table 2 = tab:rai_person）。一方 clothing segmentation は perceived gender で disjoint CI（masculine の方が高 mIoU）= bias あり（appendix Table app:tab:rai_clothing）。
- (verified 2026-05-20) Edge 結果の HED/EDETR 比較を訂正 (tab:edges 値で SAM は R50 のみ HED 超え、ODS/OIS/AP は HED 未満)、データエンジン Stage 3 の crop 記述 (2×2/4×4 overlap windows 上の 16×16/8×8 点グリッド) と filter 閾値 (IoU≥88, stability≥95) を §app:dataset_generation から補正、4 つ目 output token の挙動を §app:model から追加、RAI の CI 説明を tab:rai_person 注釈に合わせて訂正、表・節番号を本文の番号 (Table 3/4/5, §7.x) に揃えた。

## Related Papers

- Bommasani+ 2021 "On the Opportunities and Risks of Foundation Models" — foundation model 概念の定義元。
- Radford+ 2021 CLIP — text prompt の参照、text encoder の流用元。
- He+ 2022 MAE — image encoder の self-supervised 初期化。
- Dosovitskiy+ 2021 ViT, Li+ 2022 "Exploring plain ViT backbones for object detection" — image encoder 構造の元。
- Carion+ 2020 DETR, Cheng+ 2021 MaskFormer — mask decoder の token / dynamic head 設計の元。
- Sofiiuk+ 2022 RITM, Liu+ 2022 SimpleClick, Chen+ 2022 FocalClick — interactive segmentation baseline。
- Li+ 2022 ViTDet, He+ 2017 Mask R-CNN, Cai+ 2018 Cascade — instance segmentation baseline。
- Gupta+ 2019 LVIS, Lin+ 2014 COCO, Zhou+ 2019 ADE20K, Open Images — 比較対象データセット。
- Schumann+ 2021 MIAP — RAI 評価用データセット。
- Wu+ 2023 MCC — SAM を component として組む 3D 復元の応用例。
