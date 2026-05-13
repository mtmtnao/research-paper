# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- arXiv: https://arxiv.org/abs/2010.11929
- source: ../papers/arXiv-2010.11929v2/
- authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Research, Brain Team)
- venue / year: ICLR 2021
- tags: [vision-transformer, image-classification, transfer-learning, scaling, attention]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: NLP では Transformer が標準なのに、画像認識では CNN（ResNet 系）が依然 SOTA。これまでの「画像 × self-attention」研究は (a) CNN に attention を継ぎ足すか、(b) 局所/疎/軸方向に近似した特殊な attention で畳み込みを置き換える方向であり、(b) は理論効率は良くても modern accelerator に乗りにくい。純粋な Transformer を画像にそのまま当てて競争力を出せるか、というのが問い。
- **手法**: 画像 $\mathbf{x}\in\mathbb{R}^{H\times W\times C}$ を $P\times P$ の固定サイズ patch $N=HW/P^2$ 個に分割し、各 patch を flatten して線形射影 $\mathbf{E}$ で $D$ 次元に embed。BERT 流に学習可能な `[class]` token を先頭に prepend し、学習可能な 1D position embedding を加算して standard Transformer encoder（pre-LayerNorm、MSA + MLP(GELU)、residual）に入れる。`[class]` token の最終層出力を画像表現として MLP/Linear head で分類する。原 Transformer から「画像特化の inductive bias」をほぼ抜く設計（patch 化と fine-tune 時の位置埋め込み 2D 補間だけが 2D 構造への手当て）。Hybrid 版として ResNet feature map 上の patch を入力にする構成も用意。ViT-Base/Large/Huge の 3 サイズ（Layers 12/24/32, Hidden 768/1024/1280, Heads 12/16/16, Params 86M/307M/632M, Table 1）と patch サイズ {16, 32}（H は 14）を組み合わせ、ViT-L/16 のように表記。pre-train は Adam（β1=0.9, β2=0.999, batch 4096, weight decay 0.1）、fine-tune は SGD momentum 0.9, batch 512、しばしば pre-train より高解像度（ImageNet 用 ViT-L/16 で 512、H/14 で 518、Polyak 0.9999 平均）。
- **結果**:
  - Table 2 (Comparison to SOTA): JFT-300M で pre-train した ViT-H/14 が ImageNet 88.55%、ImageNet-ReaL 90.72%、CIFAR-10 99.50%、CIFAR-100 94.55%、Oxford-IIIT Pets 97.56%、Oxford Flowers-102 99.68%、VTAB(19 tasks) 77.63%。BiT-L(ResNet152x4) は 87.54 / 90.54 / 99.37 / 93.51 / 96.62 / 99.63 / 76.29、Noisy Student(EfficientNet-L2) は ImageNet 88.4/88.5*、ReaL 90.55。
  - 計算量: ViT-H/14 は **TPUv3-core-days 2.5k**、ViT-L/16 (JFT) は **0.68k**、ViT-L/16 (ImageNet-21k) は **0.23k** に対し BiT-L 9.9k、Noisy Student 12.3k。同等以上の精度を 1/4〜1/5 程度の compute で達成。
  - VTAB breakdown（Fig 2）: Natural と Structured で BiT-R152x4 / VIVI / S4L を上回り、Specialized は BiT と同等。
  - データ依存性（§4.2）: ImageNet 1k だけで pre-train すると ViT-L < ViT-B、ImageNet-21k で同等、JFT-300M で大きく逆転。JFT のランダム部分集合 9M/30M/90M/300M で linear few-shot を比較すると小データでは ViT が ResNet に負け、大データで上回る。ViT-B/32 は 9M で ResNet50 より大きく劣るが 90M+ で逆転、ResNet152x2 vs ViT-L/16 でも同様。
  - Scaling study（§4.3, Fig 5, Table 6）: ViT は ResNet と同精度を **約 2〜4 倍少ない compute** で出す。hybrid は小モデルでは ViT より少し良いが大モデルでは差が消える。ViT は試した範囲（最大 ViT-H/14 4262 exaFLOPs）で飽和していない。
  - 内部可視化（§4.4, Fig 7）: patch embedding の主成分は patch 内構造の基底関数様。学習された位置埋め込みは行列状に近い patch ほど cos 類似度が高く、大 grid では正弦波様パターン。attention distance は浅層から大きく取れるヘッドと局所的なヘッドが共存し、深層では全ヘッドが広く attend。hybrid（ResNet 前置）では浅層の局所 attention が弱まる。
  - Self-supervision（§4.5, Appx D.3）: BERT 的 masked patch prediction（patch embedding の 50% を [mask]/random/keep の 80/10/10 で破壊、3-bit mean color を予測）で ViT-B/16 を JFT 14 epoch ≒ 1M step で事前学習 → ImageNet 79.9%、from scratch +2% 改善、ただし supervised pre-train より 4% 低い。
  - ObjectNet（Appx D.7）: ViT-H/14 で top-5 82.1%、top-1 61.7%。
  - 位置埋め込み（Appx D.4）: 位置情報なし 0.61382、1D 0.64206、2D 0.64001、Rel 0.64032 と「ある/なし」は大差だが種類は誤差程度。
- **貢献**: (1) 純 Transformer を画像 patch 列に直接当てる極めてシンプルな ViT を提案し、画像特化の inductive bias をほぼ排しても large-scale pre-train で SOTA CNN（BiT-L, Noisy Student）に並ぶか凌駕することを示した。(2) 必要データ規模を ImageNet / ImageNet-21k / JFT-300M の系列で実証し、「inductive bias の欠如は大規模事前学習で補える」ことを定量化。(3) 同 compute 予算下で ViT は ResNet/hybrid に対して優位（2〜4× 効率）。(4) attention distance や position embedding の解析、masked patch prediction の予備実験を含む網羅的アブレーション。

## Takeaway（自分にとっての要点）

- 「データを増やせば inductive bias は要らない」というメッセージが定量的に効いているのは **JFT-300M のような 300M 級** から。ImageNet-1k 単独では Large が Base に負ける（Table 5: ImageNet pretrain で ViT-L/16 76.53 < ViT-B/16 77.91）ので、データ規模を見ずに「ViT は ResNet より強い」と一般化してはダメ。
- 計算コスト面の強みは精度より大きい（同精度を 2〜4× 少ない FLOPs で）。Fig 8（Appx）が示すように ViT は ResNet より per-core batch size が大きく取れる＝メモリ効率も良い。実装の単純さと相まって、production 投入のハードルが低い。
- 位置情報の入れ方は本質ではなく「ある/なし」が支配的（Table 7）。patch 単位だと空間解像度が 14×14 程度なので、どう符号化しても学習可能、という説明は他の patch 系手法を設計する際の指針になる。
- hybrid（ResNet stem + ViT）が小モデルで勝つが大モデルで差が消える、という観察は「畳み込みは optimization の初期化を助けるが ceiling を上げない」と読める。小データ向けには hybrid、大データ向けには pure ViT、という使い分けの根拠。
- `[class]` token vs GAP head は学習率を別にすれば同等（Appx D.2）。実装上は GAP の方が NLP 由来要素を減らせて素直、という選択肢が残る。
- masked patch prediction の 79.9% は contrastive 系（後の MoCo-v3 / DINO / MAE 等）と比べる出発点。論文自身が contrastive を future work と明言している。

## Critical Thoughts（評価・疑問）

- **強み**:
  - **シンプルさが最大の貢献**。「画像を 16×16 patch に切って BERT に入れるだけ」を真面目に scaling すれば動く、という事実を提示し、後続の vision backbone 研究の前提を書き換えた。
  - データ規模の効果を ImageNet → ImageNet-21k → JFT-300M で系列的に示し、「データ × inductive bias」の trade-off を定量化した点が説得力ある（Fig 3, Fig 4）。
  - 計算量比較が TPUv3-core-days 単位で具体的に出ているのが珍しく、SOTA より大幅に安いという主張が検証可能。
  - 内部解析（attention distance、position embedding の構造、Fig 7）が「Transformer は何を学んだか」という直感を補強しており、単なる SOTA 主張に終わらない。
- **弱み / 疑問**:
  - 著者自身が結論で挙げている **未解決事項**: (a) detection / segmentation など他タスクへの適用、(b) self-supervised と supervised の 4% ギャップ、(c) さらなる scaling。
  - JFT-300M は **非公開データセット**で、外部からは ImageNet-21k 結果（ViT-L/16 で ImageNet 85.30%）までしか再現できない。論文の「効率の良さ」主張の核心が非公開データ依存なのは再現性上の最大の弱点。
  - 小データ regime での弱さは「強い正則化＋小モデル」で部分的に救えるとされるが（§4.2）、Table 5 を見ると ImageNet pretrain 時の ViT は依然 BiT に劣る。実務で 1k〜100k クラスの専用データだけしかない場合、ViT を選ぶ動機が弱い。
  - VTAB の Specialized で BiT と同等止まりなのは「medical/satellite ではまだ畳み込みが拮抗」と読める。ドメインによって inductive bias 不要論の強度が違う。
  - self-supervision は masked patch prediction のみで、しかも JFT で 1M step も回して +2% から -4% gap という結果。contrastive を試していない（自ら future work と明言）。
  - Hyperparameter sweep の規模が不明瞭。fine-tune 時 ImageNet で {0.003, 0.01, 0.03, 0.06} の grid を切るが、ViT 側に有利に効いた可能性は否定しきれない。
  - 「inductive bias の欠如を大データで補う」は環境負荷・コストの問題と表裏。研究コミュニティ全体としては良いが、現場では replicable でない。
- **次に試したいこと**:
  - ImageNet-21k pre-train + 小データ fine-tune の curve を、同 compute 予算の BiT / hybrid と並べた pareto curve として描き直す（公開可能データ範囲での真の優位を確認）。
  - position embedding の「ある/なし」が支配的という結果を踏まえ、patch 数を可変にした curriculum（小 patch 数で start → 増やす）が学習効率を上げるか。
  - masked patch prediction の延長で、ターゲットを 3-bit color ではなく continuous pixel L2 / DINO 型の features に置き換えた self-sup ablation（後の MAE / iBOT の起点）。
  - attention distance による「層別 receptive field」を pruning 指針として使い、浅層の局所ヘッドを畳み込みに置換しても精度が保てるか（hybrid の小データ優位を解析的に再現できるか）。
  - ViT-Huge より一段大きい設定を、データ量を J<FT-300M 以上に増やしながら回し、Fig 5 の右端で本当に飽和しないかを確認（後の ViT-22B 等への布石）。
  - 異なるパッチ化（重複あり / multi-scale）を試して、計算量とのトレードオフを Table 6 に追記する。

## Notes / Quotes

- "We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks." (abstract)
- "Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data." (§1)
- "We find that large scale training trumps inductive bias." (§1) — 本論文を一言で要約する文。
- Eq.1 patch embedding: $\mathbf{z}_0 = [\mathbf{x}_\text{class}; \mathbf{x}_p^1 \mathbf{E}; \cdots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos}$, $\mathbf{E}\in\mathbb{R}^{(P^2 C)\times D}$, $\mathbf{E}_{pos}\in\mathbb{R}^{(N+1)\times D}$.
- ViT-{Base, Large, Huge} = {12, 24, 32} 層 / Hidden {768, 1024, 1280} / Heads {12, 16, 16} / Params {86M, 307M, 632M}（Table 1）。
- ViT-H/14 (JFT) 88.55% on ImageNet vs Noisy Student 88.4/88.5%、BiT-L 87.54%; TPUv3-core-days 2.5k vs 12.3k vs 9.9k（Table 2）。
- "When pre-trained on the smallest dataset, ImageNet, ViT-Large models underperform compared to ViT-Base models, despite (moderate) regularization." (§4.2) — スケール則の警鐘。
- "ViT uses approximately 2-4× less compute to attain the same performance (average over 5 datasets)." (§4.3)
- Self-sup: "our smaller ViT-B/16 model achieves 79.9% accuracy on ImageNet, a significant improvement of 2% to training from scratch, but still 4% behind supervised pre-training." (§4.5)
- Position embedding ablation: 位置情報なし 0.61382 → 1D 0.64206 / 2D 0.64001 / Rel 0.64032 で「ある/なし」だけが大きい（Table 7, Appx D.4）。
- "ViT models have a clear advantage in terms of memory-efficiency over ResNet models." (Appx D.6)
- ObjectNet (Appx D.7): ViT-H/14 top-5 82.1%, top-1 61.7%。

## Related Papers

- Vaswani+ 2017, *Attention Is All You Need* — Transformer 本体。ViT はこの encoder をほぼそのまま流用。
- Devlin+ 2019, BERT — `[class]` token、masked prediction、pre-train→fine-tune の枠組みの直接の祖先。
- Cordonnier+ 2020 (SA-CNN) — 2×2 patch + full self-attention の先行研究。ViT は patch を大きくして高解像度へ拡張、かつ大規模 pre-train を入れた点が違い。
- Kolesnikov+ 2020, Big Transfer (BiT) — JFT-300M pre-train + ResNet152x4 の SOTA baseline、§4 全体の主たる比較対象。fine-tune protocol も流用。
- Xie+ 2020, Noisy Student (EfficientNet-L2) — ImageNet SOTA baseline。
- Chen+ 2020, image-GPT (iGPT) — 画素列に Transformer を生成的に当てる先行研究。ViT は patch 単位 + 教師ありで段違いに効率化。
- Carion+ 2020, DETR — detection への self-attention 適用。Conclusion で次の応用先として言及。
- Ramachandran+ 2019 (SASA), Wang+ 2020 (Axial-DeepLab) — 局所/軸方向 attention で畳み込みを置換する系。ViT が「特殊 attention 不要」と主張する対象。
- Sun+ 2017 — JFT データセットと CNN のスケーリング則の前例。
- Touvron+ 2019 — high-resolution fine-tune の根拠。
