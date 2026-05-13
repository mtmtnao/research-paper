# Deep Residual Learning for Image Recognition

- arXiv: https://arxiv.org/abs/1512.03385
- source: ../papers/arXiv-1512.03385v1/
- authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- venue / year: arXiv 2015-12 / CVPR 2016 (ILSVRC & COCO 2015 winner entry)
- tags: [CNN, image-classification, residual-learning, ILSVRC, deep-network, optimization]
- read_date: 2026-05-12

---

## Summary（著者の主張）

- **問題**: 深いネットが層を積めば積むほど精度が上がるという仮説に対し、実際は逆。BN や He init で gradient vanish/explode は概ね解決されたにもかかわらず、plain な VGG 風ネット（18 vs 34 層など）で **training error が深くしたほうが高くなる** "degradation problem" が観測される（Fig.~\ref{fig:teaser}、CIFAR-10 / ImageNet 両方）。「浅い解 + identity 層」という構成上の解が存在するのに、SGD ソルバはそれを見つけられない。これは overfitting ではない（training error 側で起こる）。
- **手法**: 数層ブロックに目標写像 $\mathcal{H}(x)$ を直接フィットさせるのではなく、残差 $\mathcal{F}(x):=\mathcal{H}(x)-x$ をフィットして $y=\mathcal{F}(x)+x$ を出力する "residual block" を導入。$\mathcal{F}+x$ の加算は **parameter-free な identity shortcut connection**（次元が合うとき）で実現し、次元増加時のみ (A) zero-padding か (B) $1\times1$ 投影 $W_s x$ を使う。ImageNet 用には VGG 風の 3x3 中心の plain ネットに shortcut を挿入して 18/34 層、さらに $1{\times}1{-}3{\times}3{-}1{\times}1$ の **bottleneck block** で 50/101/152 層を構成（Table~\ref{tab:arch}）。CIFAR では $6n+2$ 層の細身アーキテクチャを 20/32/44/56/110/1202 層まで。学習は SGD、BN、weight decay 1e-4、momentum 0.9、dropout 不使用。
- **結果**:
  - ImageNet val top-1 (10-crop): plain-34 28.54 → ResNet-34 (option A) 25.03、ResNet-152 21.43（top-5 5.71）。
  - Single-model val (Table~\ref{tab:single}): ResNet-152 top-1 19.38 / top-5 4.49（既存単一モデルでは BN-inception の 5.81、PReLU-net の 5.71 を上回り、過去のアンサンブル結果すら超える）。
  - 6 モデル（うち 152 層 2 個）のアンサンブルで ImageNet test top-5 **3.57%**（ILSVRC 2015 分類タスク 1 位、Table~\ref{tab:ensemble}）。
  - shortcut option (A) parameter-free vs (B) 次元増加のみ projection vs (C) 全部 projection の比較で A<B<C だが差は小さく、(C) の改善はパラメータ増のため、と帰属。以後は B を採用しメモリ節約。
  - CIFAR-10: plain は深いほど error 増、ResNet は単調改善。ResNet-110 で best **6.43%**（5 試行で 6.61±0.16）。ResNet-1202 層は training error <0.1% で最適化問題は無いが test 7.93% と 110 層に劣る（19.4M パラメータが小データセットに対し過大で overfitting と著者は解釈）。
  - 層応答の std を比較（Fig.~\ref{fig:std}）すると ResNet のほうが小さく、深くするほどさらに小さい → 残差は実際に identity 近傍で小さくなる、という motivation を裏付け。
  - 検出: Faster R-CNN のバックボーンを VGG-16→ResNet-101 に置換しただけで、PASCAL VOC07 73.2→76.4 / VOC12 70.4→73.8 mAP、COCO mAP@[.5,.95] 21.2→27.2（+6.0、**相対 28% 改善**）。box refinement + context + multi-scale testing + 3 ネットアンサンブルで COCO test-dev 37.4 mAP@[.5,.95]、ImageNet DET test 62.1 mAP（2 位を 8.5 ポイント絶対差で凌駕）、ImageNet localization test top-5 9.0%（VGG/GoogLeNet 比 64% 相対誤差削減）。
- **貢献**: (1) plain net で起きる degradation 現象の明示と特性化、(2) identity shortcut という最小限の構造変更で 100 層超を最適化可能にする residual learning の提案、(3) bottleneck design による計算量効率（152 層 ResNet 11.3GFLOPs < VGG-19 19.6GFLOPs）、(4) 分類・検出・位置推定・セグメンテーションの 5 タスク同時 1 位という generalization の実証、(5) CIFAR-10 で 1202 層という当時前代未聞の規模を実際に最適化して見せたこと。

## Takeaway（自分にとっての要点）

- 「深いネットが学習できない」のは **vanishing gradient ではなく optimization の難しさ**、というのが BN 後の正しい問題設定。BN を入れても plain は劣化するので、これは別軸の現象として切り離して扱うべき。
- 解決策はアーキテクチャ側で「**identity を fallback として確保する**」こと。これは新たな帰納バイアス: もし上層が無意味ならゼロにすればよく、ソルバが恒等写像を非線形層の合成で再現する必要がなくなる（preconditioning として機能、§3.1）。今後ネットを深くする時の標準パターン。
- shortcut が parameter-free（option A）でも degradation はちゃんと解決される（plain-34 28.54 → ResNet-34 A 25.03）。projection は次元合わせのときだけで十分、というのは「効くのは projection そのものではなく恒等成分の保持」だという強い証拠。bottleneck では projection 全部にすると complexity 倍増するので尚更 identity が要る。
- 152 層単一モデルが過去のアンサンブルより強い（4.49% top-5）という事実は、ensemble より depth 増のほうがコスト効率がよい場面があることを示している。
- 検出側で「**VGG-16 を ResNet-101 に差し替えるだけで COCO +6 mAP**」というのは features の質がそのまま下流タスクに転送される教科書的サンプル。事前学習バックボーンを上げる投資効率の根拠としていまだに引かれる理由がわかる。
- 1202 層が training error <0.1% で安定して学習できるという事実は応用より「最適化障害は本当に消えた」というメッセージとして強い。逆に test で劣化するのは regularization 設計の話に変わる、と問題が切り分けられる点が綺麗。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 提案が極めてシンプル（足し算 1 つ追加）で、現有の SGD / Caffe をそのまま使える。だから世界中で即再現された。
  - 主張に対し対照実験が徹底している: plain vs ResNet を同じ depth/width/params で比較（option A）、shortcut の (A)(B)(C) アブレーション、深さスイープ（18/34/50/101/152、CIFAR 20/32/44/56/110/1202）、応答 std の可視化、複数タスクへの転移。
  - 「3.57% top-5 で 1 位」だけでなく、検出・位置推定・セグメンテーション全部で 1 位という generalization が示されていて、結果が分類タスク特有のチューニング芸ではないことが裏付けられている。
  - 著者自身、何でも肯定するのではなく「(C) の僅差は parameter 増のせい」「projection は本質ではない」「1202 層が 110 層に test で負けるのは overfitting」と慎重に解釈している。
- **弱み / 疑問**:
  - **plain net が最適化困難な根本原因が説明されていない**（§4.1 で「vanishing gradient ではない、conjecture: exponentially low convergence rate、Future Work」と本文中で明言）。residual の有効性は実証だが、なぜ identity を入れると loss landscape がそんなに変わるかは後続研究（Li+ 2018, Veit+ 2016 など）に委ねられている。
  - 1202 層の overfitting は **maxout/dropout を入れていない** 状況での結論（§4.2）。「dropout を併用すれば改善するかも、future work」と自分で言っており、本当に depth が無駄かどうかは結論保留。
  - shortcut option の比較が ResNet-34 のみ。bottleneck（50+ 層）で (A) を試していないため、深層での identity-only の挙動はこの論文では未検証（後の "Identity Mappings in Deep Residual Networks" (He+ 2016) で扱われるが、本論文では未決）。
  - 検出側の improvement（box refinement / context / multi-scale）はバックボーン交換ゲインとは別の汎用技で、各 ablation の純粋効果はやや混ざる。「multi-scale training は時間がなくて未実施」と著者も限界を明記（§Appendix MS COCO の Multi-scale testing 節）。
  - CIFAR-10 の評価は test set 直接（ResNet-110 のみ 5 回平均 6.61±0.16、他のサイズは単発）。1 試行の数値ぶれは ImageNet ほど厳しく検証されていない。
  - ImageNet test set の本番提出は何回か（規定の提出回数）残されていないので、3.57% の信頼区間は不明。
  - "residual is closer to zero than non-residual" の主張は std 比較（Fig.~\ref{fig:std}）のみで、定量的に「ほぼ identity」と言えるほどの zero 近接性は示されていない。
- **次に試したいこと**:
  - degradation の原因仮説（loss landscape のシャープネス、Jacobian の条件数、ensemble of shallow paths 説 など）を実験的に切り分け。具体的には ResNet と plain で Hessian の固有値分布や gradient confusion 指標を比較。
  - identity 以外の固定 shortcut（例えば学習しない random projection や PCA 投影）でも degradation が消えるかを試して、「identity の何が効いているか」を分離。
  - 1202 層に強い regularization（CutMix / stochastic depth / dropout）を入れて、test error が 110 層を超えるかを再評価。stochastic depth はまさに後発で実装され、効くことが分かっている。
  - bottleneck の $1{\times}1$ 圧縮率（256→64 など）と深さのトレードオフを Pareto curve として測り、現在の ConvNeXt / RegNet と並べて compute-accuracy 効率を比較。

## Notes / Quotes

- "Deeper neural networks are more difficult to train." (Abstract)
- "Unexpectedly, such degradation is *not caused by overfitting*, and adding more layers to a suitably deep model leads to *higher training error*." (§1 Introduction)
- "if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers." (§1)
- "We argue that this optimization difficulty is *unlikely* to be caused by vanishing gradients. These plain networks are trained with BN ..." (§4.1 Plain Networks)
- "We conjecture that the deep plain nets may have exponentially low convergence rates ... The reason for such optimization difficulties will be studied in the future." (§4.1)
- "small differences among A/B/C indicate that projection shortcuts are not essential for addressing the degradation problem." (§4.1 Identity vs Projection Shortcuts)
- "Our method shows *no optimization difficulty*, and this $10^3$-layer network is able to achieve *training error* $<$0.1%." (§4.2 Exploring Over 1000 layers)
- "this is because of overfitting. The 1202-layer network may be unnecessarily large (19.4M) for this small dataset." (§4.2、著者自身による limitation 表明)
- 単一モデル ResNet-152: top-1 19.38 / top-5 4.49（Table~\ref{tab:single}）
- 6 モデルアンサンブル: ImageNet test top-5 **3.57%**（Table~\ref{tab:ensemble}）
- Faster R-CNN のバックボーン VGG-16→ResNet-101 で COCO mAP@[.5,.95] 21.2→27.2（+6.0、相対 28%、§4.3 / Appendix）
- ImageNet localization test top-5 9.0%（VGG/GoogLeNet 比 64% 相対削減、Table~\ref{tab:localization_all}）

## Related Papers

- Simonyan & Zisserman 2015, *VGG* — 比較対象 baseline、3x3 conv 哲学のベース。
- Szegedy+ 2015, *GoogLeNet / Inception* — 同時期の "very deep" の代表、auxiliary classifier 経由の shortcut が部分的な前駆。
- Srivastava+ 2015, *Highway Networks* — 同時期に gated shortcut を提案。本論文は parameter-free identity であることと、100+ 層でも精度向上を示した点で差別化。
- Ioffe & Szegedy 2015, *Batch Normalization* — vanishing gradient を実用上潰し、本論文が "degradation は別問題" と言うための前提。
- He+ 2015, *PReLU / He init* — 同一研究グループの直前の SOTA、本論文の初期化に採用。
- Bengio+ 1994 / Glorot & Bengio 2010 — vanishing/exploding gradient の古典。
- Krizhevsky+ 2012, *AlexNet*; Russakovsky+ 2014, *ImageNet* — タスク・データの基礎。
- Ren+ 2015, *Faster R-CNN*; Girshick 2015, *Fast R-CNN*; Girshick+ 2014, *R-CNN* — 検出側の評価フレーム。
- Lin+ 2014, *MS COCO*; Everingham+ 2010, *PASCAL VOC* — 検出データセット。
- Krizhevsky 2009, *CIFAR-10*; Goodfellow+ 2013 *Maxout*; Lin+ 2013 *NIN*; Lee+ 2014 *DSN*; Romero+ 2015 *FitNet* — CIFAR の比較対象。
