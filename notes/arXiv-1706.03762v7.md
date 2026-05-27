# Attention Is All You Need

- arXiv: https://arxiv.org/abs/1706.03762
- source: ../papers/arXiv-1706.03762v7/
- authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- venue / year: NIPS 2017 (Google Brain / Google Research / University of Toronto)
- tags: [transformer, attention, sequence-transduction, machine-translation, NMT]
- read_date: 2026-05-12
- rating:

---

## Summary（著者の主張）

- **問題**: 既存の最強系列変換モデル（RNN/LSTM/GRU ベースの encoder-decoder, あるいは ConvS2S/ByteNet/Extended Neural GPU のような CNN 系）は、(1) RNN は位置 $t$ の隠れ状態 $h_t$ が $h_{t-1}$ に依存するため学習時にも系列方向の並列化ができず長系列で致命的、(2) CNN ベースは並列化はできるが、離れた 2 位置間の依存を結ぶのに ConvS2S なら線形、ByteNet で対数の演算が必要で長距離依存が学びにくい。attention 自体は seq2seq に組み込まれていたが「RNN の補助」としてしか使われていなかった（例外は decomposable attention だけ）。
- **手法**: recurrence も convolution も使わず、self-attention と position-wise FFN だけで encoder-decoder を構築する **Transformer** を提案。要素技術：
  - **Scaled Dot-Product Attention**: $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T/\sqrt{d_k})V$。$\sqrt{d_k}$ で割るのは、$q,k$ の各成分を平均 0・分散 1 とすると内積の分散が $d_k$ になり、$d_k$ が大きいと softmax が飽和して勾配が消えるため。
  - **Multi-Head Attention**: $Q,K,V$ を $h$ 通りに線形射影して並列に attention を計算し、結果を concat して再射影。$h=8$, $d_k=d_v=d_{\text{model}}/h=64$。総計算量は単一ヘッドと同程度。
  - **3 種類の attention 用途**: (1) encoder-decoder attention（decoder→encoder 出力）、(2) encoder self-attention（全位置→全位置）、(3) decoder masked self-attention（subsequent 位置を $-\infty$ でマスクして自己回帰性を保つ）。
  - encoder/decoder ともに $N=6$ 層の同型ブロック。各 sub-layer は `LayerNorm(x + Sublayer(x))` の residual + LN 構成。decoder には encoder-decoder attention の sub-layer が追加されて計 3 つ。
  - **Position-wise FFN**: $\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$、$d_{\text{model}}=512$, $d_{ff}=2048$。
  - **Positional Encoding**: 学習なしの正弦/余弦 $PE_{(pos,2i)}=\sin(pos/10000^{2i/d_{\text{model}}})$、$PE_{(pos,2i+1)}=\cos(\cdot)$。任意のオフセット $k$ に対し $PE_{pos+k}$ を $PE_{pos}$ の線形関数で表せるので相対位置学習に有利と仮説。学習済み positional embedding と「ほぼ同性能」（Table `tab:variations` 行 E）だが、訓練時より長い系列への外挿を期待して sinusoidal を採用。
  - 入出力 embedding と pre-softmax 線形層は重み共有、embedding は $\sqrt{d_{\text{model}}}$ 倍する。
- **結果**: WMT 2014 EN-DE で **Transformer (big) 28.4 BLEU**（既存 best ensemble +2.0 BLEU 超）、base でも **27.3 BLEU** で過去の全 single/ensemble モデルを上回り、訓練 FLOPs は $3.3\times10^{18}$ と ConvS2S ($9.6\times10^{18}$) の約 1/3。WMT 2014 EN-FR は **big 41.8 BLEU** で新 single-model SOTA、訓練は 8 GPU × 3.5 日（base は 12 時間）。汎化検証として WSJ 英語句構造解析で **WSJ-only 4 層 Transformer が F1=91.3**（RNNG 91.7 に次ぐ）、semi-supervised で **F1=92.7** と多くの先行を上回る（task-specific tuning ほぼなし）。Self-Attention 層は計算量 $O(n^2 d)$ で並列演算は $O(1)$, 最大経路長 $O(1)$、recurrent の $O(n)$、conv の $O(\log_k n)$ と比較して長距離依存に有利（Table `tab:op_complexities`）。
- **貢献**: (1) RNN/CNN なしで self-attention のみで構築された初の系列変換モデル、(2) WMT'14 EN-DE/EN-FR で訓練コスト激減しつつ SOTA、(3) multi-head + scaled dot-product + sinusoidal PE のレシピを提示、(4) head 数・$d_k$・層数・モデル幅・dropout・label smoothing・PE のアブレーション（Table `tab:variations`）、(5) 翻訳以外（英語句構造解析）への直接転用で汎用性を示した。

## Takeaway（自分にとっての要点）

- **「並列化できる」ことが主要な動機**。Introduction は recurrent model の sequential nature が training examples 内の並列化を妨げることを問題にし、Transformer は "significantly more parallelization" を可能にすると述べる。
- $1/\sqrt{d_k}$ スケーリングは「dot-product の分散が $d_k$ になる」という統計的根拠から来ている。$d_k$ が小さいうちは additive とほぼ同性能だが、大きくなると additive に負けるのを補正するための実用的な設計。Table `tab:variations` 行 (B) で $d_k$ を 16/32 に削ると BLEU が落ちることから、互換性関数の表現力が効くという指摘もしている。
- **multi-head は「representation subspaces で異なる位置に同時に注意する」ための工夫**。head 数 1 や 32 は base の 8 より悪い（Table `tab:variations` 行 A: 24.9 / 25.4 vs 25.8）。
- positional encoding は **sinusoidal と learned でほぼ同じ**（Table `tab:variations` 行 E: 4.92 PPL, 25.7 BLEU; base は 4.92 PPL, 25.8 BLEU）。著者は訓練時より長い系列への外挿可能性を理由に sinusoidal を選んでいるが、この外挿は実験では示されていない。
- decoder の masked self-attention は softmax の入力で illegal な位置を $-\infty$ にする実装、という具体記述。実装時はここを間違えるとリークする。
- 訓練のレシピ：Adam ($\beta_2=0.98$ がやや特殊), warmup 4000 steps の inverse-sqrt LR schedule, $P_{drop}=0.1$（big は 0.3, EN-FR big は 0.1）, label smoothing $\epsilon_{ls}=0.1$。big は 300K steps / 3.5 day on 8×P100。
- 推論：base はラスト 5 ckpt 平均、big は 20 ckpt 平均、beam size 4・length penalty $\alpha=0.6$。出力長上限は input + 50。parsing では beam 21・$\alpha=0.3$・出力長 input+300。
- self-attention は計算量が $O(n^2 d)$ なので、著者は self-attention が recurrent より速い条件を「系列長 $n$ が表現次元 $d$ より小さい場合」と書いている。長系列向けには "restricted self-attention"（近傍 $r$ のみ、経路長は $O(n/r)$）を将来課題として明示。
- WSJ 約 40K 文の小データ設定で、Transformer は他タスク向けチューニングをほぼせずに F1=91.3 に届く。著者は RNN sequence-to-sequence model が small-data regimes で state-of-the-art に届いていないことと対比している。

## Critical Thoughts（評価・疑問）

- **強み**:
  - 「並列化」「経路長」「層あたり計算量」という 3 軸での recurrent / conv / self-attention 比較表（Table `tab:op_complexities`）が明確。
  - SOTA 取得と訓練コスト削減の両立。EN-DE で best ensemble を ensemble なしで超え、FLOPs は ConvS2S の約 1/3（base $3.3\times10^{18}$ vs ConvS2S $9.6\times10^{18}$）。
  - アブレーション（Table `tab:variations`）が広く、head 数 / $d_k$ / 層数 $N$ / $d_{\text{model}}$ / $d_{ff}$ / dropout / label smoothing / PE の各軸を動かしていて読みやすい。
  - 翻訳から離れた句構造解析でも task-specific tuning をほぼせずに F1=91.3 を出していて、汎用性の主張が薄っぺらくない。
- **弱み / 疑問**:
  - 著者自身が認める限界として **self-attention の計算量が $O(n^2 d)$** であり、長文では recurrent を逆転される。"restricted self-attention" は将来課題として残されたまま（why_self_attention.tex 末尾 + Conclusion で "investigate local, restricted attention mechanisms" と明記）。
  - **2 つの翻訳タスクと 1 つの parsing しか評価していない**。汎用アーキテクチャを主張するわりに NLP の他タスク（要約、QA、言語モデリング）での比較は無い。
  - **学習の安定性に対する考察が薄い**。Adam $\beta_2=0.98$ や warmup 4000、$\sqrt{d_{\text{model}}}$ 倍 embedding など実用上のクセが多いが、なぜ必要かの ablation はない（特に warmup なし条件は試されていない）。
  - **multi-head の「different representation subspaces」という説明は qualitative**。Table `tab:variations` 行 (A) は BLEU 差 1 以内で、head の役割分担を裏付ける定量分析は appendix の attention 可視化（3 図）に依拠していて、人間解釈に頼っている。
  - **Inference 時間の議論がない**。訓練は並列化で速いが、autoregressive decoding 自体は依然系列処理。big モデルの inference latency は触れられていない。
  - **scaling law がない**。$N$ や $d_{\text{model}}$ を上げると精度が上がる（行 C: $N=8$ で 25.5、$d=1024$ で 26.0）ことは示されるが、どこで頭打ちかは未調査。
  - parsing で WSJ-only 91.3 は RNNG 91.7 に負けており、「surprisingly well」とは書いてあるが SOTA ではない点はやや控えめに記述されている。
- **次に試したいこと**:
  - 同じ $3.3\times10^{18}$ FLOPs 予算で **層深さ ($N$)・幅 ($d_{\text{model}}$, $d_{ff}$)・head 数 ($h$) の Pareto curve** を引く。Table `tab:variations` は単独軸の摂動が中心なので最適配分が見えない。
  - **sinusoidal PE が「訓練系列長より長い系列に外挿できる」**主張の実証（本論文では仮説のみ）。学習済み PE と外挿性能を比較する実験。
  - **restricted self-attention（近傍 $r$）の効果**を翻訳で確認。$r=16,32,64$ で BLEU と計算量のトレードオフ。
  - **attention head の役割分析を定量化**。例えば head ごとに依存タイプ（anaphora, syntactic, long-range）を分類して、prune 耐性を測る。
  - **decoder の autoregressive 性を非系列化**（Conclusion で著者も "Making generation less sequential" と書いている）。non-autoregressive Transformer 系の起点として読むべき項目。

## Notes / Quotes

- "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely." (abstract, ms.tex)
- "In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention" (background.tex)
- "To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean 0 and variance 1. Then their dot product ... has mean 0 and variance $d_k$." (model_architecture.tex, scaled dot-product attention の脚注)
- "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this." (model_architecture.tex §3.2.2)
- 重要数値（TeX から直接）：
  - WMT'14 EN-DE: Transformer base 27.3 / big **28.4** BLEU、FLOPs base $3.3\times10^{18}$ / big $2.3\times10^{19}$（results.tex, Table `tab:wmt-results`）
  - WMT'14 EN-FR: base 38.1 / big **41.8** BLEU（ms.tex abstract; results.tex, Table `tab:wmt-results`。results.tex 本文には 41.0 とも書かれており TeX 内で不一致）
  - base 構成: $N=6, d_{\text{model}}=512, d_{ff}=2048, h=8, d_k=d_v=64, P_{drop}=0.1, \epsilon_{ls}=0.1$、100K steps、65M params（Table `tab:variations`）
  - big 構成: $N=6, d_{\text{model}}=1024, d_{ff}=4096, h=16, P_{drop}=0.3$、300K steps、213M params、PPL 4.33 / BLEU 26.4 (dev)
  - hardware: 8×P100、base step 0.4s, big step 1.0s, 訓練 12h / 3.5d
  - データ: EN-DE 約 4.5M 文対・BPE 共有語彙 37K、EN-FR 36M 文・word-piece 32K
  - 推論: beam 4・$\alpha=0.6$、base 5 ckpt 平均 / big 20 ckpt 平均
  - parsing: WSJ Section 23 F1 = 91.3 (WSJ-only) / 92.7 (semi-sup)、4 層 $d_{\text{model}}=1024$
- 著者が明示している limitation: "self-attention could be restricted to considering only a neighborhood of size $r$ ... We plan to investigate this approach further in future work." (why_self_attention.tex) と Conclusion の "local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video" "Making generation less sequential is another research goal."
- 可視化（visualizations.tex）: layer 5/6 で long-distance dependency（"making ... more difficult"）、anaphora resolution（'its'）、構文構造を捉える head の存在を例示。あくまで qualitative。
- (verified 2026-05-20) FLOPs 比の記述「ConvS2S の 1/3 以下 / 1/3 弱」を「約 1/3」に修正 (results.tex Table `tab:wmt-results`: Transformer base $3.3\times10^{18}$ vs ConvS2S $9.6\times10^{18}$、比は約 0.344 で 1/3 より僅かに大きい)。
- (verified 2026-05-26) 図表番号の取り違えを避けるため `tab:op_complexities` / `tab:wmt-results` / `tab:variations` の TeX ラベル参照に修正し、TeX 外の後続研究・帰納バイアスに関する記述を削除または TeX 根拠に限定 (why_self_attention.tex, results.tex, model_architecture.tex)。
- (verified 2026-05-26) WMT'14 EN-FR big BLEU は abstract と Table `tab:wmt-results` の 41.8 を採用し、results.tex 本文には 41.0 とも書かれている TeX 内不一致を明記 (ms.tex, results.tex)。

## Related Papers

- Bahdanau+ 2014 (NMT with attention) — additive attention の原点、本論文の比較対象。
- Sutskever+ 2014 (seq2seq), Cho+ 2014 (RNN encoder-decoder) — Transformer が置き換えにいった枠組み。
- Gehring+ 2017 ConvS2S, Kalchbrenner+ 2017 ByteNet, Kaiser+ Extended Neural GPU — CNN 系の並列 seq2seq、計算量・経路長比較の主な相手。
- Wu+ 2016 GNMT — word-piece 語彙、length penalty $\alpha$、beam search の手法源。Table `tab:wmt-results` baseline。
- Shazeer+ 2017 MoE (Outrageously Large NNs) — 条件付き計算 baseline。
- Cheng+ 2016, Parikh+ 2016 (Decomposable Attention), Paulus+ 2017, Lin+ 2017 — self-attention の先行応用。
- He+ 2016 ResNet, Ba+ 2016 LayerNorm — sub-layer の residual + LN 構成の出典。
- Press & Wolf 2016 — 入出力 embedding と pre-softmax の重み共有。
- Sennrich+ 2015 BPE, Britz+ 2017 — トークナイザと NMT アーキ大規模探索。
- Kingma & Ba 2015 Adam, Srivastava+ 2014 Dropout, Szegedy+ 2015 (label smoothing) — 訓練レシピの構成要素。
- Vinyals & Kaiser+ 2015 (Grammar as a Foreign Language), Dyer+ 2016 RNNG, Petrov+ 2006, Zhu+ 2013, McClosky+ 2006, Huang & Harper 2009, Luong+ 2015 multi-task — parsing 比較対象（Table `tab:parsing-results`）。
