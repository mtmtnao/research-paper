# Scaling Laws for Neural Language Models（Transformer 言語モデルの経験的スケーリング則と compute-optimal 学習）

- arXiv: https://arxiv.org/abs/2001.08361
- 一次ソース: ../papers/arXiv-2001.08361v1/
- 正規ノート: ../notes/arXiv-2001.08361v1.md

---

## 一言で言うと

Transformer 言語モデルの cross-entropy loss は、他の要因でボトルネックされない範囲では、非埋め込みパラメータ数 $N$、データセットサイズ $D$、最小訓練 compute $C_{\rm min}$ に対して滑らかな power law に従う、という経験則を大規模実験で示す論文である。著者はこの経験則から、固定 compute では「大きなモデルを、比較的少ないデータ・少ない serial step で、収束前に止める」ことが compute-efficient だと主張する（main.tex abstract, Section 1.1, Section 6）。

## 何を議論する論文か

- **問題設定**: autoregressive language modeling の test cross-entropy loss $L$ が、モデルサイズ $N$、データ量 $D$、訓練 compute $C$、モデル形状、文脈長、batch size にどう依存するかを実験的に調べる。中心は decoder-only Transformer で、性能指標は 1024-token context 上で平均した log-likelihood / cross-entropy loss in nats である（main.tex Section 2）。
- **対象範囲 / 仮定**: 訓練データは WebText2、tokenizer は BPE、語彙数は $n_{\rm vocab}=50257$。モデルサイズ $N$ は vocabulary embedding と positional embedding を除く non-embedding parameters と定義する。主な compute 見積もりは $C \approx 6NBS$ で、文脈長依存の項は主実験の範囲では小さいとして落としている（main.tex Section 2.1）。
- **既存研究との差分**: 先行研究 Hestness et al. は model size と data size の scaling を扱ったが、本論文は WebText2 上で $N, D, C_{\rm min}$、さらに $L(N,D)$ と $L(N,S_{\rm min})$ の同時依存まで一つの枠組みに入れる。Related Work では、Hestness et al. が super-linear な data scaling を報告したのに対し、本論文は sub-linear scaling を得たと明示している（main.tex Section 7, main.bbl）。
- **この論文で答えたい問い**: 固定された計算予算のもとで、compute をモデルサイズ、batch size、serial training steps、データ使用量へどう配分すれば test loss を最も下げられるか。著者は $N\propto C_{\rm min}^{0.73}$、$B\propto C_{\rm min}^{0.24}$、$S\propto C_{\rm min}^{0.03}$ という配分を empirical / theoretical の両側から導く（main.tex Section 6, Appendix "Empirical Model of Compute-Efficient Frontier"）。

## 背景と前提

- この論文の「性能」は downstream task accuracy ではなく、主に WebText2 test distribution 上の autoregressive cross-entropy loss $L$ である。Discussion では、loss の滑らかな改善が言語タスク上の改善にどう変換されるかは重要な未検討点として扱われる。
- $N$ は total parameters ではない。embedding parameters を含めると depth 依存が強く見え、non-embedding parameter count にすると異なる depth のモデルが一つの trend にまとまる、というのが Figure \ref{fig:PerformancevsModelSizeBody} の要点である。
- WebText2 は WebText の拡張版で、Reddit outbound links をもとに作られている。TeX では 20.3M documents、96 GB text、$1.62\times10^{10}$ words、$2.29\times10^{10}$ tokens、test set $6.6\times10^8$ tokens と記載される（main.tex Section 2.3）。
- 実験では model size は 768 から 1.5B non-embedding parameters、dataset size は 22M から 23B tokens、context length は多くの run で 1024、batch size は多くの run で $2^{19}$ tokens。標準訓練は Adam、$2.5\times10^5$ steps、512 sequences of 1024 tokens、3000-step warmup 後 cosine decay to zero。1B parameters 超の最大モデルでは Adafactor を用いる（main.tex Sections 2.2, 3）。
- 比較対象は、主に形状の異なる decoder-only Transformers、LSTMs、recurrent / Universal Transformers、他分布評価として Books Corpus、Common Crawl、English Wikipedia、Internet Books である。TeX 中では後年の大規模言語モデル研究との比較は扱われない。

## 提案手法

### コアアイデア

この論文は新しいアーキテクチャを提案するというより、言語モデルの scaling を測るための実験設計と経験式を提案する。著者は、$N$、$D$、$C_{\rm min}$ のうち一つだけがボトルネックになる設定では $L(X)\propto X^{-\alpha_X}$ という power law が現れ、さらに $N$ と $D$、$N$ と訓練 step を同時に変えた場合も簡単な閉形式で近似できるとする。

重要なのは、単に「大きいほど良い」と言うのではなく、どの要因が制限になっているかを分けて測る点である。十分なデータで収束近くまで訓練すれば $L(N)$ が見え、十分大きいモデルを小さいデータで early stopping すれば $L(D)$ が見え、batch size の非効率を $B_{\rm crit}$ で補正すれば $C_{\rm min}$ に対するよりきれいな trend が見える、という構成になっている。

### 重要な定義・数式

$$
N \approx 2d_{{\rm model}}n_{{\rm layer}}\left(2d_{{\rm attn}}+d_{{\rm ff}}\right)
= 12 n_{\rm layer} d_{{\rm model}}^2
\quad
\text{with the standard } d_{\rm attn}=d_{\rm ff}/4=d_{{\rm model}},
\qquad
C \approx 6NBS
$$

**式の意味**: Transformer の non-embedding parameter count $N$ と、訓練 compute $C$ の近似を定義する式である。$C\approx6NBS$ は forward pass と backward pass を含む training FLOPs の近似で、TeX では context-dependent terms を主実験では落としている。

**記号の定義**:
- $N$ ... vocabulary / positional embeddings を除いたモデルサイズ
- $d_{\rm model}$ ... residual stream の次元
- $n_{\rm layer}$ ... Transformer layer 数
- $d_{\rm attn}$ ... attention output の次元
- $d_{\rm ff}$ ... feed-forward intermediate layer の次元
- $B$ ... batch size in tokens
- $S$ ... parameter update steps
- $C$ ... estimated non-embedding training compute

**この論文での役割**: 以後のすべての scaling law は、この $N$ と $C$ の定義に依存する。特に embedding を除いた $N$ を使うことが、Figure \ref{fig:PerformancevsModelSizeBody} の「cleaner scaling laws」につながる。

$$
\begin{aligned}
L(N) &= \left( \frac{N_{\mathrm{c}}}{N} \right)^{\alpha_N},
&\alpha_N &\sim 0.076,
&N_{\mathrm{c}} &\sim 8.8 \times 10^{13},\\
L(D) &= \left( \frac{D_{\mathrm{c}}}{D} \right)^{\alpha_D},
&\alpha_D &\sim 0.095,
&D_{\mathrm{c}} &\sim 5.4 \times 10^{13},\\
L(C_{\rm min}) &= \left( \frac{C_{\mathrm{c}}^{\rm min}}{C_{\rm min}} \right)^{\alpha_C^{\rm min}},
&\alpha_C^{\rm min} &\sim 0.050,
&C_{\mathrm{c}}^{\rm min} &\sim 3.1 \times 10^8 \text{ PF-days}.
\end{aligned}
$$

**式の意味**: それぞれ、モデルサイズだけ、データ量だけ、最小 compute だけが性能を制限しているときの test loss の power law である。Appendix の "Key parameters to trend fits" では、naive compute trend として $\alpha_C=0.057$, $C_c=1.6\times10^7$ PF-days も併記されるが、著者は extrapolation には $C_{\rm min}$ trend を使うべきだと述べる。

**記号の定義**:
- $L$ ... cross-entropy loss in nats
- $N$ ... non-embedding parameters
- $D$ ... dataset size in tokens
- $C_{\rm min}$ ... target loss に到達する最小 non-embedding compute の推定値
- $N_c,D_c,C_c^{\rm min}$ ... tokenizer / vocabulary に依存する scale parameters
- $\alpha_N,\alpha_D,\alpha_C^{\rm min}$ ... scaling exponents

**この論文での役割**: Section 1.2 と Figure \ref{fig:BasicPowerLaws} の中心結果である。著者は、これらの関係が $C_{\rm min}$ で 8 桁、$N$ で 6 桁、$D$ で 2 桁超にわたり成立すると主張する。

$$
L(N, D)
= \left[
\left( \frac{N_c}{N} \right)^{\frac{\alpha_N}{\alpha_D}}
+ \frac{D_c}{D}
\right]^{\alpha_D}
$$

**式の意味**: model size と dataset size を同時に変えたときの early-stopped test loss を表す経験式である。$D\to\infty$ では $L(N)$ に、$N\to\infty$ では $L(D)$ に近づくように作られている。

**記号の定義**:
- $L(N,D)$ ... サイズ $N$ のモデルを $D$ tokens のデータで訓練したときの test loss
- $N_c,D_c,\alpha_N,\alpha_D$ ... joint fit の parameters
- $\alpha_N/\alpha_D$ ... model size と data size の釣り合いを決める指数

**この論文での役割**: Section 4 の overfitting 解析の核である。Section 4.2 の joint fit では $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$ が得られ、Figure \ref{fig:DatasetModelSizevsPerformance} で検証される。この式から、overfitting を小さく保つにはおおよそ $D\gtrsim(5\times10^3)N^{0.74}$ が必要だと推定される。

$$
\begin{aligned}
B_{\rm crit}(L)&=\frac{B_{\ast}}{L^{1/\alpha_{B}}},
\qquad
B_{\ast}\sim 2\cdot10^8 \text{ tokens},\quad \alpha_B\sim0.21,\\
S_{\rm min}(S)&=\frac{S}{1+B_{\rm crit}(L)/B},
\qquad
C_{\rm min}(C)&=\frac{C}{1+B/B_{\rm crit}(L)}.
\end{aligned}
$$

**式の意味**: $B_{\rm crit}$ は、batch size を増やしても compute-efficiency が大きく落ちにくい境目を表す。$S_{\rm min}$ と $C_{\rm min}$ は、実際の fixed-batch training の step / compute を、極端に大きい batch または小さい batch で訓練した場合の最小 step / 最小 compute に換算する補正式である。

**記号の定義**:
- $B_{\rm crit}(L)$ ... target loss $L$ における critical batch size
- $B_\ast$ ... critical batch size の scale parameter
- $\alpha_B$ ... $B_{\rm crit}$ の power-law exponent
- $S_{\rm min}$ ... target loss に到達する最小 training steps の推定値
- $C_{\rm min}$ ... target loss に到達する最小 compute の推定値

**この論文での役割**: Section 5.1 と Figure \ref{fig:OptimalBatchSize} の要点である。著者は、$B_{\rm crit}$ は model size に直接依存せず、到達した loss に主に依存するとし、loss が 13% 減るごとに $B_{\rm crit}$ がほぼ倍になると述べる。この補正が、Section 6 の compute-efficient frontier を作る。

$$
\begin{aligned}
L(N, S_{\rm min})
&= \left( \frac{N_c}{N} \right)^{\alpha_N}
+ \left( \frac{S_c}{S_{\rm min}} \right)^{\alpha_S},
\qquad
\alpha_S\approx0.76,\quad S_c\approx2.1\times10^3,\\
N &\propto C^{\alpha_C^{\rm min}/\alpha_N},\quad
B \propto C^{\alpha_C^{\rm min}/\alpha_B},\quad
S \propto C^{\alpha_C^{\rm min}/\alpha_S},\\
\alpha_C^{\rm min}=
\frac{1}{1/\alpha_S+1/\alpha_B+1/\alpha_N}
\end{aligned}
$$

**式の意味**: 前半は、十分大きなデータがあるときの loss を model-size term と training-time term の和として近似する式である。後半は、その式と critical batch size を使って、固定 compute の最適配分を導く式である。

**記号の定義**:
- $S_{\rm min}$ ... batch 補正後の最小 step
- $S_c,\alpha_S$ ... learning curve の scale と exponent
- $C$ ... ここでは compute-efficient training の compute budget。TeX の導出では $C$ と書くが、実測の配分は $C_{\rm min}$ に対して報告される
- $B$ ... optimal batch size、実際には $B_{\rm crit}$ に従って伸びる
- $\alpha_C^{\rm min}$ ... $N,B,S$ の配分を合成して出る compute exponent

**この論文での役割**: Section 5.2, Section 6.2, Appendix "Empirical Model of Compute-Efficient Frontier" の中心である。$L(N,S_{\rm min})$ の fit は $\alpha_N=0.077$, $\alpha_S=0.76$, $N_c=6.5\times10^{13}$, $S_c=2.1\times10^3$。これから予測される $N\propto C_{\rm min}^{0.71}$ は、実測の $N\propto C_{\rm min}^{0.73}$ と数%以内で合うと著者は述べる。

### 実装 / アルゴリズム上の要点

- step1: WebText2 を BPE vocabulary size $50257$ で tokenization し、主指標を 1024-token context 平均の autoregressive cross-entropy loss に固定する。
- step2: decoder-only Transformer の $N$、depth、width、heads、$d_{\rm ff}$、context length、batch size、dataset subset size を変えた多数の run を行う。形状比較では、$N\approx12n_{\rm layer}d_{\rm model}^2$ を保つように depth と width を同時に変える。
- step3: $L(N)$、$L(D)$、fixed-batch の $L(C)$ をまず empirical に fit し、embedding を含める / 除く差、shape dependence、他分布 generalization、LSTM / Universal Transformer 比較を確認する。
- step4: finite data の run では 10% dropout を入れ、test loss が改善しなくなった時点で early stopping する。$L(N,D)$ と $\delta L(N,D)=L(N,D)/L(N,\infty)-1$ で overfitting を整理する。
- step5: batch size scan から $B_{\rm crit}(L)$ を測り、fixed-batch run の $S$ と $C$ を $S_{\rm min}$ と $C_{\rm min}$ に換算する。
- step6: $L(N,S_{\rm min})$ を fit し、その式から fixed compute budget における optimal $N,B,S,D$ の scaling を導く。

## 実験・結果

- **データセット / ベンチマーク**: 主データは WebText2。原 WebText は 2017 年 12 月までの Reddit outbound links、WebText2 は 2018 年 1 月から 10 月までの links を追加したもの。データ量は 20.3M documents、96 GB、$2.29\times10^{10}$ tokens、test set $6.6\times10^8$ tokens。追加評価は Books Corpus、Common Crawl、English Wikipedia、publicly-available Internet Books。
- **比較対象 / baseline**: 主に decoder-only Transformers の size / shape / training 条件違いを比較する。追加比較として、同じ dataset と context length で訓練した LSTMs、parameter reuse する recurrent / Universal Transformers を用いる。Related Work では Hestness et al.、Rosenfeld et al.、EfficientNet などと比較されるが、実験 baseline というより scaling 研究の文脈づけである。
- **指標**: cross-entropy loss in nats。主に WebText2 test distribution 上で測り、他分布では同じ model を評価して loss offset と相関を見る。compute は PF-days で報告され、1 PF-day は $10^{15}\times24\times3600=8.64\times10^{19}$ floating point operations。
- **主な結果**: $L(N)$ は $\alpha_N\sim0.076$、$L(D)$ は $\alpha_D\sim0.095$、$L(C_{\rm min})$ は $\alpha_C^{\rm min}\sim0.050$ の power law に従う。著者は、$C_{\rm min}$ で 8 桁、$N$ で 6 桁、$D$ で 2 桁超にわたる trend と述べる。
- **主な結果**: depth / width / number of heads / $d_{\rm ff}$ の影響は、同じ non-embedding $N$ では数%程度に弱い。Figure \ref{fig:HeadsLayersIndependence} の caption では、aspect ratio を factor 40 変えても影響は小さく、$(n_{\rm layer},d_{\rm model})=(6,4288)$ が Radford et al. の $(48,1600)$ model の 3% 以内の loss に達するとされる。
- **主な結果**: finite data での overfitting penalty はおおむね $N^{\alpha_N/\alpha_D}/D$ により整理できる。random seed による loss variation を約 0.02 と見積もり、overfitting をこの閾値内に抑えるには $D\gtrsim(5\times10^3)N^{0.74}$ が必要だと述べる。
- **主な結果**: fixed compute で最適な model size は $N_{\rm opt}=N_e C_{\rm min}^{p_N}$、$p_N=0.73$、$N_e=1.3\times10^9$ params。batch は $p_B=0.24$、$B_e=2.0\times10^6$ tokens、step は $p_S=0.03$、$S_e=5.4\times10^3$ steps、1 epoch とした data usage は $p_D=0.27$、$D_e=2\times10^{10}$ tokens（Appendix "Trends for compute-efficient training"）。
- **主な結果**: compute-efficient training は converged loss の $\alpha_N/\alpha_S\approx10\%$ 上で止めるのが最適だと Appendix "Empirical Model of Compute-Efficient Frontier" の "Efficient Training" で導く。典型的な $f'=2\%$ near-convergence training と比べると、同じ loss に到達するために 2.7x larger model、7.7x fewer parameter updates、65% less compute と予測する。
- **主な結果**: LSTM は context の早い token では Transformer と同程度だが、後の token では Transformer に届かない（Figure \ref{fig:LSTMvsTransformers} 周辺）。Universal Transformers は同じ parameter count では少し良いが、parameter reuse の compute を考えると per FLOP では少し悪い（Appendix Supplemental Figures）。
- **主な結果**: 他分布への generalization は WebText2 validation loss と強く相関し、training duration や convergence への近さには依存しない、と著者は述べる（Section 3.2.2, Figure \ref{fig:GeneralizationVsModelSize}）。
- **著者が主張する貢献**: $N,D,C_{\rm min}$ の単独 scaling、$L(N,D)$ と $L(N,S_{\rm min})$ の同時依存、critical batch size を使った compute 補正、固定 compute budget の最適配分、large models の sample efficiency、収束前停止の compute efficiency を一つの予測的枠組みにまとめた点。

## 妥当性と限界

- **この主張を支える根拠**: 実験設計は、model size、dataset size、shape、context length、batch size を広く振り、複数の図で power-law fit を確認する構成になっている。特に Figure \ref{fig:BasicPowerLaws}、Figure \ref{fig:LossvsModelDatasetSize}、Figure \ref{fig:DatasetModelSizevsPerformance}、Figure \ref{fig:ComputeEfficientAdjusted}、Figure \ref{fig:ComputevsPerformance} が主張の根拠である。
- **この主張を支える根拠**: $L(N,S_{\rm min})$ から導いた理論的配分 $N\propto C_{\rm min}^{0.71}$ が、実測の $N\propto C_{\rm min}^{0.73}$ と近い点は、単なる curve fitting 以上の整合性を与えている。ただしこの「理論」は経験式を前提にした導出であり、power law 自体の第一原理的説明ではない。
- **著者が認めている limitations / future work**: Caveats では、提案した scaling laws に solid theoretical understanding がないこと、$B_{\rm crit}(L)$ を探索範囲外に外挿する自信が高くないこと、小データ領域を十分調べておらず最小 $D$ では fit が悪いことを挙げる。
- **著者が認めている limitations / future work**: regularization と data augmentation を最適化していない。dropout は 10% 固定であり、これが $L(N,D)$ や overfitting の係数を変える可能性がある。
- **著者が認めている limitations / future work**: $C\approx6NBS$ は context-dependent terms を無視しており、$n_{\rm ctx}\gtrsim12d_{\rm model}$ の regime では compute scaling が confounded されうる。初期化 scale、momentum などの hyperparameter tuning も完全ではなく、target loss によって最適 learning rate が変わる可能性も述べられている。
- **著者が認めている limitations / future work**: Discussion では、scaling relations が images、audio、video、random network distillation など他の generative modeling tasks にも成り立つか、また loss 改善が relevant language tasks の改善に結びつくかを今後の重要課題としている。
- **読者として注意すべき点**: $N_c,D_c,C_c$ は tokenizer と vocabulary に依存し fundamental meaning はない、と TeX 中で明示される。したがって、係数を別データ・別 tokenizer にそのまま移すべきではなく、主に exponent と関係式の形を読むべきである。
- **読者として注意すべき点**: compute-optimal の議論は training compute を中心にしており、inference cost は目的関数に入っていない。Appendix "Suboptimal Model Sizes" では、inference cost を考えると小さい model size が有用な場合があるとだけ触れられている。
- **読者として注意すべき点**: Section 6.3 の $C^*\sim10^4$ PF-days、$N^*\sim10^{12}$、$D^*\sim10^{12}$、$L^*\sim1.7$ nats/token は、著者自身が "highly uncertain" とする conjectural な交点である。自然言語の entropy-per-token 推定として読む可能性は提示されるが、確定的な結果ではない。
- **追加で確認したい実験 / 疑問**: Caveats と Discussion に沿えば、データ分布・データ品質・正則化・長文脈 regime を変えたときに exponents と optimal allocation が保たれるか、また cross-entropy loss の改善が具体的な言語タスクでどう現れるかを確認する必要がある。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **$L$ / loss**: autoregressive language modeling の cross-entropy loss in nats。通常は context 内 token で平均するが、appendix では context position ごとの loss も扱う。
- **$N$ / model size**: vocabulary embedding と positional embedding を除いた non-embedding parameters。embedding を含めた total parameters では主 scaling が崩れやすい。
- **$D$ / dataset size**: 訓練データの token 数。WebText2 全体は $2.29\times10^{10}$ tokens で、test set に $6.6\times10^8$ tokens を取る。
- **$C$**: 実際の training compute の近似。主に $C=6NBS$ と置くが、fixed batch の $C$ は optimal compute ではない。
- **$C_{\rm min}$**: target loss へ到達するための最小 compute の推定。$B\ll B_{\rm crit}$ で訓練した場合に対応する補正量として定義される。
- **$S_{\rm min}$**: target loss へ到達するための最小 parameter update steps の推定。$B\gg B_{\rm crit}$ で訓練した場合に対応する。
- **$B_{\rm crit}$**: critical batch size。batch size を増やしても compute-efficiency が大きく悪化しない境目で、論文では loss の power law として fit される。
- **sample efficiency**: この論文では、同じ loss に到達するために必要な data examples または optimization steps が少ないことを指す。large models は small models より sample-efficient だとされる。
- **compute-efficient training**: fixed compute で最小 loss を得る training allocation。著者の結論では、非常に大きい model を収束前に止める設定になる。
- **overfitting penalty $\delta L$**: finite data loss を infinite-data limit と比べた相対増加、$\delta L(N,D)=L(N,D)/L(N,\infty)-1$。主に $N^{0.74}/D$ で整理される。
- **WebText2**: WebText の拡張版。Reddit outbound links のうち minimum 3 karma を満たすものをもとにしたデータで、2018 年 1 月から 10 月分を追加している。
- **Universal / recurrent Transformer**: parameters を reuse する Transformer。TeX では同じ $N$ で少し良く、compute を考えると少し悪い比較対象として扱われる。

## 読む順番の提案

- まず `notes/arXiv-2001.08361v1.md` の Summary を読んで、$N,D,C_{\rm min}$、$B_{\rm crit}$、compute-efficient allocation の全体像を押さえる。ただし数値は必ず main.tex の Section 1.2 と Appendix "Summary of Power Laws" に戻って確認する。
- 次に main.tex の Section 2 を読む。ここで $N$ が non-embedding parameters であること、$C\approx6NBS$ の前提、WebText2 のサイズ、optimizer と training schedule を確認する。正規ノートの "手法" と "Takeaway: 埋め込みを含めると trend が汚れる" に対応する。
- その後 Section 3 と Figure \ref{fig:BasicPowerLaws}、Figure \ref{fig:HeadsLayersIndependence}、Figure \ref{fig:PerformancevsModelSizeBody} を読む。単独の $L(N),L(D),L(C)$ と shape independence がここで出る。
- Section 4 では Equation \ref{eq:FundamentalLikelihioodvsModelandDataSize} と Figure \ref{fig:DatasetModelSizevsPerformance} を優先して読む。正規ノートの "overfitting はほぼ $N^{0.74}/D$" につながる。
- Section 5 では $B_{\rm crit}$、Equation \ref{eq:AdjustedSteps}、Equation \ref{eq:AdjustedCompute}、Equation \ref{eq:FundamentalLikelihioodvsModelandSteps2} を読む。正規ノートの "critical batch size の道具化" に対応する。
- Section 6 と Appendix "Empirical Model of Compute-Efficient Frontier" は、compute-efficient training の結論を理解するために読む。特に $N\propto C_{\rm min}^{0.73}$、$S\propto C_{\rm min}^{0.03}$、converged loss の約 10% 上で止めるという導出が正規ノートの中心的 Takeaway に対応する。
- 最後に Section 8 Discussion と Appendix "Caveats" を読む。power law の理論的根拠がないこと、他ドメインや downstream tasks への外挿が未検証であることを確認してから、正規ノートの Critical Thoughts を読むと評価と事実を分けやすい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2001.08361v1/`
- 正規ノート: `notes/arXiv-2001.08361v1.md`
