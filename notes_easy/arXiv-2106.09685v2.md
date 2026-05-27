# LoRA: Low-Rank Adaptation of Large Language Models（大規模言語モデルのパラメータ効率的適応）

- arXiv: https://arxiv.org/abs/2106.09685
- 一次ソース: ../papers/arXiv-2106.09685v2/
- 正規ノート: ../notes/arXiv-2106.09685v2.md

---

## 一言で言うと

LoRA は、巨大な事前学習済み言語モデルの重み $W_0$ を凍結し、下流タスクで必要な更新 $\Delta W$ だけを低ランク分解 $BA$ として学習する方法である。著者は、RoBERTa、DeBERTa、GPT-2、GPT-3 175B の実験で、full fine-tuning と同等以上の性能を、少ない trainable parameters と追加推論レイテンシなしで得られると主張している。

## 何を議論する論文か

- **問題設定**: 大規模な事前学習済み言語モデルを複数の下流タスクへ適応するとき、full fine-tuning はタスクごとに元モデルと同じ規模のパラメータ差分 $\Delta\Phi$ を持つ。GPT-3 175B のようなモデルでは、独立した fine-tuned モデルをタスクごとに保存・配備することが「prohibitively expensive」になる、というのが出発点である（`iclr2022_conference.tex`, Abstract / Sec. 1 / Sec. 2）。
- **対象範囲 / 仮定**: 手法自体は dense layer 一般に適用できると述べるが、実験では Transformer language models の self-attention weights に焦点を当てる。特に多くの実験では $W_q$ と $W_v$ に LoRA を適用し、MLP modules は凍結する（Sec. 4.2, `expt.tex` Sec. 5.1）。
- **既存研究との差分**: adapter layers はモデル深さを増やすため推論レイテンシを増やしうる。prefix / prompt 系は trainable special tokens が入力長を消費し、最適化が難しく非単調な性能変化を示すと著者は述べる（Sec. 3, Table 1, Fig. 2）。LoRA は低ランク更新を元の重みに merge できるため、fully fine-tuned model と同じ形で推論できる点が主要な差分である。
- **この論文で答えたい問い**: 事前学習済みモデルの下流タスク適応における重み更新 $\Delta W$ は、低い「intrinsic rank」で表現できるのか。その低ランク更新だけで品質・保存量・訓練メモリ・推論レイテンシのトレードオフを改善できるのか。

## 背景と前提

- この論文では、事前学習済み autoregressive language model を $P_\Phi(y|x)$ と書く。下流タスクは context-target pairs $\mathcal{Z}=\{(x_i,y_i)\}_{i=1,..,N}$ で表され、例として summarization、machine reading comprehension、NL2SQL が挙げられる（Sec. 2）。
- Transformer の記法として、入力・出力次元を $d_{model}$、self-attention の query/key/value/output projection matrices を $W_q, W_k, W_v, W_o$ と呼ぶ。$W$ または $W_0$ は事前学習済み重み、$\Delta W$ は adaptation 中に蓄積される重み更新、$r$ は LoRA module の rank である（Sec. 1, "Terminologies and Conventions"）。
- full fine-tuning は全パラメータ $\Phi$ を更新する。parameter-efficient adaptation は、タスク固有の増分 $\Delta\Phi(\Theta)$ を小さいパラメータ集合 $\Theta$ で表し、$|\Theta|\ll|\Phi_0|$ にする方針である（Sec. 2）。
- 比較対象は Fine-Tuning、$\mathrm{FT}^{Top2}$、BitFit、Prefix-embedding tuning (PreEmbed)、Prefix-layer tuning (PreLayer)、Adapter$^H$、Adapter$^L$、Adapter$^P$、Adapter$^D$ である（`expt.tex` Sec. 5.1）。著者は、prior work の設定や報告値を再利用する場合があるため、すべての baseline が全実験に出るわけではないと明記している。
- LoRA の低ランク仮説は、over-parametrized model が low intrinsic dimension にあるという prior work（Li et al. 2018, Aghajanyan et al. 2020）から着想されている。ただし、この論文が直接扱うのは「モデル全体の intrinsic dimension」ではなく、adaptation 中の重み更新 $\Delta W$ の「intrinsic rank」である。

## 提案手法

### コアアイデア

LoRA は、事前学習済み重み $W_0\in\mathbb{R}^{d\times k}$ を凍結し、その更新 $\Delta W$ を $BA$ に制約する。ここで $B\in\mathbb{R}^{d\times r}$、$A\in\mathbb{R}^{r\times k}$、$r\ll\min(d,k)$ である。学習中に gradient update を受けるのは $A$ と $B$ だけで、$W_0$ は更新されない（Sec. 4.1, Eq. 3）。

forward pass では、元の出力 $W_0x$ と低ランク更新の出力 $BAx$ を coordinate-wise に足す。$A$ は random Gaussian initialization、$B$ は zero initialization なので、学習開始時は $\Delta W=BA=0$ である。さらに著者は $\Delta Wx$ を $\alpha/r$ で scale し、$\alpha$ は最初に試した $r$ に設定して tune しないと述べる（Sec. 4.1）。

デプロイ時には $W=W_0+BA$ を明示的に計算・保存できる。したがって推論時は通常の dense layer と同じ形で実行でき、LoRA module の追加計算を別に通す必要がない。タスクを切り替えるときは $BA$ を引いて別の $B'A'$ を足す、という操作で元の $W_0$ から別タスクの重みに移れる（Sec. 4.1, "No Additional Inference Latency"）。

### 重要な定義・数式

$$
    \max_{\Phi} \sum_{(x,y)\in\mathcal{Z}} \sum_{t=1}^{|y|}  \log \left(  P_{\Phi}(y_{t} | x, y_{<t}) \right)
$$

**式の意味**: full fine-tuning の conditional language modeling objective である。事前学習済みモデルを初期値 $\Phi_0$ とし、全パラメータ $\Phi$ を更新して、各 target token $y_t$ の条件付き確率を最大化する（Sec. 2, Eq. 1）。

**記号の定義**:
- $\Phi$ ... 言語モデル全体のパラメータ
- $\mathcal{Z}$ ... 下流タスクの training dataset
- $(x,y)$ ... context と target のペア
- $y_t$ ... target sequence の $t$ 番目の token
- $y_{<t}$ ... $t$ 番目より前の target tokens
- $P_\Phi(y_t|x,y_{<t})$ ... パラメータ $\Phi$ のもとでの次 token 条件付き確率

**この論文での役割**: full fine-tuning が何を最適化しているかを定義し、その後の「全パラメータを更新する必要があるのか」という問題提起の基準になる。

$$
    \max_{\Theta} \sum_{(x,y)\in\mathcal{Z}}  \sum_{t=1}^{|y|}  \log\left(p_{\Phi_0+\Delta\Phi(\Theta)}(y_{t} | x, y_{<t})\right)
$$

**式の意味**: parameter-efficient adaptation の目的関数である。元の $\Phi_0$ を直接すべて更新するのではなく、小さいパラメータ集合 $\Theta$ から作る $\Delta\Phi(\Theta)$ でモデルを適応する（Sec. 2, Eq. 2）。TeX の括弧崩れは、式の意図が読めるように補っている。

**記号の定義**:
- $\Theta$ ... task-specific increment を符号化する小さい trainable parameters
- $\Phi_0$ ... 事前学習済みの frozen parameters
- $\Delta\Phi(\Theta)$ ... $\Theta$ により表されるタスク固有のパラメータ増分
- $|\Theta| \ll |\Phi_0|$ ... この論文が狙うパラメータ効率性の条件

**この論文での役割**: LoRA はこの $\Delta\Phi(\Theta)$ の具体化として、dense weight update を低ランク分解 $BA$ で表す。GPT-3 175B では $|\Theta|$ が $|\Phi_0|$ の $0.01\%$ まで小さくできると述べられる（Sec. 2）。

$$
h = W_0 x + \Delta W x = W_0 x + BA x
$$

**式の意味**: LoRA の modified forward pass である。元の frozen weight $W_0$ による出力と、低ランク更新 $BA$ による出力を足して layer output $h$ を作る（Sec. 4.1, Eq. 3）。

**記号の定義**:
- $W_0\in\mathbb{R}^{d\times k}$ ... 事前学習済みの frozen weight matrix
- $\Delta W$ ... adaptation 中の weight update
- $B\in\mathbb{R}^{d\times r}$ ... LoRA の trainable matrix
- $A\in\mathbb{R}^{r\times k}$ ... LoRA の trainable matrix
- $r$ ... LoRA module の rank、$r\ll\min(d,k)$
- $x$ ... dense layer への入力
- $h$ ... dense layer の出力

**この論文での役割**: LoRA の中心式であり、追加推論レイテンシがない理由もここから来る。$W_0$ と $BA$ は同じ shape の行列なので、デプロイ時に $W=W_0+BA$ と merge できる。

$$
|\Theta| = 2 \times \hat{L}_{LoRA} \times d_{model} \times r
$$

**式の意味**: LoRA の trainable parameters 数を数える式である（`expt.tex` Sec. 5.1）。元の weight shape と rank $r$、LoRA を適用する weight matrices の数から、追加で学習するパラメータ数が決まる。

**記号の定義**:
- $|\Theta|$ ... LoRA で学習するパラメータ数
- $\hat{L}_{LoRA}$ ... LoRA を適用する weight matrices の数
- $d_{model}$ ... Transformer layer の input/output dimension size
- $r$ ... LoRA rank

**この論文での役割**: full fine-tuning や adapter / prefix 系との比較で、LoRA がどれだけ少ない trainable parameters で動くかを定量化する。GPT-3 の実験では 4.7M や 37.7M trainable parameters の LoRA が Table 4 に出る。

$$
  \phi(A_{r=8}, A_{r=64}, i, j) = \frac{||U_{A_{r=8}}^{i\top} U_{A_{r=64}}^j||_{F}^2}{\min(i, j)} \in [0,1]
$$

**式の意味**: rank が異なる LoRA matrices の特異方向がどの程度重なるかを測る normalized subspace similarity である。Grassmann distance に基づく類似度として使われる（Sec. 7.2, Eq. 4）。

**記号の定義**:
- $A_{r=8}$ ... rank 8 の LoRA で学習された adaptation matrix
- $A_{r=64}$ ... rank 64 の LoRA で学習された adaptation matrix
- $U_{A_{r=8}}^i$ ... $A_{r=8}$ の top-$i$ singular vectors に対応する columns
- $U_{A_{r=64}}^j$ ... $A_{r=64}$ の top-$j$ singular vectors に対応する columns
- $||\cdot||_F$ ... Frobenius norm
- $\phi(\cdot)\in[0,1]$ ... 1 が完全な overlap、0 が完全な separation を表す類似度

**この論文での役割**: 「小さい $r$ で足りるのはなぜか」を経験的に調べるための式である。Fig. 3 では $A_{r=8}$ と $A_{r=64}$ の top singular vector directions が大きく重なることを示し、rank-deficiency の主張を支える。

### 実装 / アルゴリズム上の要点

- step1: 事前学習済み weight matrix $W_0$ を凍結し、gradient update を受けないようにする。
- step2: 対象 weight matrix に LoRA matrices $A$ と $B$ を追加する。$A$ は random Gaussian、$B$ は zero で初期化し、初期状態の $\Delta W=BA$ を 0 にする。
- step3: forward では $W_0x$ と $BAx$ を足す。TeX 本文では、実装上 $\Delta Wx$ を $\alpha/r$ で scale し、$\alpha$ は最初に試した $r$ に固定して tune しないと説明される。
- step4: Transformer 実験では self-attention の $W_q,W_k,W_v,W_o$ のどこに適用するかを調べる。多くの実験では $W_q$ と $W_v$ に適用する。
- step5: MLP modules は凍結する。MLP layers、LayerNorm layers、biases への LoRA 適用は future work として残されている（Sec. 4.2）。
- step6: デプロイ時は $W=W_0+BA$ を計算して通常の weight として使う。別タスクに切り替える場合は $BA$ を取り除き、別の $B'A'$ を加える。

## 実験・結果

- **データセット / ベンチマーク**: RoBERTa と DeBERTa では GLUE benchmark を使う。GLUE の内訳は MNLI、SST-2、MRPC、CoLA、QNLI、QQP、RTE、STS-B である（Appendix C）。GPT-2 では E2E NLG Challenge を本文に示し、appendix で WebNLG と DART も報告する。GPT-3 175B では WikiSQL、MultiNLI-matched、SAMSum を使う（`expt.tex` Sec. 5）。
- **比較対象 / baseline**: Fine-Tuning、$\mathrm{FT}^{Top2}$、BitFit、PreEmbed、PreLayer、Adapter$^H$、Adapter$^L$、Adapter$^P$、Adapter$^D$、LoRA。Adapter$^H$ は Houlsby et al. 型、Adapter$^L$ は Lin et al. 型として扱われる（`expt.tex` Sec. 5.1）。
- **指標**: GLUE では MNLI の overall accuracy、CoLA の Matthew's correlation、STS-B の Pearson correlation、その他の tasks の accuracy を報告する（Table 2 caption）。GPT-2 E2E では BLEU、NIST、METEOR、ROUGE-L、CIDEr を使い、すべて higher is better（Table 3）。GPT-3 では WikiSQL の logical form validation accuracy、MultiNLI-matched の validation accuracy、SAMSum の Rouge-1/2/L を使う（Table 4 caption）。
- **主な結果**: GLUE では RoBERTa base LoRA が 0.3M trainable parameters で Avg. 87.2、RoBERTa base FT は 125.0M で 86.4。RoBERTa large LoRA は 0.8M で 89.0、FT は 355.0M で 88.9。DeBERTa XXL LoRA は 4.7M で 91.3、FT は 1500.0M で 91.1（Table 2）。
- **主な結果**: GPT-2 M の E2E NLG では、LoRA 0.35M が BLEU 70.4、NIST 8.85、METEOR 46.8、ROUGE-L 71.8、CIDEr 2.53。FT 354.92M は 68.2 / 8.62 / 46.2 / 71.0 / 2.47、PreLayer 0.35M は 69.7 / 8.81 / 46.1 / 71.4 / 2.49 である。GPT-2 L では LoRA 0.77M が 70.4 / 8.89 / 46.8 / 72.0 / 2.47、PreLayer 0.77M が 70.3 / 8.85 / 46.2 / 71.7 / 2.47（Table 3）。
- **主な結果**: GPT-3 175B では、LoRA 4.7M が WikiSQL 73.4、MNLI-m 91.7、SAMSum 53.8/29.8/45.9。FT 175,255.8M は 73.8、89.5、52.0/28.0/44.5。LoRA 37.7M は 74.0、91.6、53.4/29.2/45.1（Table 4）。Table 4 caption は、WikiSQL の fluctuation を約 $\pm0.5\%$、MNLI-m を約 $\pm0.1\%$、SAMSum を約 $\pm0.2/\pm0.2/\pm0.1$ としている。
- **主な結果**: 推論レイテンシでは、GPT-2 medium の single forward pass で、batch 1 / sequence length 128 の Fine-Tune/LoRA が 19.8 ms、Adapter$^L$ が 23.9 ms（+20.7%）、Adapter$^H$ が 25.8 ms（+30.3%）である（Table 1）。LoRA は merge すれば Fine-Tune と同じ latency として扱われる。
- **主な結果**: GPT-3 175B の実用上の削減として、VRAM consumption は 1.2TB から 350GB、checkpoint size は $r=4$ かつ query/value projection matrices のみの適用で 350GB から 35MB に下がる。training throughput は 32.5 から 43.1 tokens/s per V100 GPU になり、著者は 25% speedup と書いている（Sec. 4.2）。
- **著者が主張する貢献**: LoRA は trainable parameters を大幅に減らし、Adam 使用時の GPU memory requirement を 3 倍改善し、adapter と異なり no additional inference latency を持つ。さらに、rank-deficiency の経験的分析により、なぜ低い rank で adaptation が可能かを調べている（Abstract, Sec. 7）。

## 妥当性と限界

- **この主張を支える根拠**: 評価は encoder 系の RoBERTa / DeBERTa と decoder 系の GPT-2 / GPT-3 175B、NLU と NLG の両方を含む。LoRA が full fine-tuning と同等以上であるという主張は、Table 2、Table 3、Table 4 の複数モデル・複数データセットの結果で支えられている。
- **この主張を支える根拠**: 追加推論レイテンシがないという主張は、低ランク分岐を $W_0+BA$ に merge できる線形設計に基づく。adapter latency の問題は Table 1 と Appendix B の GPT-2 medium latency study で定量化されている。
- **この主張を支える根拠**: rank-deficiency の主張は Table 5 / Table 6 と Sec. 7.2 の subspace similarity analysis に基づく。同じ 18M trainable parameter budget では、$W_q$ 単独 $r=8$ より、$W_q,W_v$ $r=4$ または $W_q,W_k,W_v,W_o$ $r=2$ の方が良い結果を出す（Table 5）。Table 6 では、$W_q,W_v$ に適用する場合、WikiSQL で $r=1$ が 73.4、$r=64$ が 73.5、MNLI で $r=1$ が 91.3、$r=64$ が 91.4 である。
- **この主張を支える根拠**: $\Delta W$ と $W$ の関係について、48th layer の GPT-3 $W_q$ で $\|W_q\|_F=61.95$、$r=4$ の $\|\Delta W_q\|_F=6.91$、$\Delta W_q$ の subspace への射影 $\|U^\top W_qV^\top\|_F=0.32$、random baseline は 0.02、$W_q$ の top-$r$ directions は 21.67 と報告される（Table 7）。著者はここから、$\Delta W$ は $W$ の top singular directions を単に繰り返すのではなく、$W$ で強調されていない方向を amplify すると結論する。
- **著者が認めている limitations / future work**: $A,B$ を $W$ に吸収して追加レイテンシを消す場合、異なる $A,B$ を持つ複数タスクの inputs を同一 forward pass の batch に混ぜることは straightforward ではない（Sec. 4.2）。非マージで動的に LoRA modules を選べば可能だが、latency が critical でない場合の選択肢として述べられている。
- **著者が認めている limitations / future work**: どの weight matrices に LoRA を適用するかは heuristic に依存しており、more principled ways は future work とされる（Sec. 8）。MLP layers、LayerNorm layers、biases への適用も future work として残されている（Sec. 4.2）。
- **著者が認めている limitations / future work**: Sec. 7.2 の footnote では、小さい $r$ がすべての task / dataset で働くとは期待しないと述べる。例として、pre-training と異なる言語の downstream task では、小さい $r$ の LoRA より entire model retraining の方がよい可能性が挙げられている。
- **読者として注意すべき点**: GPT-3 実験では training cost のため、各 entry の standard deviation ではなく task ごとの typical standard deviation を報告している（`expt.tex`, `sec:gpt3_expts`）。差が小さい比較では、この点を意識して読む必要がある。
- **読者として注意すべき点**: baseline の一部は prior works の reported numbers であり、すべてが同一実装・同一チューニング条件で再評価されているわけではない（`expt.tex` Sec. 5.1）。Table 2 では adapter と比較しやすい restricted setup の runs に $\dagger$ が付く。
- **追加で確認したい実験 / 疑問**: Sec. 4.2 が future work とする MLP / LayerNorm / bias への適用、Sec. 8 が future work とする適用 weight matrices の principled selection、Sec. 7.2 footnote が触れる pre-training と異なる言語での小 rank の限界は、LoRA の仮定を読む上で自然な確認点である。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **LoRA** ... Low-Rank Adaptation。凍結した pre-trained weights に対して、低ランク行列 $BA$ で表される更新だけを学習する adaptation method。
- **full fine-tuning / FT** ... 事前学習済みモデルの全パラメータを更新する baseline。タスクごとに $|\Delta\Phi|=|\Phi_0|$ の差分を持つことが問題になる。
- **parameter-efficient adaptation** ... 小さい task-specific parameters $\Theta$ で $\Delta\Phi(\Theta)$ を表す方針。LoRA、adapter、prefix tuning、BitFit などが比較される。
- **$W_0$** ... 事前学習済みの frozen weight matrix。LoRA では gradient update を受けない。
- **$\Delta W$** ... adaptation 中に必要な重み更新。LoRA では $\Delta W=BA$ と低ランクに制約する。
- **$A,B$** ... LoRA の trainable rank decomposition matrices。$A\in\mathbb{R}^{r\times k}$ は Gaussian initialization、$B\in\mathbb{R}^{d\times r}$ は zero initialization。
- **rank $r$** ... LoRA module の低ランク次元。GPT-3 の Table 6 では $r=1,2,4,8,64$ が比較される。
- **intrinsic rank** ... adaptation 中の $\Delta W$ が実質的に低 rank で足りるという著者の仮説を指す語。Sec. 7 では subspace similarity と Frobenius norm により経験的に調べる。
- **$W_q,W_k,W_v,W_o$** ... Transformer self-attention module の query/key/value/output projection matrices。LoRA はこれらの一部、特に多くの実験では $W_q,W_v$ に適用される。
- **MLP modules** ... Transformer 内の feedforward module。この論文の実験では凍結され、LoRA 適用の empirical investigation は future work とされる。
- **Adapter$^H$ / Adapter$^L$** ... adapter tuning の baseline。LoRA と違い、sequential な追加層として計算されるため推論レイテンシを増やしうる。
- **PreEmbed / PreLayer** ... prefix-based baseline。PreEmbed は special tokens の embeddings を学習し、PreLayer は各 Transformer layer 後の activations を学習する。
- **BitFit** ... bias vectors だけを訓練し、他を凍結する baseline。
- **no additional inference latency** ... LoRA の $BA$ を $W_0$ に merge すれば、推論時の計算グラフが fully fine-tuned model と同じ形になる、という意味で使われる。
- **subspace similarity $\phi$** ... rank が異なる LoRA matrices の singular vector subspaces の重なりを見るための指標。Sec. 7.2、`app:grassmann_distance`、`app:corr_lora` で使われる。
- **amplification factor** ... `app:amplification_factor` で使われる $\|\Delta W\|_F/\|U^\top WV^\top\|_F$。$\Delta W$ が、元の $W$ では強調されていなかった方向をどれだけ増幅しているかを見る量として説明される。

## 読む順番の提案

- まず Sec. 2 の Eq. 1 と Eq. 2 を読むと、full fine-tuning と parameter-efficient adaptation の違いが明確になる。正規ノートの Summary 冒頭の「問題」と対応する。
- 次に Sec. 4.1 の Eq. 3 と Fig. 1 を読む。ここが LoRA の最小実装であり、正規ノートの「手法」および Notes / Quotes の forward 式につながる。
- Sec. 4.2 を読んで、Transformer では self-attention weights に限っていること、MLP / LayerNorm / bias は future work であること、GPT-3 の 350GB to 35MB・1.2TB to 350GB・32.5 to 43.1 tokens/s/V100 の根拠を確認する。
- 実験は `expt.tex` の Table 2、Table 3、Table 4 を先に見る。正規ノートの Results bullet はこの 3 つの表を圧縮したものなので、数値の対応を確認しやすい。
- LoRA の設計判断を読みたい場合は Sec. 7.1 の Table 5 と Sec. 7.2 の Table 6 を読む。同じ parameter budget でどの attention weights に当てるか、rank $r$ をどこまで小さくできるかが分かる。
- rank-deficiency の根拠を追う場合は Sec. 7.2 の Eq. 4、Fig. 3、Fig. 4、`app:grassmann_distance`、`app:corr_lora` を読む。正規ノートの「部分空間解析」と対応する。
- $\Delta W$ と $W$ の関係は Sec. 7.3 の Table 7、`app:corr_w_delta_w`、`app:amplification_factor` を読む。正規ノートの「$\Delta W$ vs $W$」と「amplification factor」の記述につながる。
- 最後に Sec. 8 を読み、limitations / future work を確認する。正規ノートの Critical Thoughts のうち、著者が明示した制限と読者側の疑問を分けて読むとよい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2106.09685v2/`
- 正規ノート: `notes/arXiv-2106.09685v2.md`
