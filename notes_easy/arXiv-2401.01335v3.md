# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models（追加人手データなしの LLM 自己改善・アラインメント）

- arXiv: https://arxiv.org/abs/2401.01335
- 一次ソース: ../papers/arXiv-2401.01335v3/
- 正規ノート: ../notes/arXiv-2401.01335v3.md

---

## 一言で言うと

SFT 済み LLM を、追加の preference data や reward model なしに、同じ SFT dataset と過去 iteration の自分の生成だけでさらに改善できるかを問う論文である。提案手法 Self-Play fIne-tuNing (`SPIN`) は、旧モデルの応答 $\yb'$ と target data distribution の応答 $\yb$ を識別する目的関数でモデルを反復 fine-tuning し、`zephyr-7b-sft-full` の HuggingFace Open LLM Leaderboard 平均を 58.14 から 63.16 へ上げたと報告する（main.tex, Table `tab:main`）。

## 何を議論する論文か

- **問題設定**: SFT は人間デモンストレーション、RLHF / RLAIF / DPO は preference data や reward function に依存する。著者は Introduction で `Can we empower a weak LLM to improve itself without acquiring additional human annotated data?` と問い、SFT 済みモデル $p_{\btheta_0}$ を追加 human/AI feedback なしに強くする方法を扱う。
- **対象範囲 / 仮定**: 入力は高品質 SFT dataset $S_{\mathrm{SFT}}=\{(\xb,\yb)\}_{i=1}^{n}$ と SFT 済み LLM $p_{\btheta_0}$ である。理論では target data distribution $p_{\mathrm{data}}(\cdot|\xb)$ が固定され、かつ必要な分布が LLM space $\{p_{\btheta}(\yb|\xb)|\btheta\in\bTheta\}$ に含まれる、という仮定が置かれる（Theorem 1/2）。
- **既存研究との差分**: DPO は $(\xb,\yb_w,\yb_l)$ という preference dataset と Bradley-Terry model に基づく。一方、`SPIN` は SFT pairs $(\xb,\yb)$ だけを使い、旧 iteration のモデル $p_{\btheta_t}$ が生成した $\yb'$ を比較相手にする。GAN / GAIL に近い二者ゲームの形を持つが、main player と opponent player は別ネットワークではなく同じ LLM の連続する iteration である。
- **この論文で答えたい問い**: 追加データを集めずに、SFT dataset の未活用の信号を self-play で引き出せるか。実験では、単純な追加 SFT が平均 58.14 から 57.23 に下がる一方、`SPIN` は 50k prompts の利用で平均 63.16 まで改善することを示す（Table `tab:ablation_data`, `tab:main`）。

## 背景と前提

- LLM は prompt $\xb=[x_1,\ldots,x_n]$ を受け取り response $\yb=[y_1,\ldots,y_m]$ を逐次生成する条件付き分布 $p_{\btheta}(\cdot|\xb)$ として扱われる。TeX では自己回帰分解 $p_{\btheta}(\yb|\xb)=\prod_{j=1}^{m}p_{\btheta}(y_j|\xb,\yb_{<j})$ を Problem Setting で導入する。
- $q(\cdot)$ は prompts の分布、$p_{\mathrm{data}}(\cdot|\xb)$ は training data における high-quality responses の条件付き分布である。SFT は $p_{\btheta}$ を $p_{\mathrm{data}}$ に近づける negative log-likelihood 最小化として定義される。
- RL fine-tuning は reward $r(\xb,\yb)$ を最大化しつつ KL regularization で reference model $p_{\mathrm{ref}}$ から離れすぎないようにする。ただし reward function や preference dataset の構築が主なコストになる、というのが本論文の動機である。
- `SPIN` の近縁概念は self-play, IPM, GAN, DPO, curriculum learning である。Appendix の Further Related Work では、training data が「easy-to-hard」に変わる点で curriculum learning と似ていると説明されるが、手法の核はあくまで旧 LLM と target data distribution の識別である。
- 実験上の出発点は `zephyr-7b-sft-full` で、これは Mistral-7B を Ultrachat200k で SFT したモデルとして説明される。Ultrachat200k は、UltraChat corpus の high-quality 200k subset で、元 corpus は約 1.4M dialogues and produced using OpenAI's Turbo APIs と TeX に記載されている（Experiment Setup）。

## 提案手法

### コアアイデア

`SPIN` は iteration $t+1$ で旧モデル $p_{\btheta_t}$ を opponent player、新しく学ぶモデル $p_{\btheta_{t+1}}$ を main player とみなす。旧モデルは SFT dataset の prompt $\xb$ に対して synthetic response $\yb'\sim p_{\btheta_t}(\cdot|\xb)$ を生成する。main player は、SFT dataset 側の response $\yb\sim p_{\mathrm{data}}(\cdot|\xb)$ と旧モデルの response $\yb'$ を区別できるように訓練される。

重要なのは、ここで外部の人間評価や GPT-4 preference label を作らない点である。`SPIN` は「SFT target response を preferred 側、旧モデル生成を non-target 側」とする distribution-level の比較を作り、DPO に似た形の損失へ落とし込む。ただし著者は、DPO は Bradley-Terry preference model に基づく single-iteration 的手法で、`SPIN` は IPM に基づく iterative self-play だと区別している（Section `Comparison between SPIN and DPO`）。

実験表の `SPIN iteration 0` は、base model `zephyr-7b-sft-full` が生成した response を使う最初の self-play fine-tuning 結果を指す。その後、iteration 1, 2, 3 では直前の `SPIN` checkpoint が新たな synthetic response を生成し、それが次の訓練に使われる。

### 重要な定義・数式

$$
L_{\mathrm{SFT}}(\btheta) =
-\EE_{\xb\sim q(\cdot), \yb\sim p_{\mathrm{data}}(\cdot|\xb)}
\Big[\log p_{\btheta}\big(\yb|\xb\big)\Big].
$$

**式の意味**: SFT の negative log-likelihood loss である。training data の prompt と high-quality response に対して、モデル $p_{\btheta}$ が response $\yb$ を高い確率で出すようにする。

**記号の定義**:
- $L_{\mathrm{SFT}}(\btheta)$ ... SFT で最小化する loss（main.tex, Eq. `eq:sft`）
- $\xb$ ... prompt sequence
- $\yb$ ... response sequence
- $q(\cdot)$ ... prompts の分布
- $p_{\mathrm{data}}(\cdot|\xb)$ ... training data の high-quality responses の条件付き分布
- $p_{\btheta}(\yb|\xb)$ ... LLM policy が $\xb$ に対して $\yb$ を生成する条件付き確率

**この論文での役割**: `SPIN` は SFT の後段に置かれる。SFT loss の最小化だけでは、同じ SFT dataset での追加訓練が頭打ちまたは劣化するため、著者は別の目的関数で SFT data を再利用する。

$$
f_{t+1}
= \argmin_{f \in \cF_{t}}
\EE\big[
\ell\big(f(\xb, \yb) - f(\xb, \yb')\big)
\big].
$$

**式の意味**: main player $f_{t+1}$ を、target response $\yb$ と旧モデル response $\yb'$ の差を大きくするように訓練する式である。期待値は $\xb\sim q(\cdot)$, $\yb\sim p_{\mathrm{data}}(\cdot|\xb)$, $\yb'\sim p_{\btheta_t}(\cdot|\xb)$ に関して取る（main.tex, Eq. `eq:f-star`）。

**記号の定義**:
- $f_{t+1}$ ... iteration $t+1$ の main player、response が target data 側らしいほど高い値を出す関数
- $\cF_t$ ... $p_{\btheta_t}$ に依存する function class
- $\yb'$ ... 旧 LLM $p_{\btheta_t}$ が生成した synthetic response
- $\ell(\cdot)$ ... monotonically decreasing and convex な loss。実装・理論説明では logistic loss $\ell(t)=\log(1+\exp(-t))$ を用いる

**この論文での役割**: `SPIN` の出発点である IPM 的な識別問題を表す。linear loss では objective が unbounded になるため、著者は non-negativity, smoothness, exponentially decaying tail を理由に logistic loss を選ぶ。

$$
L_{\mathrm{SPIN}}
= \EE\bigg[
\ell\bigg(
\lambda \log \frac{p_{\btheta}(\yb | \xb)}{p_{\btheta_t}(\yb | \xb)}
-\lambda \log \frac{p_{\btheta}(\yb' | \xb)}{p_{\btheta_t}(\yb' | \xb)}
\bigg)
\bigg].
$$

**式の意味**: `SPIN` の end-to-end training objective である。新しい候補モデル $p_{\btheta}$ が、旧モデル $p_{\btheta_t}$ と比べて target response $\yb$ の確率を上げ、旧モデル response $\yb'$ の確率を下げるほど loss が小さくなる（main.tex, Eq. `eq:loss`）。

**記号の定義**:
- $L_{\mathrm{SPIN}}$ ... TeX では `L_{\method}` と書かれる `SPIN` loss
- $p_{\btheta_t}$ ... previous iteration の opponent player
- $p_{\btheta}$ ... current iteration で最適化する LLM policy
- $\lambda>0$ ... KL regularization と対応する regularization parameter。実装説明では $\beta$ として $0.1$、最後の iteration で $5.0$ が使われる
- $\log \frac{p_{\btheta}(\yb|\xb)}{p_{\btheta_t}(\yb|\xb)}$ ... target response に対する、新旧モデルの log probability ratio

**この論文での役割**: Algorithm 1 の update step は、この loss の有限サンプル版を最小化する。DPO と形が似るのは logistic loss を選んだ場合だけで、著者は DPO との違いとして iterative procedure、SFT pairs のみを使う点、loss class の自由度を挙げる。

$$
p_{\btheta_{t+1}}(\yb|\xb)
\propto
p_{\btheta_t}(\yb|\xb)
\bigg(
\frac{p_{\mathrm{data}}(\yb|\xb)}{p_{\btheta_t}(\yb|\xb)}
\bigg)^{1/\lambda}.
$$

**式の意味**: logistic loss のとき、global minimum にある次 iteration の opponent player がどういう分布になるかを特徴づける式である（Theorem 2）。旧モデルが $p_{\mathrm{data}}$ より過小評価していた response は確率が上がり、過大評価していた response は下がる。

**記号の定義**:
- $p_{\btheta_{t+1}}(\yb|\xb)$ ... 次 iteration の LLM policy
- $p_{\btheta_t}(\yb|\xb)$ ... 現在の opponent player
- $p_{\mathrm{data}}(\yb|\xb)$ ... target data distribution
- $\propto$ ... 正規化定数を除いて比例すること
- $1/\lambda$ ... 更新の大きさを制御する指数。Theorem 2 の remark では、小さい $\lambda$ は大きな変化、大きい $\lambda$ は小さな変化をもたらすと説明される

**この論文での役割**: 理論上、`SPIN` が $p_{\btheta}$ を $p_{\mathrm{data}}$ へ近づける方向に働くことを示す中心的な根拠である。著者は、収束に近づくほど $\lambda$ を大きくすると training stability が高まるという解釈を与え、実装でも last iteration で $\beta=5.0$ に上げる。

### 実装 / アルゴリズム上の要点

- Algorithm 1 の入力は SFT Dataset $\{(\xb_i,\yb_i)\}_{i\in[N]}$, 初期 LLM $p_{\btheta_0}$, iteration 数 $T$ である。
- 各 iteration で、全 prompt $\xb_i$ に対し旧モデルから $\yb_i'\sim p_{\btheta_t}(\cdot|\xb_i)$ を生成する。
- 次に $\sum_i \ell(\lambda\log\frac{p_{\btheta}(\yb_i|\xb_i)}{p_{\btheta_t}(\yb_i|\xb_i)}-\lambda\log\frac{p_{\btheta}(\yb_i'|\xb_i)}{p_{\btheta_t}(\yb_i'|\xb_i)})$ を最小化して $\btheta_{t+1}$ を得る。
- 実験では Ultrachat200k から 50k prompts を random sample し、iteration 0 は synthetic dataset size 50k、iteration 1, 2, 3 は「most recent iteration の synthetic data」と新規生成分を合わせて 100k にする。各 iteration は 2 epochs。
- 実装は Alignment Handbook を codebase とし、DeepSpeed ZeRO-3 と FlashAttention-2 を使う。optimizer は RMSProp、weight decay なし、global batch size 64、10% warmup、bfloat16 precision。
- peak learning rate は iterations 0, 1 で 5e-7、iterations 2, 3 で 1e-7。$\beta=0.1$、max sequence length 2048 tokens、last iteration (`iter-3`) では $\beta=5.0$。
- synthetic data generation は Accelerate による distributed inference、global batch size 64。prompting template は `### Instruction: {prompt}\n\n### Response: `。Ultrachat200k の multi-round conversations からは first round だけを prompt / ground truth completion pair として sample する。

## 実験・結果

- **データセット / ベンチマーク**: base model は `zephyr-7b-sft-full`。これは Mistral-7B を Ultrachat200k で SFT した checkpoint と説明される。評価は HuggingFace Open LLM Leaderboard、MT-Bench、Big-Bench-Hard の Causal Judgment / Formal Fallacies / Sports Understanding、OpenBookQA。
- **比較対象 / baseline**: `zephyr-7b-sft-full`、Ultrachat200k で 1 epoch 追加 SFT したモデル、DPO checkpoint（本文では `zephyr-7b-beta`、表では `zephyr-7b-dpo-full` と記載）、および `SPIN iteration 3 + DPO`。
- **指標**: Open LLM Leaderboard は Arc `acc_norm` 25-shot、TruthfulQA `mc2` 0-shot、Winogrande `acc` 5-shot、GSM8k `acc` 5-shot、HellaSwag `acc_norm` 10-shot、MMLU `acc` 5-shot（Table `tab:open-llm-info`）。MT-Bench は average score、Big-Bench-Hard は standard few-shot CoT evaluation の accuracy、OpenBookQA は 1-shot の `acc_norm`。
- **主な結果**: Open LLM Leaderboard 平均は `zephyr-7b-sft-full` 58.14 から、`SPIN iteration 0` 60.80、iteration 1 62.12、iteration 2 62.97、iteration 3 63.16 へ改善した（Table `tab:main`）。個別には Arc 60.41 -> 65.87、TruthfulQA 43.73 -> 54.90、GSM8k 26.76 -> 38.97、HellaSwag 82.85 -> 85.54 が上がる。一方、Winogrande は 74.19 -> 73.72、MMLU は 60.92 -> 59.99 で、全タスクが単調改善するわけではない。
- **DPO との比較**: `zephyr-7b-dpo-full` は Arc 63.65、TruthfulQA 55.19、Winogrande 72.61、GSM8k 33.43、HellaSwag 84.44、MMLU 58.52、平均 61.31。`SPIN iteration 1` は平均 62.12 でこれを上回る。DPO は UltraFeedback Binarized の約 62k preference data を使い、chosen / rejected completions は GPT-4 で評価されたと説明される。
- **追加 SFT との比較**: `zephyr-7b-sft-full` を Ultrachat200k でさらに 1 epoch fine-tune すると、平均は 58.14 から 57.23 に下がる。Arc 57.76、TruthfulQA 44.39、Winogrande 75.77、GSM8k 25.85、HellaSwag 81.69、MMLU 57.89（Table `tab:ablation_data`）。
- **Ablation**: Figure `fig:training_size` では `SPIN` の training data size を 14k, 26k, 50k と変え、larger dataset contains the smaller dataset という設定で比較する。本文は training size を増やすと `SPIN` が改善する一方、Ultrachat200k での SFT epoch 2/3 は 1% を超える改善を出せないと述べる。Figure `fig:ablation_epoch` では、iteration 0 の training epochs を増やしても iteration 1 の性能には届かず、iterative training が必要だと主張する。
- **SPIN + DPO**: `SPIN iteration 3` から 62k UltraFeedback Binarized で DPO を 2 epochs 追加すると、平均 64.05 になる。個別値は Arc 66.47、TruthfulQA 60.07、Winogrande 78.06、GSM8k 37.98、HellaSwag 86.17、MMLU 59.68（Table `tab:additionspindpo`）。
- **追加ベンチマーク**: MT-Bench は 5.94 -> 6.46 -> 6.65 -> 6.78。Big-Bench-Hard / OpenBookQA は BB-causal 56.15 -> 59.36、BB-formal 49.6 -> 51.2、BB-sports 96.0 -> 94.4、OpenBookQA 45.4 -> 47.6（Table `tab:ablation_bbh`）。著者は no significant degradation と述べるが、BB-sports は下がっており、BB-formal も iteration 0 以降は単調ではない。
- **著者が主張する貢献**: 追加 preference data や reward model なしで SFT 済み LLM を改善する fine-tuning 方法を提案したこと、Theorem 1/2 で停止点と更新の性質を示したこと、50k SFT prompts だけで 62k preference data を使う DPO と同等以上の結果を得たこと、`SPIN` を DPO の前段として積み重ねられること。
- **計算コスト**: 8xA100 (80G) GPUs で、各 iteration の generation time は 1.45h。training time は iter 0 が 4.32h、iter 1-3 が各 8.64h。per 64 examples では generation 6.69s、training 10s と記載される（Table `tab:times`）。

## 妥当性と限界

- **この主張を支える根拠**: 同じ SFT checkpoint から、追加 SFT、DPO、`SPIN` を比較している点が重要である。Table `tab:ablation_data` は「同じ SFT dataset をもう一度 SFT するだけ」では改善しないことを示し、Table `tab:main` は `SPIN` が iteration ごとに平均を伸ばすことを示す。Theorem 1 は、単調減少・凸な $\ell$ の下で $p_{\btheta_t}=p_{\mathrm{data}}$ なら $\btheta_t$ が global minimum であり、$p_{\btheta_t}\neq p_{\mathrm{data}}$ なら適切な $\lambda$ に対して $\btheta_t$ は global minimum ではない、と述べる。
- **著者が認めている limitations / future work**: Conclusion は fixed target data distribution generated by humans を前提にするため、fine-tuned LLM の performance ceiling が $p_{\mathrm{data}}$ によって imposed されると明記する。future work として dynamically changing target data distribution と required synthetic data volume の削減を挙げる。
- **読者として注意すべき点**: 理論結果は global minimum や LLM space に関する仮定を含むため、有限データ・有限計算で実際に到達する保証とは分けて読む必要がある。実験で大きく伸びるのは TruthfulQA, GSM8k, Arc などで、Winogrande と MMLU は最終的に base より低い。さらに、本文は SFT dataset を human-annotated / high-quality responses として扱う一方、Ultrachat200k の出自は OpenAI's Turbo APIs による corpus 由来と説明している。
- **追加で確認したい実験 / 疑問**: Mistral-7B / Zephyr 系以外の backbone で同じ傾向が出るか、sampling temperature など synthetic generation の詳細が性能にどう効くか、出力多様性や calibration がどう変わるかは、この TeX 中には明示的な評価がない。DPO 比較も Zephyr 系 checkpoint が中心なので、同一 compute budget での他の iterative preference optimization との横比較は追加で見たい。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **`SPIN` / Self-Play fIne-tuNing**: SFT 済み LLM が過去 iteration の自分の response を生成し、それを target response と区別するように現在の LLM を fine-tune する手法。
- **main player**: iteration $t+1$ で学ぶ識別側。関数 $f_{t+1}$ として導入され、最終的には $p_{\btheta_{t+1}}$ の log probability ratio で表される。
- **opponent player**: previous iteration の LLM $p_{\btheta_t}$。SFT prompts に対して synthetic response $\yb'$ を生成する。
- **$p_{\mathrm{data}}(\cdot|\xb)$**: prompt $\xb$ に対する target data distribution。SFT dataset の high-quality responses の分布として使われる。
- **$\yb'$**: 旧 LLM が生成した synthetic response。preference label の rejected response そのものではなく、distribution-level に target data と区別される相手として使われる。
- **IPM (Integral Probability Metric)**: $p_{\mathrm{data}}$ と $p_{\btheta_t}$ の expected value gap を function class 上で測る考え方。Eq. `eq:f*1` / `eq:f-star` の動機。
- **logistic loss**: $\ell(t)=\log(1+\exp(-t))$。`SPIN` では non-negativity, smoothness, exponentially decaying tail を理由に採用される。
- **DPO**: Direct Preference Optimization。本文では Bradley-Terry model と preference dataset $(\xb,\yb_w,\yb_l)$ に基づく手法として `SPIN` と対比される。
- **$\lambda$ / $\beta$**: KL regularization と対応する更新の強さのパラメータ。理論式では $\lambda$、実装説明では $\beta$ として現れ、通常 0.1、last iteration では 5.0。
- **Ultrachat200k**: `zephyr-7b-sft-full` の SFT dataset であり、`SPIN` の prompts / target completions の供給源。実験ではこの中から 50k prompts を random sample する。
- **UltraFeedback Binarized**: DPO 比較と `SPIN + DPO` で使う約 62k preference data。chosen / rejected completions は GPT-4 で評価されたと説明される。
- **iteration**: `SPIN` の反復単位。実験の iteration 0 は base model 生成を使った最初の `SPIN` checkpoint、iteration 1 以降は直前 checkpoint が synthetic response を生成する。

## 読む順番の提案

- まず Abstract と Introduction を読み、中心問い `without acquiring additional human annotated data` と、`zephyr-7b-sft-full` で 58.14 -> 63.16 という主張の位置づけを確認する。正規ノートでは `Summary（著者の主張）` の前半につながる。
- 次に Problem Setting の SFT loss Eq. `eq:sft` と RL fine-tuning の KL-regularized objective を読む。ここで $q(\cdot)$, $p_{\mathrm{data}}$, $p_{\btheta}$ の意味を固めると、正規ノートの数式メモが読みやすい。
- Method では Figure `fig:observation`、Eq. `eq:f-star`、function class Eq. `eq:function class0`、end-to-end loss Eq. `eq:loss`、Algorithm 1 を順に追う。DPO との違いは Section `Comparison between SPIN and DPO` の enumerate 3 点を読む。
- Theoretical Analysis は Assumption 1、Theorem 1、Theorem 2 とその remarks を先に読む。証明は Appendix `Proof of Theorems` で、必要なら Lemma `thm:closed-form solution` まで確認する。
- Experiments は本文の Experiment Setup と Figure `fig:average`, `fig:dpo`, `fig:ablation_epoch`, `fig:training_size` を見たあと、Appendix の Table `tab:main`, `tab:ablation_data`, `tab:additionspindpo`, `tab:ablation_bbh`, `tab:times` で数値を確認する。正規ノートでは `Critical Thoughts` と `Notes / Quotes` の実験値確認に対応する。
- 最後に Conclusion and Discussion の limitation を読む。特に fixed target data distribution が performance ceiling になる、という著者自身の制限が `妥当性と限界` を読む上で重要である。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2401.01335v3/`
- 正規ノート: `notes/arXiv-2401.01335v3.md`
