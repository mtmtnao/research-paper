# Self-Consistency Improves Chain of Thought Reasoning in Language Models（Chain-of-Thought 推論のデコーディング手法）

- arXiv: https://arxiv.org/abs/2203.11171
- 一次ソース: ../papers/arXiv-2203.11171v4/
- 正規ノート: ../notes/arXiv-2203.11171v4.md

---

## 一言で言うと

Chain-of-Thought (CoT) prompting の greedy decoding を、複数の reasoning paths をサンプリングして最終回答を多数決する **self-consistency** に置き換える論文。固定された答え集合をもつ arithmetic / commonsense / symbolic reasoning タスクで、追加学習なしに CoT-prompting を大きく改善し、一部の symbolic setting では同値になると示す。

## 何を議論する論文か

- **問題設定**: CoT prompting は多段推論を改善するが、従来の greedy decoding は 1 本の reasoning path だけを採用する。論文は、greedy decoding の "repetitiveness and local-optimality" を避けて「naive greedy decoding」を置き換える問題として扱う（abstract, Section 1）。
- **対象範囲 / 仮定**: 最終回答が固定された集合 $\mathbb{A}$ に属し、生成出力から `The answer is X.` のような形式で $\mathbf{a}_i$ を parse できるタスクが中心である。著者は Section 2 末尾で、self-consistency は「final answer is from a fixed answer set」の問題に適用できると明記している。
- **既存研究との差分**: Cobbe et al. の verifier や Thoppilan et al. の re-ranker は追加の教師信号・人手アノテーション・訓練を必要とするが、self-consistency は事前学習済み LM に対する prompting と decoding の変更だけで動く。通常の model ensemble とも異なり、単一モデル上の "self-ensemble" と説明される（Section 1, Section 3.3）。
- **この論文で答えたい問い**: 多様な reasoning paths をサンプリングし、reasoning path を周辺化して最終回答の一致度で選ぶと、greedy CoT や sample-and-rank / beam search / prompt ensemble より reasoning accuracy が改善するか。

## 背景と前提

- **Chain-of-Thought prompting**: LM に最終回答だけでなく、途中の短い推論文を生成させる prompting。Wei et al. 2022 を直接の土台にしており、実験ではこの CoT-prompting + greedy decoding を主 baseline とする。
- **Decoding の違い**: この論文での greedy decoding は CoT prompting で単一の reasoning path を取る decoding として扱われる。一方 self-consistency は temperature sampling, top-$k$ sampling, nucleus sampling などで複数出力を作る（Section 2）。
- **Reasoning path と final answer**: 出力は $(\mathbf{r}_i,\mathbf{a}_i)$ として扱われる。$\mathbf{r}_i$ は reasoning path、$\mathbf{a}_i$ は最終回答であり、$\mathbf{r}_i$ は $\mathbf{a}_i$ に到達するための optional な中間系列として導入される（Section 2）。
- **Baseline との関係**: 比較対象は CoT-prompting, standard-prompting, sample-and-rank, beam search, prompt order permutation, multiple prompt sets, multiple-model ensemble。Previous SoTA には GSM8K の GPT-3 finetuned / verifier など、タスク別訓練を使う手法も含まれる（Table `tab:sota`, `tab:commonsense`）。
- **評価モデル**: UL2-20B, GPT-3 175B の `code-davinci-001` / `code-davinci-002`, LaMDA-137B, PaLM-540B。UL2 と GPT-3 は public model / public API、LaMDA-137B と PaLM-540B は not publicly available と Reproducibility Statement に書かれている。

## 提案手法

### コアアイデア

Self-consistency は、複雑な reasoning task では「正しい reasoning processes は多様でも最終回答では一致しやすく、誤った processes は同じ答えに集まりにくい」という仮説に基づく。論文中の短い根拠フレーズは "correct reasoning processes, even if they are diverse, tend to have greater agreement in their final answer than incorrect processes" である（Section 2）。

手順は Figure `fig:overview` の caption に沿うと 3 段階である。まず CoT exemplars で LM を prompt する。次に greedy decode ではなく decoder から複数の diverse reasoning paths を sample する。最後に sampled reasoning paths を marginalize out し、最終回答集合の中で最も consistent な答えを選ぶ。

### 重要な定義・数式

$$
\mathbf{a}_i \in \mathbb{A}, \quad i=1,\ldots,m, \quad (\mathbf{r}_i,\mathbf{a}_i), \quad \mathbf{r}_i \rightarrow \mathbf{a}_i
$$

**式の意味**: 生成された $m$ 個の候補出力それぞれを、reasoning path $\mathbf{r}_i$ と final answer $\mathbf{a}_i$ の組として見る。$\mathbf{a}_i$ は固定回答集合 $\mathbb{A}$ に属し、$\mathbf{r}_i$ は最終回答 $\mathbf{a}_i$ へ到達するための中間系列である。

**記号の定義**:
- $\mathbf{a}_i$ ... $i$ 番目の生成出力から parse された最終回答
- $\mathbb{A}$ ... 固定された final answer set
- $m$ ... decoder からサンプリングする candidate outputs の数
- $\mathbf{r}_i$ ... $i$ 番目の出力に含まれる reasoning path、すなわち token sequence
- $\mathbf{r}_i \rightarrow \mathbf{a}_i$ ... reasoning path が final answer に到達するために使われるという論文中の関係表現

**この論文での役割**: Self-consistency が「生成文全体」ではなく、reasoning path と final answer を分けて扱うための基本定義である。後続の多数決では $\mathbf{r}_i$ を捨て、$\mathbf{a}_i$ の一致だけを見る。

$$
\argmax_a\sum\nolimits_{i=1}^m \mathbbm{1}(\mathbf{a}_i = a)
$$

**式の意味**: $m$ 個の生成結果に含まれる final answer $\mathbf{a}_i$ のうち、同じ答え $a$ が何回出たかを数え、最多の答えを選ぶ。TeX 本文ではこれを majority vote、または "most consistent" answer と呼ぶ（Section 2）。

**記号の定義**:
- $a$ ... 集約後の候補回答
- $\mathbf{a}_i$ ... $i$ 番目の sampled output の final answer
- $m$ ... sampled candidate outputs の数
- $\mathbbm{1}(\mathbf{a}_i = a)$ ... $\mathbf{a}_i$ が $a$ と一致すれば 1、そうでなければ 0 になる indicator
- $\argmax_a$ ... 合計票数を最大化する $a$ を選ぶ操作

**この論文での役割**: 提案手法の中心である。Table `tab:aggregation` では、この unweighted sum / majority vote が PaLM-540B で GSM8K 74.4, MultiArith 99.3, AQuA 48.3, SVAMP 86.6, CSQA 80.7, ARC-c 88.7 を出し、normalized weighted sum とほぼ同等であることを示す。

$$
P(\mathbf{r}_i, \mathbf{a}_i \mid \text{prompt}, \text{question}) = \exp^{\frac{1}{K}\sum_{k=1}^{K}{\log P(t_k \mid \text{prompt}, \text{question}, t_1, \ldots, t_{k-1})}}
$$

**式の意味**: 生成された reasoning path と final answer の組 $(\mathbf{r}_i,\mathbf{a}_i)$ に対する条件付き生成確率を、出力長 $K$ で正規化して計算する式である。TeX では Eq. `eq2` として提示され、weighted aggregation の比較に使われる。

**記号の定義**:
- $P(\mathbf{r}_i, \mathbf{a}_i \mid \text{prompt}, \text{question})$ ... prompt と question が与えられたときに、その reasoning path と answer の組を生成する確率
- $t_k$ ... $(\mathbf{r}_i,\mathbf{a}_i)$ 内の $k$ 番目の token
- $K$ ... $(\mathbf{r}_i,\mathbf{a}_i)$ の token 数
- $\log P(t_k \mid \text{prompt}, \text{question}, t_1,\ldots,t_{k-1})$ ... 直前までの token に条件づけた $k$ 番目 token の log probability
- $\exp$ ... 平均 log probability を確率スケールへ戻す操作

**この論文での役割**: Majority vote ではなく、生成確率で候補を重み付けする aggregation の基準として使われる。Table `tab:aggregation` では normalized weighted sum が GSM8K 74.1 で majority vote 74.4 と近い一方、weighted avg は特に normalized で GSM8K 22.1 と大きく悪化する。著者は、LM が各生成を "similarly likely" と見なしており、"not well calibrated" で正誤を十分区別できないことを理由として述べる。

### 実装 / アルゴリズム上の要点

- step1: Wei et al. 2022 と同じ CoT exemplars を prompt として使う。Arithmetic reasoning では 8 個の manually written exemplars、commonsense reasoning では training set から 4-7 exemplars を選び、manually composed chain-of-thought prompts を使う（Section 3.1）。
- step2: decoder から $m$ 個の candidate outputs を独立にサンプリングする。Main Results では 10 runs 平均で、各 run につき 40 outputs を sample する（Section 3.2）。
- step3: 各出力を reasoning path $\mathbf{r}_i$ と answer $\mathbf{a}_i$ に分ける。Arithmetic reasoning では `The answer is` の後の first numerical part、commonsense reasoning では `The answer is` の後の full string answer を parse する（Section 2 footnote）。
- step4: $\mathbf{a}_i$ に majority vote を取り、最多の final answer を採用する。
- sampling 設定は、UL2-20B と LaMDA-137B で $T=0.5, k=40$、PaLM-540B で $T=0.7, k=40$、GPT-3 で $T=0.7$ かつ top-$k$ truncation なし（Section 3.1）。
- GPT-3 models では 128 max tokens、frequency penalty / presence penalty なし。全モデルで次の `Q:` の開始までを生成出力として扱い、final answer を parse する（Appendix, Details on Resources and Inference）。

## 実験・結果

- **データセット / ベンチマーク**: Arithmetic reasoning は MAWPS 系の AddSub, MultiArith, ASDiv、AQUA-RAT, GSM8K, SVAMP。Commonsense reasoning は CommonsenseQA, StrategyQA, ARC。Symbolic reasoning は last letter concatenation と Coinflip。追加で common NLP tasks として BoolQ, HotpotQA, e-SNLI, ANLI, RTE を扱う（Section 3.1, 3.2）。
- **比較対象 / baseline**: CoT-prompting with greedy decoding が主 baseline。さらに standard-prompting, sample-and-rank, beam search, prompt order permutation, multiple sets of prompts, multiple-model ensemble, Previous SoTA と比較する。
- **指標**: ほとんどは accuracy。HotpotQA は EM/F1。Self-consistency の main results は 10 runs 平均で、Table `tab:sota` の標準偏差は全 task で $\leq 0.5$ のため省略されている。
- **主な結果**: Arithmetic reasoning の Table `tab:sota` では、PaLM-540B が GSM8K 56.5→74.4 (+17.9), SVAMP 79.0→86.6 (+7.6), AQuA 35.8→48.3 (+12.5), ASDiv 74.0→81.9 (+7.9), MultiArith 94.7→99.3 (+4.6), AddSub 91.9→93.7 (+1.8)。GPT-3 `code-davinci-002` では GSM8K 60.1→78.0 (+17.9), SVAMP 75.8→86.8 (+11.0), AQuA 39.8→52.0 (+12.2), ASDiv 80.1→87.8 (+7.6), MultiArith 96.2→100.0 (+3.8)。
- **主な結果**: Commonsense / symbolic reasoning の Table `tab:commonsense` では、PaLM-540B が CSQA 79.0→80.7, StrategyQA 75.3→81.6, ARC-e 95.3→96.4, ARC-c 85.2→88.7, Letter (4) 65.8→70.8, Coinflip (4) 88.2→91.2。GPT-3 `code-davinci-002` は ARC-c 83.6→87.5, StrategyQA 73.4→79.8, Coinflip (4) 99.0→99.5 など。
- **主な結果**: CoT が standard-prompting より悪くなる NLP tasks でも self-consistency は上回る。Table `tab:common_nlp` では PaLM-540B で ANLI R1/R2/R3 が 69.1/55.8/55.8（standard）および 68.8/58.9/60.6（CoT）に対し self-consistency 78.5/64.5/63.4。e-SNLI は 85.8 / 81.0 / 88.4、RTE は 84.8 / 79.1 / 86.3、BoolQ は 71.3 / 74.2 / 78.4、HotpotQA は 27.1/36.8 / 28.9/39.8 / 33.8/44.6。
- **主な結果**: Beam search との比較では、UL2-20B AQuA で beam search top beam が beam size 1/5/10/20/40 に対し 23.6/19.3/16.1/15.0/10.2、一方 self-consistency using sampling は 19.7/24.9/25.3/26.7/26.9。MultiArith でも sampling self-consistency は 40 paths で 14.7、beam search top beam は 10.5（Table `tab:beam_search`）。
- **主な結果**: Ensemble-based approaches との比較では、LaMDA-137B GSM8K で CoT 17.1、3 prompt sets 18.6、40 prompt permutations 19.2、self-consistency 27.7。MultiArith では 51.8 / 57.1 / 60.9 / 75.7、SVAMP では 38.9 / 42.1 / 42.7 / 53.3（Table `tab:ensemble`）。
- **主な結果**: Robustness 実験では、LaMDA-137B GSM8K で correct CoT prompt 17.1、imperfect CoT prompt 14.9、imperfect + self-consistency 23.4。equation prompts は 5.0→6.5。PaLM-540B の zero-shot CoT は 43.0→69.2（Table `tab:robustness`）。
- **著者が主張する貢献**: 追加訓練なしの "sample-and-marginalize" decoding strategy、複数 reasoning benchmarks での大幅な accuracy gain、beam search / sample-and-rank / ensemble-based approaches への優位、consistency を uncertainty estimate として使える可能性（Figure `fig:error_rate`）。

## 妥当性と限界

- **この主張を支える根拠**: 4 つの language models with varying scales、arithmetic / commonsense / symbolic / common NLP tasks、CoT / standard prompting / decoding / ensemble 系 baseline との比較がある。特に Table `tab:sota` と `tab:commonsense` は多くの設定で CoT-prompting を上回り、Letter (4) の UL2-20B / LaMDA-137B など一部では同値になる。Table `tab:beam_search` と `tab:ensemble` は「多様性のある sampling + answer aggregation」が beam や prompt ensemble より有効であるという著者の解釈を支える。
- **著者が認めている limitations / future work**: Self-consistency は computation cost を増やす。Conclusion では実用上 5 or 10 paths から試すこと、将来は self-consistency でより良い supervised data を生成し、fine-tuning 後に single inference run で精度を上げることが future work として述べられる。
- **著者が認めている limitations / future work**: 固定 answer set をもつ問題に適用する手法であり、open-text generation へ拡張するには、複数回答が agree / contradict するかを測る consistency metric が必要とされる（Section 2）。
- **著者が認めている limitations / future work**: LM は nonsensical / non-factual reasoning paths を生成しうる。Conclusion では StrategyQA の例で population numbers が not exactly correct と述べ、rationale generations を better ground する必要があるとする。Ethics Statement でも nonsensical or non-factual reasoning paths への注意が明記される。
- **読者として注意すべき点**: Majority vote は最終回答の一致を使う手法であって、reasoning path の faithfulness を保証する手法ではない。Table `tab:example_path` や Appendix の例は「最終回答が正しくなる」ことを示すが、根拠文の factuality までは保証しない。
- **読者として注意すべき点**: "consistency is correlated with model's accuracy" は Figure `fig:error_rate` で GSM8K について示されるが、TeX 中に ECE や Brier score のような calibration 指標での比較は明示されていない。
- **追加で確認したい実験 / 疑問**: 同じ総 token budget / latency の下で、path 数、temperature、top-$k$ / nucleus をどう選ぶべきかは実用上重要である。TeX には path 数を 1, 5, 10, 20, 40 と変えた図と sampling robustness 図はあるが、コスト制約付きの詳細な pareto 分析は明示されていない。
- **追加で確認したい実験 / 疑問**: 高い consistency で誤答する問題の類型、自由生成タスクにおける consistency metric、rationale factuality を測る独立評価は、この論文の中心実験には含まれていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **self-consistency** ... CoT prompt は保ったまま、複数の reasoning paths を sample し、final answer の majority vote で最も consistent な答えを選ぶ decoding strategy。
- **reasoning path $\mathbf{r}_i$** ... final answer に到達するまでの中間 token sequence。論文では optional であり、最終的な aggregation では marginalize out される。
- **final answer $\mathbf{a}_i$** ... sampled output の末尾から parse される答え。Arithmetic では `The answer is` の後の first numerical part、commonsense では full string answer。
- **fixed answer set $\mathbb{A}$** ... self-consistency が前提にする最終回答集合。この論文の実験では、算術の数値回答、AQuA / ARC / CommonsenseQA の選択肢、StrategyQA / BoolQ の yes/no のように、final answer の一致を集計できる形式が扱われる。
- **CoT-prompting** ... Wei et al. 2022 の chain-of-thought prompting を使い、greedy decoding で 1 本の推論を生成する主 baseline。
- **greedy decoding** ... この論文では CoT-prompting の従来 decoding。複数 path を使わず、単一の最尤的な出力に依存する。
- **temperature sampling / top-$k$ sampling / nucleus sampling** ... diverse reasoning paths を得るための sampling algorithms。実験設定では $T=0.5$ または $0.7$、top-$k$ では $k=40$ が主に使われる。
- **majority vote / unweighted sum** ... $\mathbf{a}_i$ の票数だけで答えを選ぶ aggregation。Table `tab:aggregation` では normalized weighted sum とほぼ同等で、実装が単純な中心手法になっている。
- **normalized weighted sum** ... Eq. `eq2` の長さ正規化確率を使って回答を重み付けする aggregation。Table `tab:aggregation` では PaLM-540B で majority vote と近い値として比較される。
- **weighted avg** ... 各 answer の weighted sum を、その answer の票数で割る集約。Table `tab:aggregation` では特に normalized weighted avg が悪く、GSM8K 22.1 に落ちる。
- **sample-and-rank** ... 複数 sequence を sample し、log probability で top-ranked sequence を選ぶ方法。Figure `fig:gpt` では、同じサンプル数でも self-consistency の改善が大きいとされる。
- **self-ensemble** ... 複数モデルではなく単一 LM の複数 sampled outputs を集約するという意味で、著者が self-consistency を説明する表現。
- **consistency** ... 最終 aggregated answer に同意する decodes の割合。Figure `fig:error_rate` では GSM8K で accuracy と相関するとされ、uncertainty estimate としての可能性が述べられる。
- **OOD symbolic reasoning** ... prompt は 2-letters / 2-flips の例を含むが、test は 4-letters / 4-flips で行う設定。Table `tab:commonsense` の Letter (4), Coinflip (4) が該当する。

## 読む順番の提案

- まず Abstract と Figure `fig:overview` を読み、self-consistency が「CoT prompt」「sampling」「majority vote」の 3 手順であることを把握する。
- 次に Section 2 の $\mathbf{a}_i\in\mathbb{A}$、$(\mathbf{r}_i,\mathbf{a}_i)$、majority vote、Eq. `eq2` を読む。正規ノートの Summary / Takeaway にある「sample-and-marginalize」「重み付き集約と多数決が近い」という要点につながる。
- その後 Table `tab:aggregation` を見て、majority vote と normalized weighted sum の比較、および weighted avg が悪化する結果を確認する。
- 実験は Section 3.1 の setup を先に読み、モデル、prompt、sampling 設定、40 outputs × 10 runs を押さえてから、Table `tab:sota` と `tab:commonsense` を読む。
- 比較手法の妥当性を見るには Table `tab:beam_search`, Table `tab:ensemble`, Figure `fig:gpt` を読む。正規ノートの Critical Thoughts の「beam search は多様性が低い」「prompt ensemble より効く」という論点に対応する。
- 限界と実用上の注意は Conclusion and Discussion, Reproducibility Statement, Ethics Statement を読む。正規ノートの limitations / reproducibility / rationale faithfulness の議論に直結する。
- Appendix は、prompt robustness（Table `tab:robustness_prompt`）、multiple-model ensemble（Table `tab:ensemble_model`）、prompt 全文（Appendix `sec:appendix-prompt`）を確認したいときに読む。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2203.11171v4/`
- 正規ノート: `notes/arXiv-2203.11171v4.md`
