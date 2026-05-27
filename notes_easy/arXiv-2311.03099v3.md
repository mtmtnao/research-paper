# Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch（SFT delta parameter の冗長性と model merging）

- arXiv: https://arxiv.org/abs/2311.03099
- 一次ソース: ../papers/arXiv-2311.03099v3/
- 正規ノート: ../notes/arXiv-2311.03099v3.md

---

## 一言で言うと

同じ pre-trained backbone から作られた SFT language model では、fine-tuned parameter と pre-trained parameter の差である **delta parameters** が非常に冗長で、DARE により 90% から 99% を落としても性能を大きく損なわない場合がある、という論文である。著者はさらに、DARE で各 SFT model の delta を疎にしてから既存の model merging 手法に渡すと、parameter interference が軽減され、複数タスクの能力を 1 つの LM に統合しやすくなると主張している。

## 何を議論する論文か

- **問題設定**: SFT により task-specific ability を得た複数の LM を、追加データ・再学習・GPU なしで 1 つに統合したい。しかし、単純な parameter fusion では異なる task delta が干渉し、merged model の性能が落ちることがある。
- **対象範囲 / 仮定**: 中心仮定は「merge する SFT models が同じ pre-trained backbone から fine-tuned されている」こと（`section-3-methodology.tex`, "same pre-trained backbone"）。DARE の対象は fine-tuned parameter 全体ではなく、$\bm{\delta}^t=\bm{\theta}_{\text{SFT}}^t-\bm{\theta}_{\text{PRE}}$ で定義される delta parameters である。
- **既存研究との差分**: 論文は「多くの pruning methods は fine-tuned parameters を対象にする」と述べ、さらに pruning methods では retraining や extra data が必要になることが多い、と整理している。DARE は delta parameters だけを対象にし、random drop と rescale のみで retraining なしに使う。model merging については Average Merging, Task Arithmetic, Fisher Merging, RegMean, TIES-Merging などを置き換えるのではなく、それらの前段の plug-in として delta を疎にする。
- **この論文で答えたい問い**: SFT delta parameters はどの程度冗長か。random drop と $1/(1-p)$ rescale で元の embedding を近似できるか。DARE は model merging の parameter interference を減らすか。どのような delta の大きさ・backbone 条件で DARE は失敗するか。

## 背景と前提

- **SFT と delta parameters**: SFT は pre-trained LM を task-specific data で最適化し、特定タスク能力を引き出す手法として扱われている。論文は、SFT の効果を「SFT 前後の parameter の差」すなわち delta parameters に反映されるものとして分析する（`section-2-related-work.tex`, `section-3-methodology.tex`）。
- **homologous models**: 論文中では、同じ pre-trained backbone から fine-tuned された複数モデルを merge 対象にする。例えば Table 1 の WizardLM-13B, WizardMath-13B, llama-2-13b-code-alpaca はいずれも Llama-2-13b を backbone とするため、decoder-based merging の主実験に使われる。
- **model merging baseline**: Average Merging は parameters の平均、Task Arithmetic は delta に scaling term を掛けて足す、Fisher Merging は Fisher information による重み付き融合、RegMean は linear regression の closed-form solution、TIES-Merging は low-magnitude parameter の trim、sign disagreement の解消、consistent sign parameter の merge を行う（`section-2-related-work.tex`, `Appendix.tex`）。
- **pruning との違い**: Magnitude-based Pruning (MP) は absolute parameter values に基づいて除去対象を選ぶ。DARE は magnitude ではなく random drop を使い、delta parameter を落とした後に残りを $1/(1-p)$ で rescale する。
- **Dropout との違い**: DARE と Dropout は random dropping と rescaling を含む点は似ているが、DARE は delta parameters を推論用に恒久的に削る。一方 Dropout は training 中に model outputs を一時的に消す正則化として説明されている（`section-3-methodology.tex`）。

## 提案手法

### コアアイデア

DARE は **Drop And REscale** の略であり、SFT delta parameters のうち drop rate $p$ の割合をランダムに 0 にし、残った delta を $1/(1-p)$ 倍する。目的は、SFT によって得た task-specific ability を保ちながら delta の冗長部分を消し、複数モデルを merge するときの parameter interference を減らすことである。

著者の理論説明は linear transformation に限定される。FFN や self-attention の query/key/value/output projection のように、LM の多くの parameter が線形変換に関与するため、この設定で DARE 後の embedding expectation が元の embedding expectation と一致することを示す。著者自身はこれを "rough proof" と呼び、$p$ の上限を LM capacity から推定することは future direction として残している。

### 重要な定義・数式

$$
\bm{\delta}^t=\bm{\theta}_{\text{SFT}}^t - \bm{\theta}_{\text{PRE}} \in \mathbb{R}^d
$$

**式の意味**: task $t$ の SFT model が pre-trained model からどれだけ変化したかを delta parameters として定義する式である。`section-3-methodology.tex` の "SFT Delta Parameters" にある基本定義。

**記号の定義**:
- $\bm{\theta}_{\text{PRE}} \in \mathbb{R}^d$ は pre-trained LM の parameters
- $\bm{\theta}_{\text{SFT}}^t \in \mathbb{R}^d$ は task $t$ 用に SFT された LM の parameters
- $\bm{\delta}^t$ は SFT 前後の差分、すなわち task $t$ の delta parameters
- $d$ は parameter dimension

**この論文での役割**: DARE は fine-tuned parameter 全体ではなく、この $\bm{\delta}^t$ だけを落とす。後の失敗例でも、fine-tuned parameter 全体を落とすと性能が壊れることが強調される。

$$
\begin{gathered}
\bm{m}^t \sim \text{Bernoulli}(p), \\
\bm{\widetilde{\delta}}^t = \left(\bm{1} - \bm{m}^t\right) \odot \bm{\delta}^t, \\
\bm{\hat{\delta}}^t = \bm{\widetilde{\delta}}^t / (1 - p)
\end{gathered}
$$

**式の意味**: DARE の drop と rescale を表す式である。mask $\bm{m}^t$ により delta parameter の一部を 0 にし、残った delta を $1/(1-p)$ 倍する（`section-3-methodology.tex`, `equ:drop_rescale`）。

**記号の定義**:
- $\bm{m}^t$ は task $t$ の delta parameter に対する random mask
- $\text{Bernoulli}(p)$ は確率 $p$ で 1 を取る Bernoulli 分布
- $p$ は drop rate
- $\bm{1}$ は全成分が 1 の vector
- $\odot$ は element-wise product
- $\bm{\delta}^t$ は drop 対象の SFT delta parameters
- $\bm{\widetilde{\delta}}^t$ は drop 後、rescale 前の delta
- $\bm{\hat{\delta}}^t$ は DARE 後の rescaled delta

**この論文での役割**: 手法の中心そのものである。DARE model は最終的に $\bm{\theta}_{\text{DARE}}^t = \bm{\hat{\delta}}^t + \bm{\theta}_{\text{PRE}}$ として推論に使われる。

$$
\mathbb{E} [h_i]
= \sum_{j=1}^n w_{ij}x_j + b_i + \sum_{j=1}^n \Delta w_{ij}x_j + \Delta b_i
= h_i^{\text{PRE}} + \Delta h_i
$$

**式の意味**: pre-trained parameter と delta parameter の両方を含む元の SFT model について、線形変換後の embedding の $i$ 番目の成分の期待値を分解している。

**記号の定義**:
- $\bm{W}, \bm{b}$ は pre-trained parameter の weight matrix と bias
- $\Delta \bm{W}, \Delta \bm{b}$ は delta parameter の weight matrix と bias
- $w_{ij}, \Delta w_{ij}$ はそれぞれ $\bm{W}, \Delta \bm{W}$ の $i$ 行 $j$ 列の成分
- $b_i, \Delta b_i$ はそれぞれ $\bm{b}, \Delta \bm{b}$ の $i$ 番目の成分
- $\bm{x} \in \mathbb{R}^n$ は入力 vector
- $j=1,\ldots,n$ は入力 dimension に沿った添字
- $h_i^{\text{PRE}}$ は pre-trained 部分だけが作る $i$ 番目の embedding 成分
- $\Delta h_i$ は delta parameter が embedding に加える成分

**この論文での役割**: DARE 後の embedding expectation と比較する基準である。論文は「DARE が元の embeddings を近似する」ことを、この式と次の式の一致で説明する。

$$
\mathbb{E} [\hat{h}_i]
= h_i^{\text{PRE}} + (1 - p)\cdot \gamma \cdot \Delta h_i
$$

**式の意味**: DARE により delta を drop rate $p$ で落とし、残りを rescale factor $\gamma$ で増幅した場合の embedding expectation である。$\gamma = 1/(1-p)$ と置くと、$\mathbb{E}[h_i]=\mathbb{E}[\hat{h}_i]$ になる。

**記号の定義**:
- $\hat{h}_i$ は DARE 後の model が作る $i$ 番目の embedding 成分
- $h_i^{\text{PRE}}$ は pre-trained 部分だけが作る $i$ 番目の embedding 成分
- $p$ は delta parameter を落とす割合
- $\gamma$ は残った delta parameter の rescale factor
- $\Delta h_i$ は元の delta parameter が embedding に寄与する量

**この論文での役割**: rescale operation の理論的根拠である。実験節では DropOnly（rescale なし）と比較し、rescale が embedding cosine similarity と性能維持に必要であることを示す。

$$
\begin{split}
& \bm{\theta}_{\text{DARE}}^{t_k} =
\text{DARE}\left(\bm{\theta}_{\text{SFT}}^{t_k}, \bm{\theta}_{\text{PRE}}, p\right), \quad 1 \leq k \leq K, \\
\bm{\theta}_{\text{M}} =\ &
\bm{\theta}_{\text{PRE}} + \lambda \cdot \sum_{k=1}^K \bm{\hat{\delta}}^{t_k}
= \bm{\theta}_{\text{PRE}} + \lambda \cdot \sum_{k=1}^K
(\bm{\theta}_{\text{DARE}}^{t_k} - \bm{\theta}_{\text{PRE}})
\end{split}
$$

**式の意味**: Task Arithmetic に DARE を組み込むときの merged model parameter の計算である。各 SFT model に DARE を適用してから、その rescaled delta を足し合わせる。

**記号の定義**:
- $K$ は merge する task/model の数
- $t_k$ は $k$ 番目の task
- $\bm{\theta}_{\text{SFT}}^{t_k}$ は task $t_k$ 用に fine-tuned された source model の parameters
- $\bm{\theta}_{\text{PRE}}$ は共通の pre-trained backbone の parameters
- $\bm{\theta}_{\text{DARE}}^{t_k}$ は DARE 後に $\bm{\theta}_{\text{PRE}}$ と rescaled delta を足して得る parameters
- $\bm{\theta}_{\text{M}}$ は merged single model の parameters
- $\text{DARE}(\cdot)$ は `equ:drop_rescale` に従って delta を drop/rescale し、pre-trained parameters に足し戻す処理
- $p$ は DARE の drop rate
- $\lambda$ は merge 時の scaling term
- $\bm{\hat{\delta}}^{t_k}$ は task $t_k$ の DARE 後 delta

**この論文での役割**: DARE が standalone pruning 手法ではなく、Task Arithmetic など既存 merging 手法の前段 plug-in として使われることを示す。論文は同じ考え方を Average Merging, Fisher Merging, RegMean, TIES-Merging にも適用できると述べる。

### 実装 / アルゴリズム上の要点

- step1: 同じ pre-trained backbone を持つ SFT model を用意し、各 task $t_k$ について $\bm{\delta}^{t_k}$ を計算する。
- step2: 各 $\bm{\delta}^{t_k}$ に対して Bernoulli mask を作り、drop rate $p$ の割合で delta parameter を 0 にする。
- step3: 残った delta parameter を $1/(1-p)$ で rescale し、$\bm{\theta}_{\text{PRE}}$ に足して $\bm{\theta}_{\text{DARE}}^{t_k}$ を得る。
- step4: DARE 後の parameters または delta を、Task Arithmetic / TIES-Merging / Average Merging / Fisher Merging / RegMean に渡して merge する。
- step5: decoder-based LMs の主実験では、計算可能性のため Task Arithmetic と TIES-Merging を使う。scaling term は $[0.5, 1.0]$ から選び、TIES-Merging の retain ratio は $[0.5, 0.7, 0.9]$ から選ぶ（`section-4-experiments.tex`）。
- step6: encoder-based LMs では 5 種類の merging method すべてを使い、DARE の $p$ は $[0.1, 0.2, \cdots, 0.9]$ から選ぶ（`Appendix.tex`, `tab:plms_hyperparameter_searched_ranges_merging_methods`）。

## 実験・結果

- **データセット / ベンチマーク**: decoder-based LMs では AlpacaEval（instruction-following）、GSM8K と MATH（mathematical reasoning）、HumanEval と MBPP（code-generating）を使う。encoder-based LMs では GLUE の CoLA, SST-2, MRPC, QQP, STS-B, MNLI, QNLI, RTE を使う。Open LLM Leaderboard では ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K の 6 benchmark 平均を使う。
- **比較対象 / baseline**: SFT source models、DARE なしの merging、DropOnly、Magnitude-based Pruning (MP)、Average Merging、Task Arithmetic、Fisher Merging、RegMean、TIES-Merging、encoder-based の multi-task learning oracle。
- **指標**: AlpacaEval は win rate、GSM8K と MATH は zero-shot accuracy、HumanEval と MBPP は pass@1、CoLA は Matthews correlation coefficient、SST-2/QNLI/RTE は accuracy、MNLI は matched accuracy、MRPC/QQP は accuracy と F1、STS-B は Pearson と Spearman correlation。
- **評価設計の細部**: GLUE は test labels が公開されていないため、original training data を 90% / 10% に分け、original validation data を test set として使う。decoder-based LMs の inference は vLLM、temperature 0.0、max generated tokens は GSM8K で 1,024、他の 4 datasets で 2,048。実験は NVIDIA Tesla V100 と A100 GPUs で行われるが、著者は DARE / merging 自体の利点を CPU のみ・再学習なしで得られると述べる。
- **主な結果**: SFT delta redundancy の実験では、$p \in [0.0, 0.1, 0.2, \cdots, 0.9, 0.99]$ を試す。著者は「DARE can effectively remove 90% delta parameters」と結論し、場合によって $p=0.99$ まで到達すると述べる。例として WizardMath-70B は $p=0.99$ でもよく動く一方、WizardMath-7B と WizardMath-13B は失敗すると書かれている（`section-4-experiments.tex`）。
- **主な結果**: Table 1 (`tab:llms_merging_all_results`) では、source models として WizardLM-13B が AlpacaEval 67.20、WizardMath-13B が GSM8K 64.22 / MATH 14.02、llama-2-13b-code-alpaca が HumanEval 23.78 / MBPP 27.60 を示す。Task Arithmetic + DARE の LM & Math & Code は AlpacaEval 69.28、GSM8K 56.48、MATH 10.16、HumanEval 23.17、MBPP 31.60。著者は相対改善として、LM & Math & Code vs. LM on AlpacaEval が 3.10%、LM & Math vs. Math on GSM8K が 3.18%、LM & Code vs. Code on MBPP が 19.57% と述べる。
- **主な結果**: TIES-Merging + DARE の LM & Math & Code は AlpacaEval 72.50、GSM8K 58.00、MATH 9.20、HumanEval 29.27、MBPP 31.40。著者は、TIES-Merging は first step で low-magnitude delta を落とすため性能を下げる可能性があるが、DARE 後は 0 が smallest magnitude になるのでその損失を避けやすい、と説明する。
- **主な結果**: encoder-based GLUE merging では、DARE による平均改善は Average Merging 0.58%、Task Arithmetic 0.36%、Fisher Merging 0.37%、RegMean -0.03%、TIES-Merging 0.84%。ただし merged model が single model を超えられない場合も残る。
- **主な結果**: 7B merged LMs では、Table 2 (`tab:llms_merging_open_llm_leaderboard_results`) で supermario_v1 が Average 74.85、supermario_v2 が Average 75.49 を記録する。同じ表では NeuralBeagle14-7B が 74.74、Beagle14-7B が 74.76、WildMarcoroni-7B が 75.29、WestSeverus-7B が 75.29 と報告される。ただし Appendix は、supermario_v1 の source は NeuralBeagle14-7B + Turdus であり、Turdus の Open LLM Leaderboard 結果がないため Beagle14-7B を代わりに表へ載せた、と説明している。著者は 2024-01-28 時点で supermario_v2 が Open LLM Leaderboard の 1 位だと述べる。Appendix では supermario_v1 が DARE $p=0.3$、Task Arithmetic scaling 0.8、supermario_v2 が WildMarcoroni-Variant1-7B + WestSeverus-7B-DPO-v2、DARE $p=0.5$、scaling 0.5 と説明される。
- **主な結果**: rescale operation の検証では、DARE は 90% delta を落としても各層 embedding の cosine similarity を 0.95 超に保つ。一方 DropOnly は $p=0.5/0.9$ の WizardMath-7B で similarity が約 0.85/0.68 まで下がる（`section-4-rescale-operation`）。
- **主な結果**: MP との比較では、DARE は多くの場合 MP を上回り、drop rate が大きいほど差が目立つとされる。MP に rescale を加えると、$p=0.7$ の 7B LMs で AlpacaEval 43.85 から 10.61、GSM8K 46.70 から 0.37、HumanEval 21.34 から 3.05 に悪化した。
- **主な結果**: DARE の使用条件として、WizardCoder-Python-13B を本来の CodeLlama-13b-Python ではなく Llama-2-13b に対する delta と見なすと、HumanEval/MBPP の pass@1 が 63.41/55.4 から 0.0/0.0 に落ちる。著者は Code Llama が code-related data 500B tokens で追加学習されているため、Llama 2 との差分が大きくなると説明する。
- **主な結果**: fine-tuned parameter 全体を drop すると、$p=0.1$ でも decoder-based LMs は大きく壊れる。WizardLM-13B は AlpacaEval 67.20 から 8.56、WizardMath-13B は GSM8K/MATH 64.22/14.02 から 0.38/0.16、WizardCoder-Python-13B は HumanEval/MBPP 63.41/55.40 から 0.0/0.20 へ落ちる。
- **著者が主張する貢献**: SFT delta parameters の極端な冗長性を encoder / decoder の両方で示したこと、DARE という data-free / retraining-free / GPU-free な delta sparsification を提案したこと、既存 model merging 手法の plug-in として有効性を示したこと、SFT は能力を新規導入するより pre-trained LM の能力を unlock するという見方を支持する実験を示したこと。

## 妥当性と限界

- **この主張を支える根拠**: 理論面では、linear transformation において $\gamma=1/(1-p)$ とすれば DARE 後の embedding expectation が元の embedding expectation と一致することを示す。実験面では、DropOnly との embedding cosine similarity 比較、MP との比較、fine-tuned parameter 全体を落とす失敗例、wrong backbone を使う失敗例、Table 1 (`tab:llms_merging_all_results`) / Table 2 (`tab:llms_merging_open_llm_leaderboard_results`) の merging 結果が主な根拠になる。
- **著者が認めている limitations / future work**: 著者は理論を "rough proof" とし、LM capacity に対する $p$ の upper bound 推定や、fine-tuned parameters と delta parameters の本質的差異の解明を future direction とする。encoder-based LMs では改善が小さく、merged model が single model を超えられない場合がある。また、source model が十分に fine-tuned されていることが effective model merging の prerequisite だと述べる。
- **読者として注意すべき点**: DARE は同じ backbone からの SFT delta が比較的小さい場合に成立する手法である。論文は、他の 13B decoder-based LMs では多くの absolute delta が 0.002 未満で DARE に適すると述べる一方、WizardCoder-Python-13B vs. Llama-2-13b では absolute delta が 0.01 を超えることが多く、DARE が失敗すると示す。Appendix の decile table でも、WizardCoder-Python-13B vs. Llama-2-13b は min/max が -2.40 / 2.40 と非常に大きい。
- **読者として注意すべき点**: Table 1 は「全タスクで常に best source を超える」ことを示す表ではない。例えば TIES-Merging + DARE の LM & Math & Code は AlpacaEval 72.50 で LM source 67.20 を超えるが、GSM8K 58.00 は Math source 64.22 より低い。著者の主張は、DARE が多くの場合 merging を助け、merged LM が source LM を超える可能性を示す、という範囲で読むべきである。
- **追加で確認したい実験 / 疑問**: DARE は random drop を含むが、TeX 中には主要 Table 1 / Table 2 に対する乱数 seed ごとの分散や信頼区間は明示されていない。drop rate $p$、Task Arithmetic の $\lambda$、TIES-Merging の retain ratio の grid search が各結果にどれだけ寄与したかも、TeX 中では分解されていない。
- **追加で確認したい実験 / 疑問**: Impact Statement では、merged LLM でも gender bias や racial discrimination のような harmful information が残り得るため regulation が必要だと述べられている。DARE による safety / alignment 能力の変化は、TeX 中の主要評価には含まれていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **SFT (Supervised Fine-Tuning)**: pre-trained LM を task-specific data で最適化し、instruction-following / mathematical reasoning / code-generating などの能力を引き出す操作。
- **delta parameters**: $\bm{\theta}_{\text{SFT}}^t-\bm{\theta}_{\text{PRE}}$。この論文で DARE が drop する対象であり、SFT の効果を parameter 空間で見るための中心概念。
- **DARE**: Drop And REscale。delta parameters を random drop し、残りを $1/(1-p)$ で rescale する手法。
- **drop rate $p$**: delta parameters を 0 にする割合。実験では $[0.0, 0.1, \cdots, 0.9, 0.99]$ などが使われる。
- **rescale operation**: DARE で残った delta を $1/(1-p)$ 倍する操作。linear transformation の期待値保存と DropOnly 比較の両方で重要になる。
- **DropOnly**: DARE から rescale を除いた ablation。embedding cosine similarity と性能が高い $p$ で大きく落ちる比較対象。
- **homologous models**: 同じ pre-trained backbone から fine-tuned された SFT models。model merging の中心仮定。
- **parameter interference**: 複数 SFT model の parameters / deltas を merge するとき、互いの task-specific な変化が衝突して性能を下げる問題。
- **Task Arithmetic**: pre-trained parameter に task delta の和を $\lambda$ 倍して足す merging method。DARE の plug-in 例として数式化される。
- **TIES-Merging**: low-magnitude parameter の trimming、sign disagreement の解消、consistent sign parameter の disjoint merge により task conflicts を扱う merging method。
- **Magnitude-based Pruning (MP)**: magnitude が小さい parameter を落とす pruning baseline。この論文では delta parameter に適用し、retraining process は捨てて比較する。
- **embedding cosine similarity**: DARE / DropOnly 後の LM と元の LM の各層 embedding がどれだけ近いかを測る指標。DARE が元の embeddings を近似する根拠として使われる。
- **Open LLM Leaderboard**: Eleuther AI Language Model Evaluation Harness に基づき、ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K の平均で open-sourced LLM を評価する leaderboard。
- **continuous pre-training**: Code Llama が 500B tokens の code-related data で追加学習された例として出る。SFT 以上に parameter 差分が大きくなり、DARE 失敗の背景として説明される。

## 読む順番の提案

- まず `main.tex` の abstract を読み、DARE が「delta parameter を drop + rescale する」「homologous SFT models の merging に使う」という全体像を掴む。
- 次に `section-3-methodology.tex` の "SFT Delta Parameters"、`equ:drop_rescale`、linear transformation の期待値計算、Task Arithmetic with DARE の式を読む。正規ノートの `Summary` と `Notes / Quotes` の中心式に対応する。
- その後、`section-4-experiments.tex` の Experimental Setup と Table 1 を読む。どの dataset / metric / source model / merging method の結果かを確認してから、正規ノートの `Summary` の decoder merging の箇条書きに戻ると数値が追いやすい。
- 続いて `section-4-rescale-operation`、`Comparison with Magnitude-based Pruning`、`When Can DARE Be Used?`、`Can DARE Drop Fine-tuned Parameters?` を読む。正規ノートの `Takeaway` と `Critical Thoughts` の「効く条件・効かない条件」に対応する。
- Appendix では `tab:llms_SFT_backbone_correspondences`、`tab:plms_hyperparameter_searched_ranges_merging_methods`、`section-appendix-details_7b_merged_model_leaderboard`、`tab:statistics_parameters_changed_ranges` を見る。source model と backbone の対応、grid search 範囲、supermario_v1/v2 の構成、delta range の根拠がここにある。
- 最後に `section-5-conclusion.tex` の Conclusion と Impact Statement を読み、著者がどこまでを contribution とし、どこに社会的リスクを認めているかを確認する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2311.03099v3/`
- 正規ノート: `notes/arXiv-2311.03099v3.md`
