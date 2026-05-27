# Transformer-Squared: Self-adaptive LLMs（self-adaptive LLM / SVF による PEFT / test-time adaptation）

- arXiv: https://arxiv.org/abs/2501.06252
- 一次ソース: ../papers/arXiv-2501.06252v3/
- 正規ノート: ../notes/arXiv-2501.06252v3.md

---

## 一言で言うと

この論文は、LLM を一度の大規模 post-training で全タスクに合わせるのではなく、重み行列の特異値だけを RL で調整した小さな `expert` ベクトルを作り、推論時に 2 パスで選択・混合する `Transformer^2` を提案する。著者は、`SVF` が少ない trainable parameters で LoRA より安定しやすく、未見タスクや VLM、Llama から Mistral への cross-model transfer にも使えると主張する（`sections/sec1_introduction.tex`, `sections/sec3_methods.tex`, Table 1-5）。

## 何を議論する論文か

- **問題設定**: 従来の LLM post-training は、広い能力を 1 回の fine-tuning にまとめようとするため、計算コスト・training time が大きく、データ範囲を広げると overfitting と task interference の trade-off が出る。LoRA などで expert modules を作る方向も、複数 expert の累積パラメータ、狭い task domain への overfitting、runtime での flexible composition が未解決だと導入部は整理している。
- **対象範囲 / 仮定**: 対象は、既存の pre-trained / instruction-tuned LLM の内部重みを大きく作り替えず、タスクごとの小さな調整ベクトルを追加する PEFT と test-time adaptation である。重要な仮定は、pre-trained model の重みには downstream task に使える latent capabilities がすでに含まれ、特異成分の寄与度を変えるだけでもタスク性能を動かせる、というもの。
- **既存研究との差分**: LoRA は低ランク行列を追加するが、`SVF` は元の重み行列の SVD に基づき singular values の scale だけを学習する。MoE と比べると、既存 MoE は token-level routing が中心であるのに対し、`Transformer^2` は sample-level module selection を行い、expert vectors は RL で domain-specific performance を直接最適化する。
- **この論文で答えたい問い**: `SVF` は、少数パラメータ・RL 学習・expert composition を同時に満たす building block になれるか。さらに、それらの expert vectors を推論時に選択・混合することで、MATH / Humaneval / ARC-Challenge / OKVQA のような未見タスクに適応できるか。

## 背景と前提

- **SVD と singular components**: 本文は各 weight matrix を $W = U\Sigma V^\intercal$ と分解し、各 rank-1 component $u_i v_i^\intercal$ が独立に入力を処理し、singular value $\sigma_i$ がその寄与度を調整すると説明する（`sections/sec3_methods.tex`）。
- **PEFT と LoRA**: PEFT は元モデル全体を更新せず、小さな追加パラメータで task-specific updates を得る枠組みである。LoRA は低ランク行列を導入する代表的手法だが、論文は expert を多数作った場合の storage / computation と、複数 LoRA を補間するときの permutation freedom を問題視する。
- **Self-adaptive LLMs**: 関連研究節では、self-adaptive LLM を「operating environment や internal state の変化に応じて、external intervention なしに挙動を評価・修正できる LLM」と定義する。論文は macroview（複数 LLM の協調）ではなく、single LLM 内部を変える microview に焦点を置く。
- **RL 目的関数**: `SVF` の expert は、next-token prediction ではなく REINFORCE と KL penalty で学習される。reward は生成答えの正誤に基づく $r \in \{-1, 1\}$ で、弱い base model では sparse rewards が caveat になると本文が述べる。
- **CEM**: Few-shot adaptation では Cross-Entropy Method を使って、複数 expert vectors の線形結合係数 $\alpha_k$ を探索する。本文の CEM 説明では、分布 $Q$ から samples を生成し、性能上位の elite samples で $Q$ の平均・標準偏差を更新する。

## 提案手法

### コアアイデア

`Transformer^2` は、`SVF` で作った domain-specific expert vectors を推論時に使う self-adaptation framework である。学習時には、GSM8K、MBPP-Pro、ARC-Easy などの訓練タスクごとに、対象 weight matrices の singular values をスケールする $z$ ベクトルを RL で学習する。推論時には、1 パス目で入力 prompt / task の性質を見て適切な $z^\prime$ を選ぶか構成し、2 パス目で $W^\prime = U\Sigma^\prime V^\intercal$ に置き換えたモデルから実回答を生成する。

この論文の主張で重要なのは、`SVF` が単なるパラメータ削減法ではなく、expert vectors の composition をしやすい表現として設計されている点である。LoRA の $A,B$ は同じ機能でも等価な parameter permutations を持ちうるため、複数 LoRA の補間が元の挙動を保つとは限らない。一方で `SVF` は元の $U,V$ を固定し、ordered singular values の scale を変えるため、$z$ 同士の線形補間を adaptation に使いやすい、というのが著者の論理である。

### 重要な定義・数式

$$
W = U \Sigma V^\intercal,\qquad
y=\sum_{i=1}^r \sigma_i u_i v_i^\intercal x
$$

**式の意味**: weight matrix $W$ を SVD で分解し、線形変換を singular components の和として見る式である。本文は各 $u_i v_i^\intercal$ が独立した寄与を持ち、$\sigma_i$ がその寄与の大きさを調整すると説明する。

**記号の定義**:
- $W$ ... neural network 内の weight matrix。本文では $W \in \mathbb{R}^{n \times m}$ と置く。
- $U, V$ ... SVD で得られる semi-orthogonal matrices。
- $\Sigma$ ... ordered singular values を対角に並べた行列。
- $\sigma_i$ ... $i$ 番目の singular value。
- $u_i, v_i$ ... $U,V$ の列ベクトルとして表される singular component。
- $r$ ... singular values の数。本文では $r=\min(m,n)$ として扱う。
- $x,y$ ... layer への入力と出力。

**この論文での役割**: `SVF` が「重み全体を自由に更新する」のではなく、「既存の singular components の寄与度を変える」方法であることを支える基礎式である。

$$
W^\prime=U \Sigma^\prime V^\intercal,\qquad
\Sigma^\prime=\Sigma \otimes \mathrm{diag}(z)
$$

**式の意味**: `SVF` の中核定義で、元の $U,V$ を固定したまま、学習した $z$ によって singular values を成分ごとにスケールし、新しい weight matrix $W^\prime$ を作る。

**記号の定義**:
- $W^\prime$ ... `SVF` によって変更された weight matrix。
- $\Sigma^\prime$ ... $z$ によりスケールされた singular value matrix。
- $z \in \mathbb{R}^r$ ... weight matrix ごとに学習される expert vector。
- $\mathrm{diag}(z)$ ... $z$ を対角成分に持つ行列。
- $\otimes$ ... TeX 本文で使われている演算子。ここでは $\Sigma$ の対角成分を $z$ の対応成分でスケールする役割を持つ。

**この論文での役割**: LoRA のような追加低ランク行列ではなく、full-rank な weight modification を $r=\min(m,n)$ 個の scale parameters だけで行う、という PEFT 設計を定義する。

$$
J(\theta_z) =
\mathbb{E}\left[
\log\left(\pi_{\theta_{W^\prime}}(\hat{y}_i \mid x_i)\right)
r(\hat{y}_i, y_i)
\right]
- \lambda D_\mathrm{KL}(\pi_{\theta_{W^\prime}} \| \pi_{\theta_W})
$$

**式の意味**: `SVF` expert vectors を REINFORCE で学習する目的関数である。正しい出力を出した確率を高め、誤った出力を出した確率を下げる方向に更新しつつ、元モデル $\pi_{\theta_W}$ から離れすぎないよう KL penalty を加える。

**記号の定義**:
- $\theta_z=\{z_1,\cdots,z_{N\times M}\}$ ... fine-tune する weight matrices に対応する `SVF` vectors の集合。
- $\theta_W=\{W_1,\cdots,W_{N\times M}\}$ ... 元の weight matrices の集合。
- $\pi_{\theta_W}$ ... 元の language model。
- $\pi_{\theta_{W^\prime}}$ ... $W$ を $W^\prime$ に置き換えた後の language model。
- $x_i$ ... prompt。
- $\hat{y}_i$ ... model が生成した答え。
- $y_i$ ... 正解。
- $r(\hat{y}_i,y_i)$ ... correctness に基づく unitary reward。本文では $r\in\{-1,1\}$。
- $\lambda$ ... KL divergence term の係数。付録 Table `hyper_p_app.tex` では $0.0,0.1,0.2,0.3$ を sweep する。

**この論文での役割**: `SVF` が next-token prediction ではなく task performance を直接最適化する、という主張の中心である。Table 4 では、`SVF` + Policy gradient が `SVF` + Next token pred や LoRA + Policy gradient より良い結果を示す。

$$
z^\prime=\sum_{k=1}^{K}\alpha_k z_k
$$

**式の意味**: Few-shot adaptation で、$K$ 個の learned `SVF` expert vectors を係数 $\alpha_k$ で線形結合し、target task 用の新しい $z^\prime$ を作る式である。

**記号の定義**:
- $K$ ... 事前に学習した expert tasks の数。
- $z_k$ ... $k$ 番目の training task で学習した expert vector。
- $\alpha_k$ ... CEM で探索される混合係数。
- $z^\prime$ ... 2 パス目の回答生成で使う adapted expert vector。

**この論文での役割**: `Transformer^2` の最も強い adaptation strategy である Few-shot adaptation を定義する。付録では main experiments で 10 samples を self-adaptation に確保し、最大 100 CEM iterations を行うと説明されている。

### 実装 / アルゴリズム上の要点

- step1: 対象 LLM の各 selected weight matrix $W$ に SVD を適用し、各 $W$ に対応する scale vector $z$ を用意する。著者は $N$ layers と layer あたり $M$ matrices をまとめて $\theta_z=\{z_1,\cdots,z_{N\times M}\}$ と書く。
- step2: 各 training task について `SVF` vectors を RL で学習する。付録では AdamW、learning rate $2\times10^{-3}$ with cosine decay、global batch size 256、gradient clipping、validation performance による early stopping と $\lambda$ 選択を用いる。
- step3: `Llama3-70B-Instruct` と vision tasks では memory usage を減らすため half of the layers に `SVF` を適用する。vision-language task の Llama training では training stability のため small negative reward $-0.1$ を使う。
- step4: 推論時は 1 パス目で $z^\prime$ を決め、2 パス目で $W^\prime$ を使って回答する。Prompt strategy は adaptation prompt でカテゴリ分類し、`others` なら base weights を使える。Cls-expert strategy は分類専用の $z^c$ を追加で学習する。Few-shot strategy は held-out few-shot prompts 上で CEM により $\alpha_k$ を探索する。
- step5: LoRA baseline は query / value projection layers に適用し、rank 16、LoRA alpha 32、dropout 0.05、global batch size 256 などを使う。LoRA instruction training では GSM8K、MBPP、Arc-Easy、TextVQA の official sources から solutions を集めて prompt に append する。

## 実験・結果

- **データセット / ベンチマーク**: Training tasks は GSM8K、MBPP-Pro、ARC-Easy。VLM domain の確認として TextVQA で `Llama3-Llava-Next-8B` も扱う。Unseen adaptation tasks は MATH、Humaneval、ARC-Challenge、OKVQA。モデルは `Llama3-8B-Instruct`、`Mistral-7B-Instruct-v0.3`、`Llama3-70B-Instruct`。
- **比較対象 / baseline**: Base model、LoRA、`SVF (Ours)`、`Transformer^2` の Prompt / Cls-expert / Few-shot。付録の追加比較では IA3 と DORA も使う。
- **指標**: Table 1 と Table 2 は各 task の test split における performance と normalized score を報告する。Table 3 は prompt adaptation strategy の 1st / 2nd pass inference time。Table 4 は trainable parameter 数、GSM8K score、MATH zero-shot transfer score。TeX 本文には、MATH や Humaneval の個別 metric 名は明示されていない。
- **主な結果**: Table 1 では、`Llama3-8B-Instruct` の GSM8K が base 75.89、LoRA 77.18、SVF 79.15、ARC-Easy が 88.59、88.97、89.56 で SVF が最高だが、MBPP-Pro は LoRA 67.68 が SVF 66.67 を上回る。`Mistral-7B-Instruct-v0.3` は GSM8K 42.83 -> SVF 49.74、ARC-Easy 81.65 -> 85.14、MBPP-Pro は LoRA と SVF が 51.52 で同値。`Llama3-70B-Instruct` は GSM8K 85.29 -> SVF 88.32、MBPP-Pro 80.81 -> 80.81、ARC-Easy 89.10 -> 88.47 でわずかに下がる。VLM では TextVQA で `Llama3-Llava-Next-8B` を SVF fine-tuning すると base performance が over 39% 改善すると本文が述べる。
- **主な結果**: Table 2 の unseen tasks では、`Llama3-8B-Instruct` で MATH 24.54 -> Few-shot 25.47、Humaneval 60.98 -> 62.99、ARC-Challenge 80.63 -> 82.61。`Mistral-7B-Instruct-v0.3` では MATH 13.02 -> Few-shot 13.39、Humaneval 43.29 -> 47.40、ARC-Challenge 71.76 -> 75.47 だが、ARC-Challenge では LoRA 75.77 が Few-shot 75.47 を上回る。`Llama3-70B-Instruct` は Prompt のみ報告され、MATH 40.64 -> 40.44、Humaneval 78.66 -> 79.88、ARC-Challenge 87.63 -> 88.48。
- **主な結果**: Table 3 の 2-pass inference cost は MATH が 1st 42.64s / 2nd 321.19s（13%）、Humaneval が 2.76s / 14.28s（19%）、ARC-Challenge が 13.40s / 28.51s（47%）。本文は、生成 token 数が長いタスクでは 1st pass の相対コストが小さく、ARC-Challenge は single choice problems なので比率が大きいと説明する。
- **主な結果**: Table 4 では、`SVF` + Policy gradient + MLP+attention が 0.58M params で GSM8K 79.23 / MATH 25.04。`SVF` + Next token pred + attention は 0.16M params だが 60.50 / 18.52 に落ちる。LoRA + Policy gradient + attention は 6.82M params で 57.92 / 15.72 と大きく崩れ、著者は LoRA training process の instability を指摘する。
- **主な結果**: Table 5 の cross-model transfer では、Llama で学習した SVF vector を Mistral に適用すると、MATH は 13.02 -> ordered 11.96 と下がるが、Humaneval は 43.29 -> 45.12、ARC-Challenge は 71.76 -> 72.01 と改善する。shuffled $\sigma_i$ は 10.52 / 40.24 / 70.82 で、ordered より一貫して悪い。cross-model Few-shot adaptation は 12.65 / 46.75 / 75.64。
- **著者が主張する貢献**: 導入部の contribution は、(1) growing set of pre-trained skills から LLM の挙動を動的に変える `Transformer^2` framework、(2) RL で小データから compact expert vectors を作る `SVF`、(3) Prompt / Cls-expert / Few-shot の 3 つの adaptation strategies である。

## 妥当性と限界

- **この主張を支える根拠**: `SVF` の有効性は Table 1 の複数モデル・複数訓練タスク、Table 4 の parameter / objective ablation、Figure `learning_curves_new.pdf` の learning curves によって支えられている。Self-adaptation の根拠は Table 2 の unseen tasks、OKVQA への VLM adaptation 記述、Figure `confusion_matrices.pdf` の job dispatching accuracy、Figure `cem_plot_vertical_tb.pdf` の learned weights、Table 5 の ordered / shuffled cross-model 比較である。
- **著者が認めている limitations / future work**: Conclusion は、`SVF` experts の capabilities が base model の latent components に tied していることを limitation として挙げ、model merging を方向性として述べる。また、CEM-based adaptation は specialized domains が増えると one-time computational costs が増える。Methods では、weak base model では reward が sparse になりうることも caveat として書かれている。
- **読者として注意すべき点**: 著者は monotonic performance benefits を主張するが、Table 2 の個別セルでは厳密な Prompt < Cls-expert < Few-shot ではない。例えば `Llama3-8B` の MATH は Prompt 25.22 が Cls-expert 25.18 より高く、ARC-Challenge も Prompt 81.74 が Cls-expert 81.37 より高い。`Mistral` の MATH では Prompt 11.86 と Cls-expert 11.60 が base 13.02 を下回る。
- **読者として注意すべき点**: `SVF` が LoRA より常に上、とは Table 1 からは言えない。`Llama3-8B` の MBPP-Pro は LoRA 67.68 が SVF 66.67 より高く、`Mistral` の MBPP-Pro は同値、`Llama3-70B` の ARC-Easy は base 89.10 から SVF 88.47 に下がる。したがって、著者の「across nearly all tasks and base models」という表現は、例外を含む傾向として読む必要がある。
- **読者として注意すべき点**: `Llama3-70B` と vision tasks では half of the layers だけに `SVF` を適用している。特に Table 2 の `Llama3-70B` は Prompt strategy だけが示され、Cls-expert / Few-shot は載っていない。
- **追加で確認したい実験 / 疑問**: TeX 本文には seed や confidence interval の体系的な記述が見当たらないため、MATH の +0.37 や +0.93 のような小幅改善の安定性は追加確認したい。Few-shot adaptation も付録で「validation set がないため best sample を報告」と説明されるので、held-out design や task 数を増やした場合の再現性を見たい。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **`Transformer^2`**: 推論時に 2 パスを使う self-adaptation framework。1 パス目で task properties / test-time behavior を観測して $z^\prime$ を決め、2 パス目で adapted weights により回答する。
- **`SVF` / Singular Value Fine-tuning**: weight matrix の singular values の scale vector $z$ だけを学習する PEFT 手法。論文では compact、compositional、regularized な expert vectors を作るための building block。
- **expert vector $z$**: 各 weight matrix の singular values をスケールする学習済みベクトル。タスクごとに $z$ の集合を作り、math / coding / reasoning などの domain-specific skill として扱う。
- **self-adaptation**: この論文では、外部から再学習をかけるのではなく、推論時に入力条件を見てモデル自身の重み変更を選択・合成する仕組みを指す。
- **Prompt strategy**: adaptation prompt により、入力 prompt を pre-defined domain topics または `others` に分類し、対応する expert vector を選ぶ方法。
- **Cls-expert strategy**: task identification 専用の追加 expert $z^c$ を `SVF` で学習し、1 パス目でそれを使って分類する方法。
- **Few-shot adaptation**: target task から held-out few-shot prompts を取り、CEM で $\alpha_k$ を探索して $z^\prime=\sum_k \alpha_k z_k$ を作る方法。各 question prompt を長くする traditional few-shot prompting とは違い、target task ごとに一度だけ係数探索する。
- **CEM / Cross-Entropy Method**: 係数探索に使う Monte Carlo optimization。samples を評価し、elite samples の統計で探索分布を更新する。
- **LoRA baseline**: 本論文では主に next-token prediction で training した PEFT baseline。adaptation comparison では、training tasks の全 checkpoints から test task ごとの最高 performance を報告する。
- **ordered / shuffled $\sigma_i$**: cross-model transfer の統制。Llama で学習した $z$ を Mistral に移すとき、ordered singular value scale の対応を保つ場合とランダムにシャッフルする場合を比べ、ordered の方がよいことを示す。

## 読む順番の提案

- まず `sections/sec1_introduction.tex` を読み、post-training、expert modules、LoRA、composition の問題設定を押さえる。正規ノートでは `Summary（著者の主張）` の「問題」と「貢献」に対応する。
- 次に `sections/sec3_methods.tex` の `Preliminaries` と `Singular value fine-tuning` を読み、$W=U\Sigma V^\intercal$、$W^\prime=U\Sigma^\prime V^\intercal$、Eq. (1) の RL objective を確認する。正規ノートでは `Notes / Quotes` の数式部分につながる。
- その後、同じ methods 節の Prompt / Classification expert / Few-shot adaptation を読む。ここで $z^\prime=\sum_k\alpha_k z_k$ と CEM の役割を確認すると、正規ノートの `Takeaway` にある expert vector の組成性が読みやすくなる。
- 実験は `sections/sec4_experiments.tex` と `tables/svf_train_tasks.tex`、`tables/svf_ada_tasks.tex`、`tables/inference_cost.tex`、`tables/table_ablation.tex`、`tables/analysis_cross_model_main.tex` を表番号順に見る。正規ノートの `Critical Thoughts` は、これらの表の例外セルを踏まえて読むとよい。
- 再現性や限界は `sections_app/Aimplementation.tex`、`sections_app/BaddExperiments.tex`、`sections_app/DEfficiencyImprovements.tex`、最後に `sections/sec5_conclusions.tex` を読む。few-shot の 10 samples / 100 CEM iterations、half-layer SVF、CEM の one-time cost がここにある。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2501.06252v3/`
- 正規ノート: `notes/arXiv-2501.06252v3.md`
