# TIES-Merging: Resolving Interference When Merging Models（モデルマージにおける task-vector interference の診断と解消）

- arXiv: https://arxiv.org/abs/2306.01708
- 一次ソース: ../papers/arXiv-2306.01708v2/
- 正規ノート: ../notes/arXiv-2306.01708v2.md

---

## 一言で言うと

同じ事前学習モデルから fine-tuning された複数の task-specific model を、追加訓練なしで 1 つの multitask model にまとめるとき、既存の平均系手法は task vector 間の干渉を無視して性能を落とす、という問題を扱う。TIES-Merging は `Trim`, `Elect Sign`, `Disjoint Merge` により、冗長な低 magnitude 値と符号不一致を処理してからマージする手法であり、Table~\ref{tab:main} と Table~\ref{tab:ood} で複数の NLP / Vision 設定において既存 baseline を上回ると著者は主張している。

## 何を議論する論文か

- **問題設定**: タスク集合 $\{t_1,\ldots,t_n\}$ と T5 / ViT などの pre-trained model があり、各タスクで fine-tuned parameters $\theta_\textrm{ft}$ が得られている。この複数 checkpoint を、task vector を介して単一の multitask model にマージする問題を扱う（`sections/background.tex`, "Problem Setting"）。
- **対象範囲 / 仮定**: マージ対象は共通の initialization と model architecture を持つ fine-tuned model である。本文の主要実験は PEFT の (IA)$^3$ / T0-3B、full fine-tuning の T5-base / T5-large、CLIP の ViT-B/32 / ViT-L/14 visual encoder で、validation set がある場合とない場合の両方を評価する。
- **既存研究との差分**: Simple Averaging, Fisher Merging, RegMean, Task Arithmetic は重みや task vector を平均・加算するが、TIES-Merging は平均の前に「redundant parameter values」と「sign disagreement」という 2 種類の interference を明示的に処理する点が差分である（`sections/introduction.tex`, "two major causes"）。
- **この論文で答えたい問い**: どの種類の干渉が model merging の性能低下を生むのか、それを task vector の trimming と sign election で軽減できるのか、またその効果が NLP / Vision、PEFT / full fine-tuning、in-domain / out-of-domain、validation 有無の設定で成り立つのかを調べる。

## 背景と前提

- **Pre-trained model と fine-tuning**: PTM は各タスクへ fine-tuning されることで性能・収束・サンプル効率の利点を得るが、タスクごとに checkpoint を保存・デプロイする必要があり、孤立したモデル同士は相互に情報を利用できない、という導入で論文は始まる。
- **Model merging**: 複数の task-specific model を追加訓練なしで 1 つにまとめる方法である。fine-tuned model が同じ pre-trained model から出ている場合、共通の最適化軌道を一部共有しているため、permutation symmetry を明示的に扱わなくてもマージできる場合がある、という related work の流れに位置づく。
- **Task vector**: Task Arithmetic に従い、fine-tuned parameters から initialization を引いた変化分として各タスクを表す。TIES-Merging は model weights を直接平均するのではなく、この task vector の値・符号・大きさを操作してから足し戻す。
- **Interference の 2 分類**: 1 つ目は、あるタスクでは重要な parameter value が、他タスクでは redundant な小さい値と平均されて magnitude を薄められる現象である。2 つ目は、同じ parameter で正負がタスク間で食い違い、単純平均で打ち消し合う現象である。
- **Baseline との関係**: Simple Averaging は $\theta_m = \sum_{t=1}^{n}\theta_t/n$、Fisher Merging は diagonal Fisher による重要度重み、RegMean は layer activation の least-squares、Task Arithmetic は $\theta_m=\theta_\textrm{init}+\lambda * \sum_{t=1}^n \tau_t$ を使う（`sections/experiments.tex`, "Baseline Methods"）。

## 提案手法

### コアアイデア

TIES-Merging は正式には `TrIm, Elect Sign & Merge` であり、複数の task vector $\{\tau_t\}_{t=1}^{n}$ に対して 3 ステップを順に適用する。まず各 task vector で magnitude 上位 $k\%$ だけを残して低 magnitude 値を 0 にする。次に parameter ごとに、trim 後の値の合計から最終符号 $\gamma_m^p$ を選ぶ。最後に、その elected sign と同じ符号を持つ task の値だけで平均を取り、merged task vector $\tau_m$ を作る。Figure~\ref{fig:diagram_main} と Algorithm~\ref{alg:merging} がこの手順の対応箇所である。

この手法の狙いは、平均の前に interference を減らすことである。Trim は redundant parameters が influential parameter を薄める問題を抑える。Elect と Disjoint Merge は sign disagreement による打ち消しを避ける。validation set がない設定では、著者は PEFT 実験から作った固定 recipe として $k=20$, $\lambda=1$, mass-based sign election, disjoint mean を使う（`sections/experiments.tex`, "Merging in Absence of the Validation Set"; `sections/appendix.tex`, `sec:app_validation`）。

### 重要な定義・数式

$$
\tau_{t} = \theta_\textrm{ft}^t - \theta_\textrm{init}^t,\quad
\tau_t = \gamma_t \odot \mu_t,\quad
\gamma_t = \textrm{sgn}(\tau_t),\quad
\mu_t = |\tau_t|
$$

**式の意味**: タスク $t$ の fine-tuning による parameter 空間での移動を task vector $\tau_t$ として定義し、それを sign vector と magnitude vector に分解する。`sections/background.tex` の task-vector 定義と `sections/method.tex` の preliminaries に対応する。

**記号の定義**:
- $\tau_t \in \mathbb{R}^d$ ... タスク $t$ の task vector。initialization から低 loss 領域へ向かう方向と移動量を表す。
- $\theta_\textrm{ft}^t$ ... タスク $t$ で fine-tuning された parameters。
- $\theta_\textrm{init}^t$ ... task vector 定義で引かれる initialization の parameters。Algorithm~\ref{alg:merging} では $\theta_\textrm{init}$ と書かれる。
- $\gamma_t$ ... parameter ごとの符号を持つ sign vector。
- $\mu_t$ ... parameter ごとの magnitude を持つ vector。
- $\odot$ ... elementwise product。

**この論文での役割**: TIES-Merging は task vector の値を直接処理するため、この定義が手法全体の入力表現である。符号と magnitude への分解は、Trim と Elect を別々の操作として定義するための前提になる。

$$
\hat{\tau}_t \leftarrow \text{keep\_topk\_reset\_rest\_to\_zero}(\tau_t, k)
$$

**式の意味**: 各 task vector について、magnitude が大きい top-$k\%$ の値だけを残し、bottom $(100-k)\%$ を 0 にする。Algorithm~\ref{alg:merging} の Step 1 であり、method section では $\hat{\tau}_t=\hat{\gamma}_t\odot\hat{\mu}_t$ とも書かれる。

**記号の定義**:
- $\hat{\tau}_t$ ... Trim 後の task vector。
- $\tau_t$ ... Trim 前の task vector。
- $k$ ... 残す parameter value の割合。validation なし recipe では $k=20$。
- $0$ ... task vector 上で 0 にすることは、対応する fine-tuned parameter value を pre-trained model の値へ戻すことに等しい。

**この論文での役割**: Figure~\ref{fig:reset-bottomk} で、top-$20\%$ だけを残しても 11 task 平均性能が劣化しないと示し、redundant parameter values を捨てる根拠にしている。

$$
\gamma_m^p = \textrm{sgn}\left(\sum_{t=1}^{n} \hat{\tau}_t^p\right)
$$

**式の意味**: parameter $p$ について、trim 後の値をタスク方向に合計し、その合計の符号を merged model の elected sign とする。method section では、正方向と負方向の total mass を比較し、より大きい total movement の符号を選ぶ、と説明される。

**記号の定義**:
- $\gamma_m^p$ ... merged model の parameter $p$ に対して選ばれた符号。
- $\hat{\tau}_t^p$ ... task $t$ の trim 後 task vector における parameter $p$ の値。
- $n$ ... マージする task-specific model の数。
- $\textrm{sgn}(\cdot)$ ... 正なら $+1$、0 なら $0$、負なら $-1$ を返す符号関数。

**この論文での役割**: Sign disagreement を解消する中心式である。Figure~\ref{fig:imp_conflict} は、trim 後にも sign conflict が残り、マージするモデル数が増えるほど conflict が増えることを示すため、この式による election が必要になる。

$$
\mathcal{A}^p = \{t \in [n] ~|~ \hat{\gamma}^p_t = \gamma_m^p\},\quad
\tau^p_m = \frac{1}{|\mathcal{A}^p|}\sum_{t \in \mathcal{A}^p} \hat{\tau}_t^p,\quad
\theta_m = \theta_\textrm{init} + \lambda * \tau_m
$$

**式の意味**: parameter $p$ ごとに、elected sign と同じ符号を持つ task だけを集合 $\mathcal{A}^p$ に入れ、その値だけで平均する。最後に merged task vector $\tau_m$ を $\lambda$ 倍して initialization に足し戻し、merged model $\theta_m$ を得る。

**記号の定義**:
- $\mathcal{A}^p$ ... parameter $p$ で elected sign と一致する task の集合。
- $\hat{\gamma}_t^p$ ... trim 後の task vector の parameter $p$ における符号。
- $\tau_m^p$ ... merged task vector の parameter $p$ の値。
- $|\mathcal{A}^p|$ ... elected sign と一致した task 数。
- $\theta_m$ ... 最終的な merged model parameters。
- $\lambda$ ... scaling hyperparameter。validation なし recipe では $\lambda=1$。

**この論文での役割**: Disjoint Merge は、負け側の符号や trim で 0 になった値を平均に入れないため、単純平均による打ち消しと magnitude shrinkage を抑える。Table~\ref{tab:ablation} では Disjoint Mean を外すと T5-base で 74.5 から 72.6、(IA)$^3$ で 70.7 から 67.5 に下がる。

### 実装 / アルゴリズム上の要点

- step1: 各 fine-tuned model $\theta_t$ から $\tau_t=\theta_t-\theta_\textrm{init}$ を作り、magnitude top-$k\%$ 以外を 0 にして $\hat{\tau}_t$ を得る。
- step2: 全 task の $\hat{\tau}_t$ を parameter ごとに足し、$\gamma_m=\textrm{sgn}(\sum_t \hat{\tau}_t)$ を作る。
- step3: 各 parameter $p$ で $\hat{\gamma}_t^p=\gamma_m^p$ となる task の値だけを平均する。TeX は "the disjoint mean always ignores the zero values" と明記している。
- step4: $\theta_m \leftarrow \theta_\textrm{init}+\lambda * \tau_m$ で checkpoint を作る。merge 自体は重み操作であり、追加訓練を前提にしない。
- validation set がない設定では、PEFT 実験の grid search から得た $k=20$, $\lambda=1$ を T5 / ViT の full fine-tuning 設定に適用する。Appendix の search 範囲は $k\in\{10,20,30\}$、$\lambda=0.8$ から $3.0$ まで 0.1 刻みである。

## 実験・結果

- **データセット / ベンチマーク**: PEFT 実験では (IA)$^3$ を T0-3B に追加し、COPA, H-SWAG, Story Cloze, CB, RTE, WSC, Winogrande, WiC, ANLI-R1, ANLI-R2, ANLI-R3 の 11 tasks を使う（本文では ANLI を natural language inference の dataset としてまとめて述べ、Appendix Table~\ref{tab:app_main_ia3} では r1/r2/r3 を別列で報告する）。T5-base / T5-large の full fine-tuning では QASC, WikiQA, QuaRTz, PAWS, Story Cloze, Winogrande, WSC の 7 tasks を使う。Vision では CLIP の ViT-B/32 と ViT-L/14 visual encoder を Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN の 8 tasks で fine-tune する。OOD では T5 系を 7 in-domain tasks でマージした後、Cosmos QA, Social IQA, QuAIL, WiC, COPA, H-SWAG の 6 held-out tasks で評価する。
- **比較対象 / baseline**: Simple Averaging, Fisher Merging, RegMean, Task Arithmetic に加え、個別 Fine-tuned models と Multitask model を参照点として報告する。ModelSoups 風の同一タスク checkpoint merging では Fisher Merging の設定と code を使い、HuggingFace 上の top-10 BERT-base checkpoints を RTE, MRPC, WNLI でマージする。
- **指標**: Table~\ref{tab:main} は task 平均の test set performance を報告する。NLP の評価は Appendix `sec:app_training_details` にある rank classification で、候補 label strings の log probabilities を順位付けし、最上位が正解なら正答とする。(IA)$^3$ 実験では dataset ごとに全 prompt templates の median score を報告する。
- **主な結果**: validation ありの Table~\ref{tab:main} では、TIES-Merging は (IA)$^3$/T0-3B で 66.4（Task Arithmetic 63.9 に対して +2.5）、T5-base で 73.9（Task Arithmetic 73.2 に対して +0.7）、T5-large で 76.9（Task Arithmetic 73.3 に対して +3.6）、ViT-B/32 で 73.6（RegMean 71.8 に対して +1.8）、ViT-L/14 で 86.0（Task Arithmetic 84.5 に対して +1.5）である。Introduction は in-domain evaluation で strongest baseline に対し NLP 平均 +2.3%、Vision 平均 +1.7% absolute と述べる。
- **主な結果**: validation なしの Table~\ref{tab:main} では、TIES-Merging は T5-large 74.4、ViT-B/32 72.4、ViT-L/14 86.0 で各 no-validation baseline を上回る。一方、T5-base では TIES 69.7 で、Table~\ref{tab:main} 上の Task Arithmetic 73.2 を下回るため、固定 recipe が常に最良という結果ではない。
- **主な結果**: OOD Table~\ref{tab:ood} では、T5-base で TIES 35.3（RegMean 34.3 に対して +1.0）、T5-large で TIES 40.4（RegMean 36.0 に対して +4.4）であり、著者は out-of-domain generalization でも最強 baseline を上回ると主張する。
- **主な結果**: Figure~\ref{fig:num_tasks} では、マージする task 数が増えると全手法の性能は下がるが、Task Arithmetic は TIES-Merging より急に下がる。2 tasks のとき TIES と Task Arithmetic は平均 normalized accuracy がほぼ 1 で、Simple Averaging は約 $10\%$ performance drop と述べられる。
- **主な結果**: Table~\ref{tab:ablation} では、TIES から各部品を外すと性能が落ちる。特に Scale を外すと T5-base 74.5→72.0、(IA)$^3$ 70.7→65.5、Disjoint Mean を外すと T5-base 74.5→72.6、(IA)$^3$ 70.7→67.5 である。
- **主な結果**: Oracle Sign の Table~\ref{tab:oracle} では、(IA)$^3$ で TIES 66.4 に対し、multitask vector の sign を使う TIES (Oracle Sign) は 72.0、Multitask は 73.1 である。著者は、正しい correction directions が得られれば multitask model との gap をかなり埋められる、と解釈する。
- **著者が主張する貢献**: 著者は、model merging の interference を redundant parameter values と sign disagreement に分けて示したこと、Trim / Elect / Disjoint Merge による追加訓練なしの手法を提案したこと、多様な modality / model size / fine-tuning setting / validation setting で有効性を示したこと、sign の重要性を flipping-sign 実験と Oracle Sign 実験で示したことを主張している。

## 妥当性と限界

- **この主張を支える根拠**: Figure~\ref{fig:reset-bottomk} は top-$20\%$ の high-magnitude parameters だけを残しても 11 tasks 平均性能が劣化しないことを示し、Trim の前提を支える。Figure~\ref{fig:imp_conflict} は trim 後にも sign conflict が残り、マージ対象モデル数とともに増えることを示す。Figure~\ref{fig:interference} は Standard Mean が magnitude を縮め、trimming / electing がそれを抑えることを示す。Figure~\ref{fig:flip_signs} は top-$20/30\%$ の符号を反転すると性能が単調に落ち、bottom-$80/70\%$ の符号反転は影響が小さいことを示す。
- **著者が認めている limitations / future work**: Appendix `sec:limitation` は、weight interpolation がなぜ・いつ機能するかの理論理解が限られていること、merging が common initialization と model architecture に依存すること、individual-task models の merging は simultaneous multitask training にまだ遅れること、特定 domain に有用な multitask model を作るための checkpoint 選択方法が明確でないこと、multitask model にアクセスせず multitask signs を推定する方法が future work であることを挙げている。
- **読者として注意すべき点**: validation なし recipe の $k=20$, $\lambda=1$ は PEFT 設定から選ばれ、T5 / ViT の full fine-tuning へ未調整で適用されている。Appendix Figure~\ref{fig:hyperparams} は、$k$ を増やすと性能が落ちて飽和し、その曲線は task vector の parameter value 分布で変わり得ると述べる。
- **読者として注意すべき点**: Table~\ref{tab:main} の T5-base validation なし設定では TIES 69.7 が Task Arithmetic 73.2 を下回る。また TeX 内では、Appendix Table~\ref{tab:app_main_t5base} の no-validation Task Arithmetic average が 73.9 と記載されており、Table~\ref{tab:main} の 73.2 と一致しない。このノートでは主表として Table~\ref{tab:main} を優先した。
- **読者として注意すべき点**: Training Details では evaluation が rank classification であり、classification tasks と multiple-choice tasks に対応できると書かれている。要約、翻訳、コード生成、対話などの生成タスクでの結果は TeX 中には示されていない。
- **追加で確認したい実験 / 疑問**: Appendix Table~\ref{tab:app_oracle} は 32 validation examples per task の few-shot multitask sign 推定で、mean initialization の場合 67.7（TIES 66.4 から +1.2、本文は +1.3 と記述）を示す。multitask model なしによりよい sign vector を推定できるかは、この論文自身が future work として残している。
- **追加で確認したい実験 / 疑問**: Background では PEFT method の例として LoRA も引用されるが、本文の PEFT 実験対象は (IA)$^3$ である。LoRA に対する TIES-Merging の結果は TeX 中には示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **task vector**: タスクごとの fine-tuning で initialization からどちら向きにどれだけ parameter が動いたかを表す vector。TIES-Merging の操作対象である。
- **sign vector**: task vector の各 entry の符号を集めた vector。論文では符号を「その parameter 軸に沿って loss を下げる方向」として説明する。
- **magnitude vector**: task vector の絶対値を集めた vector。Trim はこの magnitude に基づいて top-$k\%$ を選ぶ。
- **redundant parameter values**: fine-tuning 中に変化していても、そのタスク性能への影響が小さいとみなされる値。TIES では bottom $(100-k)\%$ を 0 にすることでマージ時の干渉源から外す。
- **influential parameters**: magnitude が大きく、Figure~\ref{fig:reset-bottomk} の文脈では性能維持に重要と見なされる parameter values。
- **sign disagreement / sign conflict**: 同じ parameter について task vector 間で正負が分かれる状態。単純平均では magnitude shrinkage や打ち消しを生む。
- **Elect Sign**: parameter ごとに正方向・負方向の total mass を比べ、merged model の符号 $\gamma_m^p$ を選ぶ操作。
- **Disjoint Merge / disjoint mean**: elected sign と同じ符号を持つ値だけを平均する操作。0 値は常に無視される。
- **Task Arithmetic**: task vector をスケールして足し、$\theta_\textrm{init}$ に加える baseline。TIES はこの task-vector merging の枠組みを使いつつ、加算前に trimming と sign resolution を行う。
- **Oracle Sign**: multitask model から得た sign vector $\gamma_\textrm{mult}$ を elected sign として使う上限的な実験設定。Table~\ref{tab:oracle} で TIES 66.4 から 72.0 へ上がる。
- **rank classification**: NLP 評価で使う方法。候補 label strings の log probability を比較し、正解ラベルが最上位なら正答とする。

## 読む順番の提案

- まず `notes/arXiv-2306.01708v2.md` の Summary を読み、問題が「冗長値」と「符号不一致」の 2 種類の interference に分けられていることを押さえる。
- 次に `main.tex` の abstract と `sections/introduction.tex` を読み、Figure~\ref{fig:diagram_conflict} と Figure~\ref{fig:diagram_main} で論文の問題意識と手法の全体像を見る。
- `sections/background.tex` の "Problem Setting" と Algorithm~\ref{alg:merging} を読む。ここが正規ノートの「Step definitions」と対応しており、task vector、Trim、Elect、Disjoint Merge の記法がまとまっている。
- `sections/method.tex` を読み、$\tau_t=\gamma_t\odot\mu_t$、$\gamma_m^p=\textrm{sgn}(\sum_t\hat{\tau}_t^p)$、$\mathcal{A}^p$ の定義を確認する。正規ノートの Takeaway「Sign election が肝」はこの節に対応する。
- `sections/experiments.tex` の Table~\ref{tab:main} と Table~\ref{tab:ood} を先に見る。次に Figure~\ref{fig:num_tasks}, Figure~\ref{fig:flip_signs}, Table~\ref{tab:ablation}, Table~\ref{tab:oracle} を読むと、正規ノートの「干渉はモデル数に対して増える」「Oracle Sign 実験」の主張につながる。
- 最後に Appendix `sec:limitation`, `sec:app_validation`, `sec:app_estimating_multitask_sign` を読む。正規ノートの Critical Thoughts にある限界、$k=20$ / $\lambda=1$ の根拠、multitask sign 推定の話はここで裏取りできる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2306.01708v2/`
- 正規ノート: `notes/arXiv-2306.01708v2.md`
