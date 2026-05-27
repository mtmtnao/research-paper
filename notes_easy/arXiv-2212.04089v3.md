# Editing Models with Task Arithmetic（タスクベクトルによる事前学習済みモデル編集）

- arXiv: https://arxiv.org/abs/2212.04089
- 一次ソース: ../papers/arXiv-2212.04089v3/
- 正規ノート: ../notes/arXiv-2212.04089v3.md

---

## 一言で言うと

この論文は、同じ事前学習済みモデルをあるタスクで fine-tune した後の重み差分を **task vector** として扱い、その足し算・引き算・類推でモデルの挙動を編集できるかを検証する。task vector や fine-tuned checkpoints が得られていれば、CLIP、GPT-2、T5 の実験で、忘却、マルチタスク化、ラベルなし target への転移を、追加学習や推論時オーバーヘッドなしに実行できると著者は主張する。

## 何を議論する論文か

- **問題設定**: 事前学習後のモデルを、下流タスク性能の改善、望ましくない挙動の緩和、バイアス抑制、人間選好への整合、新情報への更新などのために「編集」したい。ここで著者は editing を pre-training 後の任意の介入と定義している（`01_introduction.tex`）。
- **対象範囲 / 仮定**: task は dataset と fine-tuning loss によって具体化される（`02_task_vectors.tex`）。操作は重みベクトルの element-wise な加減算なので、同じ architecture のモデルに限定され、実験では同じ pre-trained initialization から fine-tune されたモデルだけを使う（`06_discussion.tex`）。
- **既存研究との差分**: weight interpolation、weight averaging、model soups は fine-tuned モデル間の補間・平均化に近い。本論文はそれを task vector として定式化し、interpolation だけでなく negation による extrapolation、複数 task vector の addition、task analogy まで扱う（`97_related_work.tex`, `99_appendix.tex`）。
- **この論文で答えたい問い**: fine-tune で得られる重み差分は、単なる最適化の副産物ではなく、足し引きして再利用できる「タスク方向」なのか。さらに、その方向を逆向きに進むと忘却できるのか、複数足すと 1 つのモデルに複数能力を入れられるのか、3 つの関連タスクから第 4 のタスクを改善できるのかを問う。

## 背景と前提

- **fine-tuning と weight space**: pre-trained weights $\theta_\textrm{pre}$ から、タスク $t$ のデータと loss で fine-tune した weights $\theta_\textrm{ft}^t$ へ移動する。この論文は、その移動方向を weight space 上のベクトルとして扱う。
- **open-ended models**: 著者は、新しい task-specific parameter を追加せずに downstream task へ fine-tune できるモデルに焦点を当てる。例として open-vocabulary image classifiers と text-to-text models が挙げられている（`02_task_vectors.tex`）。新しい classification head などが必要な場合は shared weights だけを merge する可能性を述べるが、本論文では future work 扱い。
- **control task**: モデル編集は target task だけでなく、編集対象外の挙動を壊さない必要がある。画像忘却では ImageNet、毒性低減では WikiText-103 perplexity が control task として使われる（`03_negation.tex`）。
- **weight averaging / ensemble との関係**: Appendix は、task vector の適用や addition が pre-trained model と fine-tuned models の線形結合に等しいことを説明する。特に $\lambda=0.5$ で 2 つの task vector を足す場合、2 つの fine-tuned models の一様平均になる（`99_appendix.tex`, Figure `fig:ensembles-corr`）。
- **baseline との関係**: 忘却では gradient ascent と layer ごとに同じ大きさを持つ random vector を比較対象にする。追加で、毒性低減では non-toxic samples で fine-tune する baseline も使う（`03_negation.tex`, `99_appendix.tex`）。

## 提案手法

### コアアイデア

task vector は「ある task で fine-tune するために重みがどの方向へ動いたか」を表す差分である。著者は、この差分が task の情報を持つ方向だとみなし、同じ architecture の任意の weights $\theta$ に $\lambda$ 倍して足す。さらに、task vector に対して negation、addition、analogy という 3 種類の arithmetic operations を定義する。すべて element-wise operations on model weights なので、適用後のモデルは同じサイズで、推論時の追加 memory/compute は発生しない（`01_introduction.tex`, `02_task_vectors.tex`, `98_conclusion.tex`）。

### 重要な定義・数式

$$
\tau_t = \theta_\textrm{ft}^t - \theta_\textrm{pre}
$$

**式の意味**: task $t$ の fine-tuned weights から pre-trained weights を引いた element-wise difference を task vector と定義する（`02_task_vectors.tex`）。これは Figure `fig:main` の (a) に対応する。

**記号の定義**:
- $\tau_t \in \mathbb{R}^d$ ... task $t$ の task vector
- $\theta_\textrm{pre} \in \mathbb{R}^d$ ... pre-trained model の weights
- $\theta_\textrm{ft}^t \in \mathbb{R}^d$ ... task $t$ で fine-tune した後の対応する weights
- $d$ ... モデル重みを 1 本のベクトルとして見たときの次元

**この論文での役割**: 以降の negation、addition、analogy はすべてこの差分を材料にする。$\lambda=1$ で pre-trained model に単一 task vector を足すと、その task の fine-tuned model に戻る。

$$
\theta_\textrm{new} = \theta + \lambda \tau_\textrm{new}
$$

**式の意味**: arithmetic operation で得た $\tau_\textrm{new}$ を、同じ architecture の weights $\theta$ に $\lambda$ 倍して足し、編集後の weights を得る（`02_task_vectors.tex`）。

**記号の定義**:
- $\theta_\textrm{new}$ ... 編集後モデルの weights
- $\theta$ ... task vector を適用する元の weights
- $\lambda$ ... scaling term。実験では held-out validation sets で決める
- $\tau_\textrm{new}$ ... negation、addition、analogy などから得られた task vector

**この論文での役割**: この式が「モデル編集」の実体である。追加の fine-tuning ではなく、保存済み weights の element-wise addition だけで編集を行う。

$$
\tau_\textrm{new} = -\tau
\quad\textrm{or}\quad
\tau_\textrm{new} = \sum_i \tau_i
$$

**式の意味**: 左は task vector を反転して target task の性能を下げる negation、右は複数 task vector を足して multi-task model や単一 target task の改善を狙う addition である（`02_task_vectors.tex`）。

**記号の定義**:
- $\tau$ ... 忘却・抑制したい task の task vector
- $-\tau$ ... fine-tuned model 方向とは逆向きの vector
- $\tau_i$ ... 各 task $i$ で得た task vector
- $\sum_i \tau_i$ ... 複数 task vector の element-wise sum

**この論文での役割**: negation は Section `sec:negation` の画像分類忘却と毒性低減、addition は Section `sec:addition` の CLIP multi-task 化と T5/GLUE 改善で評価される。

$$
\tau_\textrm{new} = \tau_C + (\tau_B - \tau_A)
$$

**式の意味**: task $A$, $B$, $C$, $D$ が "$A$ is to $B$ as $C$ is to $D$" の関係を持つとき、$A,B,C$ の task vectors から $D$ 向けの vector を構成する（`02_task_vectors.tex`, `05_analogies.tex`）。

**記号の定義**:
- $\tau_A, \tau_B, \tau_C$ ... 関連する 3 task の task vectors
- $\tau_B - \tau_A$ ... $A$ から $B$ への変化方向
- $\tau_C + (\tau_B - \tau_A)$ ... 同じ変化方向を $C$ に適用して得る target 側の推定 vector

**この論文での役割**: labeled data が少ない、または無い target task を改善するための中心式である。sentiment domain generalization と画像 subpopulation の実験で使われる。

$$
\hat{\tau}_\textrm{yelp;\,sent}
= \tau_\textrm{amazon;\,sent}
+ (\tau_\textrm{yelp;\,lm} - \tau_\textrm{amazon;\,lm})
$$

**式の意味**: Yelp sentiment を target、Amazon sentiment を labeled auxiliary とした具体例である。Yelp/Amazon の unsupervised language modeling task vectors を使い、Amazon で得た sentiment 方向を Yelp ドメインへ移す（`05_analogies.tex`）。

**記号の定義**:
- $\hat{\tau}_\textrm{yelp;\,sent}$ ... Yelp sentiment 用に analogy で推定した task vector
- $\tau_\textrm{amazon;\,sent}$ ... Amazon の labeled sentiment analysis で得た task vector
- $\tau_\textrm{yelp;\,lm}$ ... Yelp 入力の unsupervised language modeling で得た task vector
- $\tau_\textrm{amazon;\,lm}$ ... Amazon 入力の unsupervised language modeling で得た task vector

**この論文での役割**: Table `tab:sentiment-analog` の domain generalization 実験を読むための具体式である。target の labeled data を使わずに target sentiment accuracy を上げる、という主張を支える。

### 実装 / アルゴリズム上の要点

- **task vector を作る**: 同じ pre-trained initialization から task ごとに fine-tune し、$\theta_\textrm{ft}^t - \theta_\textrm{pre}$ を保存する。Section `sec:forget_img` / `sec:add_img` の 8 画像分類タスクでは、CLIP を 2000 iterations、batch size 128、learning rate 1e-5、cosine annealing、200 warm-up steps、AdamW、weight decay 0.1 で fine-tune し、CLIP text encoder 由来の classification layer は freeze する（`99_appendix.tex`）。
- **scaling coefficient を選ぶ**: $\lambda$ は held-out validation で決める。CLIP negation では $\{0.0,0.05,0.1,\cdots,1.0\}$ を評価し、control task で pre-trained model の accuracy の少なくとも 95% を保つ最大値を選ぶ。GPT-2 toxicity では $\{0.0,0.1,\cdots,1.0\}$ から WikiText-103 perplexity が pre-trained から 0.5 points 以内にある最大値を報告する（`99_appendix.tex`）。
- **addition は基本的に単一 $\lambda$**: 複数 task vector の和全体に 1 つの scaling coefficient を使う。Appendix では $\lambda \in \{0,0.05,0.1,\cdots,1.0\}$ を使い、0.3 から 0.5 が多くのケースで close to optimal だが、可能なら tuning を推奨している（`99_appendix.tex`, Figure `fig:acc-per-alpha`）。
- **analogy の係数**: sentiment analogy では、sentiment task vector 用と language modeling task vectors 用の 2 つの scaling coefficients を使う。subpopulation analogy の Appendix では独立係数 $\lambda_A,\lambda_B,\lambda_C$ も試し、平均で 0.7 percentage points 改善するが、評価回数が 10 ではなく $10^3$ になる（`05_analogies.tex`, `99_appendix.tex`）。
- **適用後は 1 つのモデル**: arithmetic の結果は weights の加減算なので、ensemble のように複数モデルを推論時に保持・実行する必要はない。著者は「no extra inference cost」と述べる（`98_conclusion.tex`）。

## 実験・結果

- **データセット / ベンチマーク**: 画像分類では CLIP ViT-B/32, ViT-B/16, ViT-L/14 と Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN を使い、control は ImageNet。毒性低減では GPT-2 Large を中心に Civil Comments の toxicity score $>0.8$ で task vector を作り、Detoxify による 1000 generations の毒性と WikiText-103 perplexity を測る。addition の NLP では T5-base と GLUE の MRPC, RTE, CoLA, SST-2、Hugging Face Hub の 427 compatible checkpoints を使う。analogy では Yelp/Amazon binary sentiment、ImageNet と human sketches の 125 overlapping classes を使う。
- **比較対象 / baseline**: 忘却では Pre-trained、Fine-tuned、Gradient ascent、Random vector、Negative task vector を比較する。毒性では Fine-tuned on non-toxic も加える。addition の画像実験では複数 specialized fine-tuned models を normalized accuracy 1.0 の基準にする。sentiment analogy では Fine-tuned on auxiliary、Task analogies、Fine-tuned on target を比較し、target fine-tuning は target labeled data がある場合の上限的な比較として使われる。
- **指標**: 画像は accuracy (%) と normalized accuracy。GLUE では MRPC が $F_1$ と accuracy の平均、RTE/SST-2 が accuracy、CoLA が Matthews correlation coefficient。毒性は `% toxic generations`、`Avg. toxicity score`、WikiText-103 perplexity。NLP checkpoint addition の Appendix では IMDB accuracy、RACE/QASC exact match、MultiNews/SQuAD/CommonGen ROUGE-2 も使う。
- **主な結果**: Table `tab:forget_image` では ViT-L/14 の target average accuracy が Pre-trained 64.8 から Negative task vector 19.0 へ下がり、control ImageNet は 75.5 から 72.9 にとどまる。Gradient ascent は target 3.93 まで下げるが control 16.3 まで崩れる。Random vector は target 60.9 でほぼ忘却できない。
- **主な結果**: Table `tab:toxicity` では GPT-2 Large の toxic generations が Pre-trained 4.8% から Negative task vector 0.8% に下がり、Avg. toxicity score は 0.06 から 0.01、WikiText-103 perplexity は 16.4 から 16.9 になる。Gradient ascent は toxic generations 0.0% だが perplexity $>10^{10}$、Fine-tuned on non-toxic は 1.8% / 0.03 / 17.2。
- **主な結果**: CLIP addition では、2 task vectors の addition が specialized models の 98.9% normalized accuracy に達する（Figure `fig:clip-add-2`）。8 task の全 subset、合計 $2^8$ 通りを調べると、全 task vectors が利用可能なときの best model は average normalized performance 91.2% に達する（Figure `fig:clip-add-all`）。
- **主な結果**: Table `tab:glue` では T5-base の GLUE 4 task 平均が Fine-tuned 78.1 から Fine-tuned + task vectors 78.6 に上がる。内訳は MRPC 88.5→89.3、RTE 77.3→77.5、CoLA 52.3→53.0、SST-2 94.5→94.7。
- **主な結果**: Table `tab:sentiment-analog` では target=Yelp の Task analogies が T5-small/base/large で 89.9/93.0/95.1、Fine-tuned on auxiliary が 88.6/92.3/95.0、Fine-tuned on target が 91.1/93.4/95.5。target=Amazon では Task analogies が 89.0/92.7/95.2、auxiliary が 87.9/90.8/94.8、target fine-tuning が 90.2/93.2/95.5。
- **主な結果**: subpopulation analogy では、ImageNet と sketches の 125 overlapping classes を 4 subpopulations に分け、3 つから 4 つ目を推定する。Figure `fig:clip-analogies` は pre-trained model に対して平均 3.4 percentage points 改善し、target subpopulation の約 100 annotated samples を集めた場合と roughly same gain と述べる。
- **著者が主張する貢献**: task vector arithmetic という統一的な編集枠組みを提示し、negation/addition/analogy を CLIP、GPT-2、T5 の複数モデル・モダリティで示したこと。さらに、公開 fine-tuned checkpoints を task vector の供給源として再利用できること、編集後モデルが単一モデルなので inference cost が増えないことを主張している。

## 妥当性と限界

- **この主張を支える根拠**: target task と control task を同時に測り、negation が「忘却だけでなく保持」も満たすかを見ている。Table `tab:forget_image` と `tab:toxicity` では gradient ascent が target を下げる一方で control を大きく壊すこと、random vector が同じ効果を持たないことが示される。
- **この主張を支える根拠**: addition の結果は normalized accuracy で specialized models と比較されるため、難易度の異なる task を平均しやすい。Appendix では multi-task training も比較し、8 tasks の jointly fine-tuned model は 0.994、task vector の best result は 0.912 と報告する。これは task vector が modular だが、joint training にはまだ headroom があることを示す。
- **この主張を支える根拠**: Discussion の Figure `fig:cossim` は CLIP task vectors が多くの場合 close to orthogonal で、MNIST/SVHN/GTSRB や EuroSAT/RESISC45 のように意味的に近い task で cosine similarity が高いと述べる。著者はこれが addition の干渉の小ささを説明しうると推測している。
- **著者が認めている limitations / future work**: task vectors は same architecture に制限される。さらに、実験では same pre-trained initialization から fine-tune したモデルだけを使う。新しい parameters を導入する fine-tuning では shared weights だけを merge する可能性があるが、これは future work とされる（`02_task_vectors.tex`, `06_discussion.tex`）。
- **著者が認めている limitations / future work**: negation は、fine-tuning による gain が小さい task では忘却方向も明確でなくなる可能性がある。Appendix `sec:when-neg-works` は、fine-tuning gain と task vector subtract による accuracy drop の正の相関を報告し、すでにモデルが強く持っている挙動を消す場合は難しいと述べる。
- **読者として注意すべき点**: $\lambda$ は実験ごとに validation で選ばれており、完全なゼロコストではない。ただし、著者は scaling coefficient の変更には追加 training が不要なので通常の hyperparameter tuning より安いと述べる。public checkpoints を混ぜる NLP addition では、fine-tuning learning rate の違いが variance の一因かもしれないと著者は仮説を述べる（`06_discussion.tex`）。
- **読者として注意すべき点**: toxicity 評価は Detoxify と RealToxicityPrompts/Perspective API による自動評価であり、TeX 中に人手評価は明示されていない。Appendix は RealToxicityPrompts の challenging set ではまだ significant headroom for improvement があると述べる。
- **追加で確認したい実験 / 疑問**: same architecture / same initialization という制限を、Git re-basin などの weight alignment と組み合わせてどこまで緩和できるか。task vector ごとの独立 scaling や層ごとの scaling が、評価回数の増加に見合う改善をもたらすか。毒性低減が自動分類器のスコアだけでなく、人手評価や adversarial prompts でも保たれるか（TeX 中には明示されていない）。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **editing**: pre-training 後にモデルへ行う任意の介入。性能改善、望ましくない挙動の緩和、情報更新などを含む。
- **task**: dataset と fine-tuning loss function で具体化される対象。単なるタスク名ではなく、どのデータでどの目的関数を最適化するかまで含む。
- **task vector**: $\theta_\textrm{ft}^t - \theta_\textrm{pre}$。fine-tuning により重みが動いた方向を表す。
- **task arithmetic**: task vectors に対する negation、addition、analogy などの arithmetic operations。重みの element-wise operations として実装される。
- **negation**: $\tau_\textrm{new}=-\tau$。target task の性能や特定挙動を下げるための操作。
- **addition**: $\tau_\textrm{new}=\sum_i\tau_i$。複数 task を 1 つのモデルにまとめる、または別 task の vector で target task を改善する操作。
- **analogy**: $\tau_C+(\tau_B-\tau_A)$。3 つの task vectors から、類推関係にある 4 つ目の task を改善する操作。
- **scaling coefficient $\lambda$**: task vector をどの強さで適用するかを決める係数。held-out validation data で選ぶ。
- **control task**: 編集対象外の挙動が壊れていないかを見る task。画像では ImageNet、言語では WikiText-103 perplexity が使われる。
- **normalized accuracy**: task ごとの accuracy を、その task に fine-tuned した specialized model の accuracy で割った値。複数 task の難しさを揃えて平均するために使う。
- **open-ended models**: 新しい task-specific parameters を入れずに downstream task へ fine-tune できるモデル。本論文の task vector 操作と相性がよい。
- **gradient ascent baseline**: 忘却 baseline。通常の cross-entropy loss を下げるのではなく、loss を上げる方向に fine-tune する。target は下げられるが control を壊しやすい。
- **random vector baseline**: layer ごとに task vector と同じ norm を持つ乱数 vector。task vector の効果が単なる重み摂動ではないことを確認するために使う。
- **RealToxicityPrompts / Detoxify / Perspective API**: text generation の toxicity を測るための評価資源・分類器。主実験は Detoxify による 1000 generations、Appendix は RealToxicityPrompts と Perspective API も使う。

## 読む順番の提案

- まず正規ノート `notes/arXiv-2212.04089v3.md` の `Summary（著者の主張）` と `Notes / Quotes` で、3 つの操作と主要数値を把握する。
- 次に原論文の `02_task_vectors.tex` を読み、$\tau_t = \theta_\textrm{ft}^t - \theta_\textrm{pre}$、$\theta_\textrm{new}=\theta+\lambda\tau_\textrm{new}$、negation/addition/analogy の定義を確認する。Figure `fig:main` が全体図。
- 実験は `03_negation.tex` の Table `tab:forget_image` と `tab:toxicity` から読む。正規ノートの Summary の negation 項目と対応する。
- 続いて `04_addition.tex` の Figure `fig:clip-add-2`, Figure `fig:clip-add-all`, Table `tab:glue` を読む。正規ノートの Takeaway の「task vector は同じ initialization から...」と「HF Hub checkpoint」の話につながる。
- `05_analogies.tex` では sentiment の具体式、Table `tab:sentiment-analog`、Figure `fig:clip-analogies` を読む。Appendix の subpopulation Table `tab:sketches` は Figure の内訳。
- 最後に `06_discussion.tex` の Figure `fig:cossim`, `fig:lr`, `fig:intermediate` と Limitations を読む。正規ノートの `Critical Thoughts` と照合すると、仮定、評価設計、限界が見やすい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2212.04089v3/`
- 正規ノート: `notes/arXiv-2212.04089v3.md`
