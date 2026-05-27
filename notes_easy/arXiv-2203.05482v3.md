# Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time（大規模事前学習モデルの fine-tuning 後選択を重み平均で置き換える研究）

- arXiv: https://arxiv.org/abs/2203.05482
- 一次ソース: ../papers/arXiv-2203.05482v3/
- 正規ノート: ../notes/arXiv-2203.05482v3.md

---

## 一言で言うと

大規模事前学習モデルを複数の hyperparameter configuration で fine-tune した後、検証セットで最良の 1 モデルだけを選ぶ代わりに、複数モデルの weights を平均して 1 つの `model soup` を作る論文。特に `greedy soup` は推論時の compute / memory cost を単一モデルと同じ $\mathcal{O}(1)$ に保ちながら、CLIP、ALIGN、ViT-G/14、BASIC-L、GLUE の一部で best individual model を上回ると著者は主張する。

## 何を議論する論文か

- **問題設定**: fine-tuning では通常、同じ pre-trained initialization から複数の hyperparameter configuration を試し、held-out validation set で最高 accuracy のモデルだけを採用する。この手順は他の fine-tuned models を捨てるため、複数モデルの相補性を使えない。一方、logit ensemble は accuracy を改善しうるが、推論時に各モデルを別々に走らせるため cost が $\mathcal{O}(k)$ になる。
- **対象範囲 / 仮定**: 主対象は、大規模で多様なデータにより pre-train されたモデルを downstream task に end-to-end fine-tune する設定である。本文の中心実験は CLIP ViT-B/32、ALIGN EfficientNet-L2、JFT-3B pre-trained ViT-G/14 を ImageNet に fine-tune する設定で、追加実験として BASIC-L、WILDS-FMoW/iWildCam、CIFAR-10、ImageNet-22k pre-trained ViT-B/32、GLUE text classification、cross-dataset soups を扱う。
- **既存研究との差分**: SWA や EMA は同一 training trajectory 上の weights を平均するが、本論文は同じ initialization から独立に fine-tune された複数 run、しかも異なる hyperparameters を持つ run を平均する。Neyshabur et al. 2020 の「同じ pre-trained initialization から fine-tune された解は同じ error basin にある」という観察を背景に、hyperparameter sweep の副産物を再利用する。
- **この論文で答えたい問い**: validation で best individual model を選ぶ代わりに weights を平均しても accuracy と robustness は改善するか。どのような soup construction が安定するか。weight averaging が logit ensembling に近い振る舞いをする条件は何か。適用範囲と限界はどこにあるか。

## 背景と前提

- 論文での fine-tuning は、標準的な neural network training と同様に loss を最小化するが、parameters を random initialization ではなく pre-training で得た $\theta_0$ から始める点が重要である。TeX の Method では $\theta = \mathsf{FineTune}(\theta_0,h)$ と書かれる。
- `model soup` は、network architecture が同じで、対応する parameter vector を要素ごとに平均できるモデル群を前提とする。Method では $f(x,\theta)$、$\theta\in\mathbb{R}^d$ と置き、同じ shared initialization から得た $\theta_i$ を平均する。
- 本論文の baseline は大きく 3 種類ある。第 1 は conventional recipe の `Best on val. set`、第 2 は logits を平均する `Ensemble` / `Greedy ensemble`、第 3 は related method として distillation、fix-augmentation、SWA / EMA、SAM、WiSE-FT などである。
- ImageNet fine-tuning では、official ImageNet validation set は通常 test set として使われるため、論文は ImageNet training set の約 2% を held-out validation set として取り、greedy soup construction に使う（`sec:setup`, `app:mainres`）。
- robustness 評価では ImageNetV2、ImageNet-R、ImageNet-Sketch、ObjectNet、ImageNet-A の 5 つを natural distribution shifts として扱い、しばしばこれらの平均 accuracy を `Avg shifts` または `Dist. shifts` として報告する。
- CLIP / ALIGN / BASIC は image-text pairs で contrastive supervision により pre-train されたモデルとして扱われる。ViT-G/14 は JFT-3B で pre-train されたモデルで、BERT / T5 は text classification 実験の transformer models として使われる。
- fine-tuned models を weight space で平均しても壊れにくい理由として、Introduction は Neyshabur et al. 2020 を挙げる。ただし、uniform soup は高 learning rate などで低 accuracy のモデルを含むと壊れる場合があり、これを避けるために greedy soup が導入される。

## 提案手法

### コアアイデア

入力 $x$ と parameters $\theta$ を持つ network $f(x,\theta)$ を考える。まず、同じ pre-trained initialization $\theta_0$ から、異なる hyperparameter configurations $h_1,\ldots,h_k$ で fine-tune し、$\theta_i=\mathsf{FineTune}(\theta_0,h_i)$ を得る。従来は held-out validation set accuracy が最も高い $\theta_j$ を選び、残りを捨てる。本論文は、選ばれた subset $\mathcal{S}$ の weights を平均した $\theta_{\mathcal{S}}$ を 1 つの model として使う。

提案される soup は 3 種類である。`uniform soup` は全 fine-tuned models を単純平均する。`greedy soup` は validation accuracy の降順に candidate models を並べ、加えたときに held-out validation set accuracy が下がらないモデルだけを ingredient として採用する。`learned soup` は validation set 上で interpolation weights を gradient-based minibatch optimization により学習するが、全モデルを同時に memory に載せる必要があるため large networks では使いにくいと書かれている。

Table `tab:methods` は、best-on-validation と uniform / greedy / learned soup の inference cost を $\mathcal{O}(1)$、ensemble の cost を $\mathcal{O}(k)$ と整理する。著者の狙いは、ensemble が持つ複数モデル利用の利点に近づきつつ、推論時には 1 モデルだけを実行することである。

### 重要な定義・数式

$$
\theta_i = \mathsf{FineTune}\left(\theta_0, h_i\right),
\qquad
\theta_{\mathcal{S}} = \frac{1}{|\mathcal{S}|}\sum_{i \in \mathcal{S}} \theta_i
$$

**式の意味**: Method の model soup 定義である。同じ pre-trained initialization $\theta_0$ から、異なる hyperparameter configuration $h_i$ で fine-tune して得た $\theta_i$ を、subset $\mathcal{S}$ 上で要素ごとに平均する。

**記号の定義**:
- $\theta_0$ ... pre-training により得られた shared initialization。
- $h_i$ ... optimizer、data augmentation、training iterations、random seed などを含みうる hyperparameter configuration。
- $\theta_i$ ... $\theta_0$ から $h_i$ で fine-tune して得た parameters。
- $\mathcal{S}$ ... soup に入れるモデル index の subset。
- $|\mathcal{S}|$ ... soup ingredients の数。
- $\theta_{\mathcal{S}}$ ... averaged weights。実際の推論では $f(x,\theta_{\mathcal{S}})$ を 1 モデルとして使う。

**この論文での役割**: 提案手法そのものの定義である。uniform soup は $\mathcal{S}=\{1,\ldots,k\}$、greedy soup は Recipe `alg:greedy` で選ばれた $\mathcal{S}$ を使う。

$$
\begin{aligned}
\text{Best on val. set} &: f\left(x,\argmax_i \mathsf{ValAcc}\left(\theta_i\right)\right) \\
\text{Ensemble} &: \frac{1}{k}\sum_{i=1}^k f\left(x,\theta_i\right) \\
\text{Uniform soup} &: f\left(x,\frac{1}{k}\sum_{i=1}^k \theta_i\right)
\end{aligned}
$$

**式の意味**: Table `tab:methods` の比較である。従来法は validation accuracy が最大の model を 1 つ選ぶ。Ensemble は outputs / logits を平均する。Uniform soup は parameters を平均してから network を 1 回だけ実行する。

**記号の定義**:
- $f(x,\theta_i)$ ... input $x$ に対する $i$ 番目の fine-tuned model の出力。
- $\mathsf{ValAcc}(\theta_i)$ ... held-out validation set 上の accuracy。
- $k$ ... hyperparameter sweep で得た candidate models の数。

**この論文での役割**: model soup が ensemble とどこで異なるかを明確にする式である。ensemble は推論時 cost が $\mathcal{O}(k)$ だが、uniform soup は parameter averaging 後に 1 モデルだけを使うため $\mathcal{O}(1)$ になる。

$$
\mathsf{ValAcc}\left(\mathsf{average}\left(\mathsf{ingredients} \cup \{\theta_i\}\right)\right)
\geq
\mathsf{ValAcc}\left(\mathsf{average}\left(\mathsf{ingredients}\right)\right)
$$

**式の意味**: Recipe `alg:greedy` の採用条件である。candidate $\theta_i$ を現在の ingredients に加えた平均モデルが、加える前の平均モデルより held-out validation accuracy を下げない場合だけ、$\theta_i$ を soup に残す。

**記号の定義**:
- $\mathsf{ingredients}$ ... greedy soup の途中で採用済みの model set。
- $\theta_i$ ... validation accuracy 降順に見ている $i$ 番目の candidate model。
- $\mathsf{average}(\cdot)$ ... 与えられた parameters の要素ごとの平均。
- $\mathsf{ValAcc}(\cdot)$ ... training set と test set から disjoint な held-out validation set 上の accuracy。

**この論文での役割**: uniform soup が低 accuracy model や別 basin の model に壊されるリスクを避けるための中核手順である。候補を validation accuracy 降順に sort するため、TeX は greedy soup が held-out validation set 上で best individual model より悪くならないと述べる。

$$
\mathcal{L}^{\mathrm{soup}}_\alpha - \mathcal{L}^{\mathrm{ens}}_\alpha
\approx
\frac{\alpha(1-\alpha)}{2}
\left(
-\frac{\mathrm{d}^2}{\mathrm{d}\alpha^2}\mathcal{L}^{\mathrm{soup}}_\alpha
+ \beta^2 \mathbb{E}_{x}\,
\mathrm{Var}_{Y\sim p_\mathrm{sftmx}(\beta f(x;\theta_\alpha))}
\left[\Delta f_Y(x)\right]
\right)
$$

**式の意味**: Section `sec:theory` の Eq. `eq:approx` で、2-model soup と logit-level ensemble の cross-entropy loss の差を近似する式である。差は、soup loss の interpolation path に沿った 2 階微分と、endpoint logits の差の softmax 分布下での variance で表される。

**記号の定義**:
- $\theta_\alpha=(1-\alpha)\theta_0+\alpha\theta_1$ ... 2 つの endpoint models の weight-averaged soup。
- $\mathcal{L}^{\mathrm{soup}}_\alpha$ ... $\mathbb{E}_{x,y}\ell(\beta f(x;\theta_\alpha),y)$ として定義される soup の $\beta$-calibrated expected loss。
- $\mathcal{L}^{\mathrm{ens}}_\alpha$ ... $(1-\alpha)f(x;\theta_0)+\alpha f(x;\theta_1)$ を使う logit-level ensemble の expected loss。
- $\alpha\in[0,1]$ ... 2 endpoints の interpolation weight。
- $\beta$ ... inverse-temperature parameter。empirical evaluation では soup model を calibrate するように選ばれる。
- $p_\mathrm{sftmx}$ ... standard softmax distribution。TeX では $[\mathrm{softmax}(f)]_i=e^{f_i}/\sum_j e^{f_j}$。
- $\Delta f(x)=f(x;\theta_1)-f(x;\theta_0)$ ... endpoint logits の差。

**この論文での役割**: soup が ensemble に近い performance を持つ条件を説明するための解析である。TeX は第 1 項を loss trajectory の flatness、第 2 項を endpoint prediction difference と confidence に関わる項として解釈する。variance term は endpoint models が似ている場合、または soup の予測が confident で softmax が point mass に近い場合に小さくなる。

### 実装 / アルゴリズム上の要点

- step1: shared pre-trained initialization $\theta_0$ を用意する。本文では CLIP、ALIGN、BASIC、JFT-3B pre-trained ViT-G/14、BERT、T5 などを使う。
- step2: 同じ $\theta_0$ から複数の hyperparameter configurations で end-to-end fine-tuning し、candidate parameters $\theta_1,\ldots,\theta_k$ を得る。
- step3: `uniform soup` では全 candidate parameters を平均する。追加学習は不要だが、高 learning rate などで低 accuracy の model が含まれると低 accuracy soup になる場合がある。
- step4: `greedy soup` では、candidate models を $\mathsf{ValAcc}(\theta_i)$ の降順に sort し、Recipe `alg:greedy` の条件を満たす場合だけ ingredients に加える。CLIP と ALIGN では greedy soup はそれぞれ 5 models を選ぶ。ViT-G/14 では 58 models から 14 models を選ぶ。
- step5: `learned soup` では、validation set $\{(x_j,y_j)\}_{j=1}^n$ 上で mixing coefficients $\alpha\in\mathbb{R}^k$ と temperature $\beta$ を最適化する。TeX は $\alpha$ を softmax output として正かつ和が 1 になるようにする方が良かったと述べる。ただし全モデルを memory に同時に載せる必要があり、large networks では実用上の制約になる。
- step6: 完成した soup は通常の 1 モデルとして保存・評価する。Table `tab:methods` では inference memory / compute は single model relative cost で $\mathcal{O}(1)$ とされる。
- step7: ViT-G/14 と BASIC-L では low EMA と high EMA を保存する。TeX は high EMA が best single-model accuracy に有利な一方、greedy soup と greedy ensemble は low EMA weights の方が higher validation accuracy を得たと述べる。

## 実験・結果

- **データセット / ベンチマーク**: 主な image benchmark は ImageNet と、natural distribution shifts としての ImageNetV2、ImageNet-R、ImageNet-Sketch、ObjectNet、ImageNet-A。追加で ReaL と multilabel relabeled ImageNet validation sets、WILDS-FMoW、WILDS-iWildCam、CIFAR-10 / CIFAR-10.1、CIFAR-100 zero-shot、GLUE の MRPC / RTE / CoLA / SST-2 を使う。
- **比較対象 / baseline**: `Best individual model` / `Best model on held out val set`、`Best model on each test set (oracle)`、`Uniform soup`、`Greedy soup`、`Learned soup`、`Learned soup (by layer)`、`Ensemble`、`Greedy ensemble`。追加 baseline として distillation from ensemble、fix-aug、EMA / SWA、SAM、WiSE-FT が付録で比較される。
- **指標**: ImageNet は top-1 accuracy が中心で、ViT-G/14 / BASIC-L では ReaL、Multilabel、各 distribution shift、Avg shifts も報告する。CLIP の Table `tab:results` は `ImageNet` と `Dist. shifts`。GLUE は MRPC が accuracy と $F_1$ の平均、RTE と SST-2 が accuracy、CoLA が Matthews correlation。
- **CLIP ViT-B/32 on ImageNet**: random hyperparameter search で 72 fine-tuned models を作る。Table `tab:results` では Best individual model が 80.38 / 47.83（ImageNet / Dist. shifts）、Uniform soup が 79.97 / 51.45、Greedy soup が 81.03 / 50.75、Learned soup が 80.89 / 51.07、Learned soup (by layer) が 81.37 / 50.87、Ensemble が 81.19 / 50.77、Greedy ensemble が 81.90 / 49.44。
- **ALIGN EfficientNet-L2 on ImageNet**: grid search で 12 fine-tuned models を作る。本文は greedy soup が best model in the hyperparameter sweep を 0.5 percentage points 改善したと述べる。CLIP では同じ箇所で 0.7 percentage points 改善したとされる。
- **ViT-G/14 pre-trained on JFT-3B**: 58 models を fine-tune し、greedy soup は 14 models を選ぶ。Table `tab:vit_g_finetuned_result` では Best model on held out val set が Top-1 90.72、Avg shifts 84.38、oracle が Top-1 90.78、Avg shifts 84.68、Greedy ensemble が Top-1 90.93、Avg shifts 84.33、Greedy soup が Top-1 90.94、Avg shifts 85.02。Introduction は 90.94% が CoAtNet-7 の 90.88% を上回り、inference FLOPs は 25% 少ないと述べる。
- **BASIC-L**: Appendix `app:basic` では 20 models を fine-tune し、Table `tab:basic_finetuned_result` で Best model on held out val set が Top-1 90.83、Avg shifts 85.63、Greedy ensemble が 91.02 / 86.20、Greedy soup が 90.98 / 86.40、oracle が 90.87 / 86.54。Introduction の footnote は、この 90.98% が CoCa の reported precision と tie すると述べる。
- **NLP / GLUE**: Section `sec:nlp` では BERT と T5 を MRPC、RTE、CoLA、SST-2 に fine-tune する preliminary experiments として扱う。Table `tab:nlp` では BERT が 88.3→88.3、61.0→61.7、59.1→59.1、92.5→93.0、T5 が 91.8→92.4、78.3→79.1、58.8→60.2、94.6→94.7。Appendix Table `tab:nlp_full` では BERT-base / BERT-large / T5-small / T5-base / T5-large の 20 combinations 中 10 combinations で greedy soup が best individual model を上回ったと書かれている。
- **cross-dataset soups**: Appendix `sec:zsperf` では、CLIP zero-shot initialization と、CIFAR-10、Describable Textures、Food-101、SUN397、Stanford Cars、ImageNet で個別に fine-tune した 6 models を使い、CIFAR-100 zero-shot を評価する。各 task の class set が異なるため last layers は soup に入れず、CLIP text tower の linear head を freeze して backbone weights のみを soup にする。Figure `fig:samir` は CLIP baseline から CIFAR-100 accuracy が 6.4 percentage points 改善したと述べる。
- **robust fine-tuning / WiSE-FT**: Appendix `sec:robustft` は、fine-tuned model $\theta_1$ と initialization $\theta_0$ を線形補間する WiSE-FT と model soups を比較する。Figure `fig:scatter` の caption と本文は、uniform / greedy soups が individual model の WiSE-FT curves を越え、さらに soups と initialization を補間すると distribution shifts の accuracy が追加で改善すると述べる。
- **追加 dataset / pre-training scale**: Appendix `app:moresets` は WILDS-FMoW、WILDS-iWildCam、CIFAR-10/CIFAR-10.1、ImageNet-22k pre-trained ViT-B/32 を扱う。ImageNet-22k pre-trained ViT-B/32 では greedy soup は改善を与えるが、CLIP / ALIGN で観察された改善より less substantial とされる。random seed だけを変えた 5 runs の比較では、CLIP models の方が ImageNet-22k pre-trained models より ensembling と souping に向いていると本文は述べる。
- **著者が主張する貢献**: hyperparameter sweep 後の選択段階を置き換える simple recipe、追加 training なし・推論 cost 増なしの accuracy / robustness 改善、ViT-G/14 の ImageNet 90.94%、複数 domain への拡張、そして soup と ensemble の loss 差を flatness と confidence で説明する解析が主な貢献である。

## 妥当性と限界

- **この主張を支える根拠**: Table `tab:methods` は inference cost の違いを明示し、Table `tab:results`、`tab:vit_g_finetuned_result`、`tab:basic_finetuned_result`、`tab:nlp` は best individual model、soup、ensemble を同じ hyperparameter sweep の候補から比較している。ViT-G/14 と BASIC-L では exact McNemar test または permutation test at $\alpha=0.05$ により、best と有意差がない値を bold-faced にすると caption に書かれている。
- **この主張を支える根拠**: Figure `fig:error` は CLIP fine-tuning の 2D loss / error landscape で、endpoint models 自体より間の点が良くなりうることを示す。Figure `fig:angles` は interpolation advantage が、$\theta_1-\theta_0$ と $\theta_2-\theta_0$ の angle $\phi$ と相関し、learning rate、seed、data augmentation の違いがより orthogonal な solutions を生むことを示す。Figure `fig:ose_wse_compare` は soup performance と ensemble performance の相関を示す。
- **この主張を支える根拠**: Section `sec:theory` と Appendix `app:theory` は、2-model soup と logit ensemble の loss 差を Eq. `eq:approx` で近似する。Figure `fig:theory-eval` は、learning rate $10^{-4}$ を除くと approximation が true loss difference と error difference に strongly correlated し、true loss difference と sign も概ね一致すると述べる。
- **著者が認めている limitations / future work**: `sec:lim` は 2 つの limitation を明示する。第 1 は applicability で、実験の多くが large, heterogeneous datasets で pre-train された models に集中しており、ImageNet-22k pre-trained model では改善が less substantial である。第 2 は calibration で、ensembles は calibration を改善するが model soups do not have the same effect と書かれている。
- **著者が認めている limitations / future work**: NLP は preliminary experiments とされ、Section `sec:nlp` は more investigation is warranted と述べる。Appendix `app:nlp_ft` では、basic hyperparameters だけを変えた実験であり、broader set of hyperparameter choices が more diverse models and better soups につながる可能性を hypothesize している。
- **読者として注意すべき点**: greedy soup は validation set 上で candidate を評価しながら作るため、inference cost は $\mathcal{O}(1)$ でも、soup construction 時には候補数に応じた validation evaluation が必要である。Table `tab:methods` の cost は「during inference relative to a single model」と caption が限定している。
- **読者として注意すべき点**: uniform soup は常に安全ではない。本文は、すべての individual models が high accuracy の場合には best individual model を上回ることがある一方、高 learning rate models などでは error barrier が生じ、uniform soup の accuracy が下がりうると述べる。
- **読者として注意すべき点**: 理論近似は logits が linearly 近い領域を仮定している。Section `sec:theory` は high learning rate $10^{-4}$ の models が initialization から weight space で遠く、greedy soups でも reject されがちなので approximation が tight でないと述べ、Figure `fig:theory-eval` の中心・右パネルではそれらを除外している。
- **追加で確認したい実験 / 疑問**: 同じ総 training compute で、many-model hyperparameter sweep + soup と、少数モデルを長く・大きく train する戦略をどう比較するかは TeX 中では明示的な主実験ではない。calibration を改善しない点について、temperature scaling や ensemble と組み合わせたときの実運用上の trade-off も追加で確認したい。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **model soup**: 同じ pre-trained initialization から fine-tune された複数モデルの parameters を平均して得る 1 つの model。推論時は通常の single model と同じように $f(x,\theta_{\mathcal{S}})$ を実行する。
- **ingredient**: greedy soup の Recipe `alg:greedy` で、現在の soup に入っている採用済みモデル、または採用候補のモデルを指す語。
- **uniform soup**: hyperparameter sweep で得た全 models を平均する soup。$\mathcal{S}=\{1,\ldots,k\}$。
- **greedy soup**: validation accuracy 降順に models を試し、追加しても held-out validation accuracy が下がらないものだけを採用する soup。本論文の central method とされる。
- **learned soup**: validation set 上で mixing coefficients $\alpha$ と temperature $\beta$ を gradient-based optimization する soup。全 models を memory に同時に載せる必要がある。
- **Best on val. set**: conventional recipe の第 2 段階。$\argmax_i\mathsf{ValAcc}(\theta_i)$ に対応する model を 1 つ選び、残りを捨てる。
- **logit ensemble**: 複数モデルの logits、つまり unnormalized outputs を平均する ensemble。Table `tab:methods` では cost が $\mathcal{O}(k)$。
- **held-out validation set**: training set と test set から分離された validation data。ImageNet 実験では training set の約 2% を held-out validation set として使う。
- **distribution shifts / natural distribution shifts**: ImageNet fine-tuning 後の OOD 評価として使う ImageNetV2、ImageNet-R、ImageNet-Sketch、ObjectNet、ImageNet-A。
- **interpolation advantage**: $\mathsf{Acc}(\frac{1}{2}\theta_1+\frac{1}{2}\theta_2)-\frac{1}{2}(\mathsf{Acc}(\theta_1)+\mathsf{Acc}(\theta_2))$。Figure `fig:angles` で angle $\phi$ との相関を見る量。
- **angle $\phi$**: $\theta_1-\theta_0$ と $\theta_2-\theta_0$ の間の angle。initialization $\theta_0$ を origin とみなす。
- **low error basin / error barrier**: 同じ initialization から得た fine-tuned solutions の間を線形補間したとき、loss / error が低いまま保たれる領域を basin と呼ぶ。一方、高 learning rate では solutions 間に error barrier が生じる場合がある。
- **LP initialization**: final linear layer を linear probe から初期化する方法。CLIP / ALIGN の fine-tuning setup で使われ、zero-shot initialization と比較される。
- **zero-shot initialization**: CLIP や ALIGN の text tower から作られる classifier を final linear layer の initialization として使う方法。
- **WiSE-FT**: fine-tuned model $\theta_1$ と initialization $\theta_0$ を線形補間して robustness を改善する方法。本論文では model soups と組み合わせ可能な robust fine-tuning baseline として扱われる。
- **calibration / ECE**: model の confidence と実際の正解率の対応を見る性質。Figure `fig:cal` では Expected Calibration Error を equal-mass binning で計算し、soups は ensembles と同じ calibration 改善を示さないとされる。
- **EMA / SWA**: 同一 optimization trajectory 上で weights を平均する手法。本論文の model soup は独立 run 間の averaging であり、Appendix `app:baselines` は EMA / SWA と additive な改善を示すと述べる。

## 読む順番の提案

- まず `main.tex` の Abstract と Introduction を読む。conventional recipe、discarded models、ensemble cost、fine-tuned models in a single low error basin、90.94% top-1 という論文の問題設定と主張がまとまっている。正規ノートでは `Summary（著者の主張）` の問題・手法・結果に対応する。
- 次に Method `sec:method`、Table `tab:methods`、Recipe `alg:greedy` を読む。$f(x,\theta)$、$\theta_i=\mathsf{FineTune}(\theta_0,h_i)$、$\theta_{\mathcal{S}}$、uniform / greedy / learned soup、inference cost の違いを押さえる。正規ノートの `Takeaway` にある greedy soup の「val 最良単体より悪くならない」という点につながる。
- その後 Experiments `sec:setup` と `sec:error`、Figure `fig:error`、`fig:angles`、`fig:ose_wse_compare` を読む。なぜ weight averaging が壊れない場合があるのか、angle $\phi$、learning rate、ensemble との関係を確認できる。
- Main results `sec:mainres` では Table `tab:results` と Table `tab:vit_g_finetuned_result` を先に見る。CLIP の 80.38→81.03、ViT-G/14 の 90.72→90.94 / Avg shifts 84.38→85.02 が、正規ノートの主要数値と対応する。
- NLP は `sec:nlp` と Table `tab:nlp`、詳細は Appendix `app:nlp_ft` と Table `tab:nlp_full` を読む。本文が preliminary experiments と限定している点を確認する。
- 解析は `sec:theory` の Eq. `eq:approx` と Appendix `app:theory` を読む。正規ノートの ensemble との理論的関係、flatness、confidence、high learning rate $10^{-4}$ 除外の説明に対応する。
- 限界は `sec:lim`、Figure `fig:cal`、Appendix `app:moresets` の ImageNet-22k pre-trained ViT-B/32、Appendix `sec:robustft`、Appendix `sec:zsperf` を読む。正規ノートの limitations、WiSE-FT、cross-dataset soup の項目につながる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2203.05482v3/`
- 正規ノート: `notes/arXiv-2203.05482v3.md`
