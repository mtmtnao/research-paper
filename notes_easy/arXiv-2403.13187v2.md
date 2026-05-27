# Evolutionary Optimization of Model Merging Recipes（進化計算によるモデルマージレシピの自動最適化）

- arXiv: https://arxiv.org/abs/2403.13187
- 一次ソース: ../papers/arXiv-2403.13187v2/
- 正規ノート: ../notes/arXiv-2403.13187v2.md

---

## 一言で言うと

この論文は、既存の公開 foundation models を追加の勾配学習なしに組み合わせる model merging を、人手の「black art or alchemy」ではなく進化計算で探索する枠組みとして定式化する。著者らは Parameter Space (PS) と Data Flow Space (DFS) の 2 つの探索空間を使い、日本語数学 LLM と日本語 VLM を作り、MGSM-JA、JP-LMEH、JA-VG-VQA-500、JA-VLM-Bench-In-the-Wild で有効性を示す。

## 何を議論する論文か

- **問題設定**: model merging は複数の学習済みモデルを組み合わせて新しいモデルを作る方法で、追加の gradient-based training を必要としない。一方で、どの source model を選び、どの merging recipe と係数を使い、どの層を積むかは人間の直感と domain knowledge に強く依存している、という問題を扱う（Introduction の "black art or alchemy"）。
- **対象範囲 / 仮定**: PS では同一 architecture の fine-tuned descendants を重み空間で混ぜる。主実験の 7B LLM では `shisa-gamma-7b-v1`、`WizardMath-7B-V1.1`、`Abel-7B-002` がすべて `Mistral-7B-v0.1` 由来である。DFS では重みを固定し、tokens が通る inference path を serial connections かつ non-adaptive configurations に限定して探索する（Methods, `Merging in the Data Flow Space`）。
- **既存研究との差分**: Model Soups、Task Arithmetic、TIES-Merging、DARE、mergekit の Frankenmerging は既存の merging recipe だが、この論文は merging configuration parameters や layer path を CMA-ES で自動探索する。NAS と異なり、候補 architecture ごとの再学習は行わず、既存 Transformer blocks をそのまま評価する点が差分である。
- **この論文で答えたい問い**: 進化計算は、異なる能力を持つ公開モデルを組み合わせて、source model 単体より良い foundation model を作れるか。特に non-English language と Math、non-English language と Vision のような cross-domain merging で、人間設計より有効な組み合わせを見つけられるか。

## 背景と前提

- model merging は、fine-tuned models の重みや task vectors を組み合わせることで、複数タスクの能力を単一モデルに統合しようとする方法である。fine-tuning と違い、既存モデルをさらに学習するのではなく、既存モデルの weights や layers の組み合わせを直接作る。
- この論文での **layer** は、"the input/output embedding layers or a transformer block" を指す。つまり Transformer block だけでなく、入力・出力 embedding も merging configuration の対象に含む。
- PS は weights を混ぜる空間であり、著者らは TIES-Merging と DARE を組み合わせ、sparsification と weight mixing の設定を進化計算で最適化する。報告された PS model は、Analysis で述べられるように、各 source model を単一 layer とみなし、各 model に 2 つの DARE-TIES 関連パラメータを割り当てる最も単純な設定である。
- DFS は weights を変えず、tokens の通る層の順序と選択を変える空間である。Frankenmerging に近いが、手動 recipe ではなく indicator array $\mathcal{I}$ と scaling matrix $W$ を進化計算で探索する。
- 比較対象は、未最適化の TIES-Merge、DARE-TIES、Frankenmerging、fine-tuning baseline として LoRA / full fine-tuning、さらに Llama 2 70B、Japanese StableLM 70B、Swallow 70B、GPT-3.5、GPT-4 などである（Table~\ref{table:math}, Table~\ref{table:math-baselines}）。
- VLM では、vision encoder と projection network は固定し、VLM 内の LLM component を standalone LLM と見なして merging する。source は `LLaVA-1.6-Mistral-7B` と `shisa-gamma-7b-v1` で、両方とも `Mistral-7B-v0.1` の fine-tunes である。

## 提案手法

### コアアイデア

著者らの **Evolutionary Model Merge** は、モデルを混ぜる recipe を探索対象として、タスク別の評価指標で fitness を測り、CMA-ES によって merging configuration parameters や inference path を最適化する枠組みである。論文中では、merging process を 2 つの orthogonal configuration spaces に分解する。

1. **Merging in the Parameter Space (PS)**: 複数モデルの weights を、TIES-Merging と DARE に基づいて混ぜる。探索対象は sparsification と weight mixing の configuration parameters であり、MGSM なら accuracy、VQA なら ROUGE score のような task-specific metrics で最適化する。
2. **Merging in the Data Flow Space (DFS)**: 各 layer の weights は固定し、tokens がどの model のどの layer をどの順序で通るかを探索する。層の接続を変えると hidden states の分布がずれるため、層 $i$ から層 $j$ へ渡すときの scaling matrix $W_{ij}$ も同時に最適化する。
3. **Merging in Both Spaces**: まず PS で merged model を作り、そのモデルを source pool に戻して DFS を行う。Japanese Math LLM では Table~\ref{table:math} の Model 6 がこの PS+DFS に当たる。

この設計の重要点は、model merging では候補モデルを新たに training しないため、NAS よりも候補評価が軽いことである。著者らはこれを、既存 Transformer blocks を部品として再利用する探索問題として位置づけている。

### 重要な定義・数式

$$
\theta_\text{new} = \lambda \theta_1 + (1 - \lambda) \theta_2
$$

**式の意味**: 2 つのモデルの weight vectors を線形補間して新しいモデルの weights を作る、最も単純な model merging の式である。Background の "Linear weight averaging is performed as follows" に対応する。

**記号の定義**:
- $\theta_1, \theta_2 \in \mathbb{R}^d$ ... 2 つの distinct models の weight vectors
- $d$ ... weight space の次元
- $\theta_\text{new}$ ... merged model の weights
- $\lambda \in [0, 1]$ ... 2 つのモデルの relative contribution を決める weighting parameter

**この論文での役割**: PS merging の背景となる基本操作である。ただし本論文の主手法は単純な linear averaging だけではなく、Task Arithmetic、TIES-Merging、DARE を組み合わせた recipe のパラメータを進化計算で最適化する。

$$
\tau_k = \theta_k - \theta_\text{base}, \qquad
\theta_\text{new} = \theta_\text{base} + \sum_k \lambda_k \tau_k
$$

**式の意味**: fine-tuned model と base model の差分を task vector として取り出し、それらを base model に重み付きで足し戻す Task Arithmetic の定義である。

**記号の定義**:
- $\theta_\text{base} \in \mathbb{R}^d$ ... pre-trained base model の weights
- $\theta_k \in \mathbb{R}^d$ ... task $k$ 用に fine-tuned された model の weights
- $\tau_k$ ... task $k$ の task vector
- $\lambda_k$ ... task vector $k$ の contribution を決める scaling parameter

**この論文での役割**: PS merging で各 source model の強みを weight difference として扱うための前提である。TIES-Merging と DARE は、この task vector の干渉を減らしながら merge する方法として使われる。

$$
\hat{\tau}_k = \frac1{1 - p} \left( {(1 - m_k) \odot \tau_k} \right),
\qquad m_k \sim \text{Bernoulli}(p)
$$

**式の意味**: DARE の sparsification で、task vector $\tau_k$ に random mask をかけ、残った成分を $1/(1-p)$ で rescale する。TeX では "DARE operates as follows" として示されている。

**記号の定義**:
- $\tau_k$ ... task vector
- $m_k$ ... Bernoulli$(p)$ に従う random mask
- $p$ ... drop rate
- $\odot$ ... 要素ごとの積
- $\hat{\tau}_k$ ... sparsified and rescaled task vector

**この論文での役割**: 主実験の PS では、TIES-Merging と DARE を組み合わせた DARE-TIES の parameters を CMA-ES で最適化する。Analysis では、日本語 LM の density が大きくなったことを、長い continued pretraining を受けたモデルでは DARE の sparsification が性能を落としうるという DARE 論文 Section 4.6 と結びつけて説明している。

$$
(M+1)^T \;\longrightarrow\; 2^T, \qquad T = M \times r
$$

**式の意味**: DFS で、全 layer から任意の layer または pass-through を長さ $T$ で選ぶ素朴な探索空間 $(M+1)^T$ を、順番に並べた layers を $r$ 回繰り返し、各 slot を使うかどうかだけを indicator array で選ぶ探索空間 $2^T$ に縮小する。

**記号の定義**:
- $M$ ... 全 source models に含まれる layers の総数
- $T$ ... inference path の長さ、または indicator array のサイズ
- $r$ ... layer sequence を何回繰り返すか
- $M+1$ ... $M$ 個の layers に pass-through layer を加えた選択肢数

**この論文での役割**: DFS を進化計算で扱える規模にするための中心的な探索空間設計である。7B LLM 実験では $M=64, r=3$ なので $T=192$ とされる。

$$
W_{ij}=\pi_{\theta}(i, j, t)
$$

**式の意味**: layer $i$ から layer $j$ へ hidden states を渡すときの scaling weight $W_{ij}$ を、layer index と path step に条件づけた feed-forward network $\pi_\theta$ で出力する代替案である。TeX では、$W$ が $M$ に対して二乗で大きくなる場合の search space 抑制策として述べられている。

**記号の定義**:
- $W_{ij}$ ... layer $i$ から layer $j$ へ渡す入力の scaling factor
- $\pi_\theta$ ... scaling weights を出力する feed-forward network
- $\theta$ ... $\pi_\theta$ の進化対象 parameters
- $i, j$ ... layer indices
- $t$ ... inference path 内の step index

**この論文での役割**: DFS では、別モデルの layer に渡される hidden states の distribution shift が問題になる。著者らは実験では scaling matrix $W \in \mathcal{R}^{M \times M}$ を $\mathcal{I}$ と同時に最適化し、$W_{ij}=1$ に固定すると 10B PS+DFS で 20 percent を超える性能低下が起きると報告している（Analysis, Figure~\ref{fig:dfs_analysis}）。

### 実装 / アルゴリズム上の要点

- **PS optimization**: CMA-ES を Optuna 実装で用いる。初期値はすべて 0.5、sigma は $1/6$、population size は $4 + \lfloor 3 \ln(n_{\text{params}}) \rfloor$。fitness は 1069 個の training samples 全体の accuracy で、1000 trials の後、training accuracy が最良の trial を final model とする。
- **PS の具体設定**: Japanese Math LLM の報告モデルでは、`shisa-gamma-7b-v1`、`WizardMath-7B-V1.1`、`Abel-7B-002` を source とし、TIES-Merging with DARE の parameters を最適化する。Analysis では、複雑な layer grouping で目立つ改善がなかったため、各 source model を singular layer と見なして各 model に 2 つの DARE-TIES parameters を割り当てたと説明される。
- **DFS optimization**: 7B LLM 実験では $M=64, r=3, T=192$。training data の最後の 200 examples を validation set とし、残りを batch size 200 で最適化する。EvoJAX 上の CMA-ES で $\mathcal{I}$ と $W$ を 100 generations、population size 128、default hyper-parameters で最適化する。
- **DFS の制約**: memory と実行可能性のため DFS は 2 models $A$ and $B$ に限定される。model $A$ の tokenizer と input/output embeddings を使い、embedding layers との互換性のため、model $A$ の initial and final transformer layers が inference path の start と end を定義する。
- **DFS の初期化**: model $A$ の layers が初期 hops に含まれやすいように indicator array $\mathcal{I}$ を初期化する。13B analysis では、$\mathcal{I}$ を zeros にしつつ、first repetition の model $A$ layers に対応する値を $2\sigma$ に設定すると説明される。
- **PS+DFS**: 先に PS merged model を作り、そのモデルを source pool に戻して DFS を行う。Japanese Math では Table~\ref{table:math} の `Ours (PS+DFS)` が `Ours (PS)` と `Shisa Gamma 7B v1` の組み合わせである。
- **VLM extension**: VLM の LLM component だけを merging 対象とし、vision encoder と projection network は固定する。PS では `shisa-gamma-7b-v1` と `LLaVA-1.6-Mistral-7B` を TIES-Merging with DARE で merge する。DFS では `LLaVA 1.6 Mistral 7B` を model A、`shisa-gamma-7b-v1` を model B とし、PS+DFS では PS-merged model を model A とする。

## 実験・結果

- **データセット / ベンチマーク**: Japanese Math では MGSM Japanese test set 250 samples を final evaluation に使う。最適化には、MGSM に含まれない GSM8k test set の remaining 1069 samples、すなわち IDs 250-1318 を日本語訳したものを使い、MGSM Japanese test set とは disjoint にする。日本語一般能力は JP-LMEH の 9 tasks 平均で評価する。VLM では JA-VG-VQA-500 と JA-VLM-Bench-In-the-Wild を使い、後者は 42 images と 50 questions からなる日本文化要素を含む benchmark である。
- **比較対象 / baseline**: Japanese Math では source models である `shisa-gamma-7b-v1`、`WizardMath-7B-V1.1`、`Abel-7B-002`、未最適化 model merging の TIES-Merge、DARE-TIES、Frankenmerging、fine-tuning の LoRA / full、reference models の Llama 2 70B、Japanese StableLM 70B、Swallow 70B、GPT-3.5、GPT-4 と比較する。VLM では `LLaVA-1.6-Mistral-7B`、`Japanese Stable VLM`、TIES、DARE-TIES、Passthrough と比較する。
- **指標**: MGSM-JA は zero-shot pass@1 accuracy で、最後に現れる数値を answer とし、fasttext で出力言語を判定して日本語 reasoning であることも条件にする。JP-LMEH は 9 tasks の平均。VLM は ROUGE-L で、非日本語応答は fasttext により empty texts に置換して score zero とする。ただし ground-truth answer 自体に `UFO` のような日本語文中で一般的な非日本語語が含まれる場合は例外とする。
- **主な結果 1, Japanese Math LLM**: Table~\ref{table:math} では、source models の MGSM-JA / JP-LMEH が `Shisa Gamma 7B v1` 9.6 / 66.1、`WizardMath 7B v1.1` 18.4 / 60.1、`Abel 7B 002` 30.0 / 56.5。提案モデルは `Ours (PS)` 7B が 52.0 / 70.5、`Ours (DFS)` 10B が 36.4 / 53.2、`Ours (PS+DFS)` 10B が 55.2 / 66.2。reference として GPT-3.5 は MGSM-JA 50.4、GPT-4 は 78.8、Swallow 70B は JP-LMEH 71.5 である。
- **主な結果 2, baselines**: Table~\ref{table:math-baselines} では、未最適化 TIES-Merge は 4.4 / 63.7、DARE-TIES は 35.2 / 66.3、Frankenmerging は 0.0 / 16.1。fine-tuning の最高 MGSM-JA は `LoRA, WizardMath 7B v1.1` の 43.2 / 55.9 で、提案の PS 52.0、PS+DFS 55.2 に届かない。著者は fine-tuning が JP-LMEH を大きく下げることを catastrophic forgetting と関連づけている。
- **主な結果 3, distraction**: 無関係モデルを追加した PS 実験では、`+0` irrelevant model が MGSM-JA 50.0 / JP-LMEH 65.9、`+1` が 46.8 / 64.2、`+2` が 46.8 / 64.1、`+4` が 48.4 / 64.0、`+8` が 40.8 / 65.8。著者は、manual source model selection に対して比較的 robust で、8 個追加しても collapse しないと述べる。
- **主な結果 4, 13B scale**: Table~\ref{table:13b} では、`ELYZA-japanese-Llama-2-13b-instruct` が 13.2 / 60.2、`MetaMath-13B-V1.0` が 8.0 / 48.7。提案は `Ours (PS)` 31.2 / 59.7、`Ours (DFS)` 2+1 が 23.2 / 46.6、`Ours (PS+DFS)` 34.0 / 60.4。著者は 7B より絶対スコアが低い理由を、`Mistral-7B-v0.1` が `Llama-2-13b` より基本的な math ability で強い source model であるためと説明している。
- **主な結果 5, VLM**: `tables/vlm.tex` の Table~\ref{table:vlm} では、`LLaVA 1.6 Mistral 7B` が JA-VG-VQA-500 14.3、JA-VLM-Bench-In-the-Wild 41.1、`Japanese Stable VLM` が後者 40.5。提案は `Ours (PS)` 19.7 / 51.2、`Ours (DFS)` 16.8 / 46.5、`Ours (PS+DFS)` 20.4 / 47.6。未最適化では TIES 16.0 / 46.5、DARE-TIES 9.4 / 36.0、Passthrough 7.3 / 26.7。
- **著者が主張する貢献**: Introduction の enumerate では、Automated Model Composition、Cross-Domain Merging、State-of-the-Art Performance、High Efficiency and Surprising Generalizability、Culturally-Aware VLM が挙げられる。さらに `EvoLLM-JP` と `EvoVLM-JP` の open-source release、Data Availability Statement での datasets 公開、Appendix の Apache 2.0 版 `EvoLLM-JP-A` も示される。

## 妥当性と限界

- **この主張を支える根拠**: MGSM-JA test set 250 samples は最適化に使わず、探索には GSM8k test の ID 250-1318 由来の 1069 samples を日本語訳して使う、と TeX に明記される。この分離により、少なくとも MGSM Japanese test set を直接最適化した結果ではない。さらに、未最適化 merging、LoRA / full fine-tuning、13B scale、distraction、$W$ ablation、source order ablation、VLM tasks が比較・解析として用意されている。
- **この主張を支える根拠**: PS analysis では、optimized weighting values が 3 source models でおおむね均一であり、weight sum が 1 を超えて 2 に近いことから、simple interpolation よりも contributions を amplify する combination が有効だったと著者は解釈する。日本語 LM の density が大きいことは、`Shisa-Gamma-7B-v1` の continued pretraining on 100B tokens による weight differences が DARE で sparsify しにくいことと関連づけられている。
- **この主張を支える根拠**: DFS analysis では、scaling parameters $W_{ij}$ を消して $W_{ij}=1$ にすると 10B PS+DFS で 20 percent を超える performance decline がある。13B ablation でも `Ours (PS+DFS w/o $W$)` は 26.4 / 58.1 で、$W$ ありの 34.0 / 60.4 を下回る。
- **著者が認めている limitations / future work**: Discussion の Limitations では、merged models が source models の limitations も継承し、responses that lacked logical coherence があったこと、instruction fine-tuning や alignment を含まないため factually flawed outputs の可能性があることを認めている。DFS では serial connections and non-adaptive configurations に限定し、より flexible な merging は future work としている。
- **著者が認めている limitations / future work**: Discussion では、source models はユーザーが選ぶ必要があり、将来的には進化計算で vast population of existing models から candidate source models を探索することも可能だと述べる。また multi-objective genetic algorithms such as NSGA-II を使う拡張も Methods で示唆される。
- **読者として注意すべき点**: MGSM-JA test set は 250 samples であり、TeX 中に confidence intervals や複数 seed の分散は示されていない。したがって数ポイント差の統計的安定性は本文だけでは判断できない。
- **読者として注意すべき点**: fine-tuning baseline は同じ 1069 samples で learning rates 1e-5、5e-5、1e-4、3 epochs、LoRA / full を試しているが、同じ GPU 時間予算での evolutionary merging と fine-tuning の compute-performance curve は TeX 中には示されていない。
- **読者として注意すべき点**: `WizardMath-7B-V1.1`、`Abel-7B-002`、`japanese-stablelm-base-gamma-7b` の training data や detailed training methods が公開されていないため、source models の全 training data を混ぜて base model を fine-tune する baseline は "impossible to implement for comparison" とされる。これは実用上自然な制約だが、比較可能な上限を明確には与えない。
- **追加で確認したい実験 / 疑問**: 複数 seed の CMA-ES 結果、MGSM-JA の信頼区間、同一 GPU 時間での SFT / LoRA / PS / DFS の曲線、VLM benchmark のより大規模な外部評価、DFS で skip された layer がなぜ harmful かを予測する解析があると、主張の再現性と機構理解が強くなる。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Evolutionary Model Merge**: model merging の recipe を進化計算で自動探索する著者らの一般枠組み。PS、DFS、PS+DFS の 3 形態を含む。
- **Parameter Space (PS)**: 複数モデルの weights を混ぜる空間。同一 architecture の source models を前提とし、TIES-Merging with DARE の sparsification / weight mixing parameters を最適化する。
- **Data Flow Space (DFS)**: weights は固定し、tokens が通る inference path を探索する空間。どの layer を使うかを indicator array $\mathcal{I}$ で表し、層間の scaling matrix $W$ も探索する。
- **source model**: merging の材料となる既存モデル。Japanese Math では `shisa-gamma-7b-v1`、`WizardMath-7B-V1.1`、`Abel-7B-002` が中心で、VLM では `LLaVA-1.6-Mistral-7B` と `shisa-gamma-7b-v1` が中心である。
- **TIES-Merging**: redundant parameter values と conflicting parameter signs による interference を減らすため、minimal changes の reset、sign conflicts の resolve、aligned parameters の merge を行う方法。
- **DARE**: task vector に random mask をかけて sparsify し、残った成分を rescale する方法。本論文では TIES と組み合わせた DARE-TIES を PS の基盤 recipe とする。
- **Frankenmerging**: mergekit にある、weights を平均するのではなく異なる models の layers を stack して新しい model を作る方法。本論文の DFS baseline に近いが、手動 recipe であり進化最適化はしない。
- **indicator array $\mathcal{I}$**: DFS で、順番に並べて $r$ 回繰り返した layers の各 slot を使うか除外するかを決める配列。$\mathcal{I}_i > 0$ なら対応 layer を include し、otherwise exclude する。
- **scaling matrix $W$**: DFS で、layer $i$ から layer $j$ に hidden states を渡すときの scaling factor を集めた行列。distribution shift を緩和する経験的な補正であり、ablation で重要性が示される。
- **MGSM-JA**: MGSM の Japanese test set。GSM8k test set の first 250 samples、IDs 0-249 の翻訳で、final evaluation に使われる。
- **JP-LMEH**: Japanese Language Model Evaluation Harness。JComQA、JNLI、MARC、JSQuAD、JAQKET、XLSum、XWino、MGSM、JCoLA の 9 tasks 平均が Japanese language proficiency の指標として使われる。
- **JA-VG-VQA-500**: Japanese Visual Genome VQA dataset から抽出した 500-sample test set。VLM の日本語 VQA 能力評価に使う。
- **JA-VLM-Bench-In-the-Wild**: 日本文化要素を含む 42 images、50 questions の VLM benchmark。GPT-4V の支援で QA を作り、人間が nonsensical outcomes を filtering したと説明される。
- **EvoLLM-JP-A**: Appendix の license 対応版。`shisa-gamma-7b-v1`、`Arithmo2-Mistral-7B`、`Abel-7B-002` を MIT または Apache 2.0 license の source として merge し、MGSM-JA 52.4、JP-LMEH 69.0 と報告される。

## 読む順番の提案

- まず `main.tex` の Abstract と Introduction を読み、著者が問題を "black art or alchemy" として置いていること、contributions の 5 項目を確認する。正規ノートでは Summary の「問題」「貢献」に対応する。
- 次に Methods の Figure~\ref{fig:overview} caption と `Merging in the Parameter Space`、`Merging in the Data Flow Space` を読む。PS と DFS の違い、`layer` の定義、$\mathcal{I}$ と $W$ の役割を押さえる。正規ノートでは Summary の PS / DFS / PS+DFS と Takeaway の「PS と DFS は直交」に対応する。
- 数式は Background の linear weight averaging、Task Arithmetic、DARE の式、Methods の DFS search space と $W_{ij}=\pi_\theta(i,j,t)$ を先に読む。これで本ノートの「重要な定義・数式」と正規ノートの PS / DFS 説明が接続できる。
- 実験は Table~\ref{table:math}、Table~\ref{table:lm-eval-harness}、Table~\ref{table:math-baselines}、Table~\ref{table:13b}、`tables/vlm.tex` の Table~\ref{table:vlm} の順で読む。数値は正規ノートの Summary「結果」とほぼ対応しているが、主張の強さは TeX の表と本文に戻って確認する。
- その後、Analysis の Figure~\ref{fig:ps_analysis}、Figure~\ref{fig:dfs_analysis}、13B DFS analysis を読む。重み合計が 1 を超える話、$W$ ablation、source order、layer #30 skip の話は正規ノートの Takeaway / Critical Thoughts に対応する。
- 最後に Discussion の Limitations、Ethical and Societal Impact、Appendix の license と case studies を読む。正規ノートの Critical Thoughts、Notes / Quotes、License 注意に対応する。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2403.13187v2/`
- 正規ノート: `notes/arXiv-2403.13187v2.md`
