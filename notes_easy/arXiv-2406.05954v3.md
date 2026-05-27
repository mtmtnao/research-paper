# Aligning Large Language Models with Representation Editing: A Control Perspective（test-time alignment と representation editing を最適制御として定式化する論文）

- arXiv: https://arxiv.org/abs/2406.05954
- 一次ソース: ../papers/arXiv-2406.05954v3/
- 正規ノート: ../notes/arXiv-2406.05954v3.md

---

## 一言で言うと

LLM alignment を、重みの fine-tuning ではなく、生成中の hidden state に小さな制御信号を入れる **dynamic representation editing** として扱う論文である。提案手法 **Re-Control** は、autoregressive LLM を discrete-time stochastic dynamical system と見なし、Bellman 方程式で hidden state 上の value function を学習し、test time に勾配上昇で制御信号を求める。

## 何を議論する論文か

- **問題設定**: LLM を helpful / harmless な応答へ揃える alignment を、モデル重みを更新せずに test time で行う。TeX の abstract は、fine-tuning は不安定で計算資源を要し、prompting や guided decoding は underlying model を変えないため性能が元モデルに依存すると述べる。
- **対象範囲 / 仮定**: 対象は transformer-based autoregressive LLM の逐次生成である。状態は \(s_t=\{h_t,o_t\}\) とされ、\(h_t\) は過去ステップの key-value pairs、\(o_t\) は出力側の hidden state / logits として使われる。実験では value network を last layer の hidden states \(o_t\) 上に学習し、test time でもこの層だけに control signal を足す。
- **既存研究との差分**: RLHF / DPO / PPO などの fine-tuning ではなく、test-time に representation space を摂動する。prompt engineering は入力 prompt を変え、controlled decoding は token probabilities と reward scores を組み合わせるが、Re-Control は activation / hidden representation を動的に動かす。Static Representation Editing は固定ベクトルを生成中ずっと足すが、Re-Control は autoregressive generation の各時刻で state-dependent な制御信号を求める。
- **この論文で答えたい問い**: 「representation editing を最適制御問題として定式化し、Bellman value function と test-time gradient optimization によって、既存の test-time alignment より良い alignment と実用的な推論時間を両立できるか」を検証する。

## 背景と前提

- **LLM alignment**: TeX の introduction では、LLM は多様なデータで訓練されるため misinformation や harmful content を生成し得るとし、human objectives / safety considerations に合わせる問題として alignment を置く。
- **Fine-tuning 系**: RLHF は human preference で Reward Model を訓練し、PPO などの reinforcement learning で LLM を fine-tune する。DPO などは RLHF を簡略化するが、TeX は依然として substantial computational resources が必要だと位置づける。
- **Test-time alignment 系**: Prompt engineering と guided / controlled decoding は重みを変えない。Controlled Decoding (CD) は token probabilities と reward scores を組み合わせ、CD prefix は partially generated responses から expected reward を予測する prefix scorer を学習する。
- **Representation engineering / editing**: LLM の representation space に steering vector や perturbation を加えて生成を制御する系統である。既存の Static RE は固定方向を足すため、生成過程の逐次的な state 変化を使い切れていない、というのが本論文の出発点である。
- **最適制御との対応**: background.tex では stochastic dynamical system を \(s_{t+1}=f(s_t,u_t,\omega_t)\) と書き、制御 \(u_t\) によって累積報酬を最大化する policy \(\pi:\mathcal{S}\to\mathcal{U}\) を求める問題として説明している。Re-Control はここでの state を LLM の \(h_t,o_t\)、control を hidden-state perturbation、stochasticity を sampled token \(y_t\) に対応させる。

## 提案手法

### コアアイデア

Re-Control は、pre-trained autoregressive LLM を **language dynamical system** として見る。この定義の LLM は直接の control signal を持たない uncontrolled system なので、各生成ステップで状態 \(s_t=\{h_t,o_t\}\) に \(u_t=\{u_t^h,u_t^o\}\) を加える。報酬は途中では与えず、EOS に到達した最終応答 \([\mathbf{x},\mathbf{y}_t]\) に対して reward model \(r\) が返す値を使う。

学習時には、初期 policy を「何もしない」\(u_t=0\) とし、その zero policy の value function を hidden state 上で学習する。test time では、学習済み \(V_\phi\) が大きくなる方向へ \(u_t\) を勾配上昇で更新し、その制御信号を加えたあと LLM の forward pass で次 token を生成する。著者は、global optimum を探すのではなく、original state から大きく離れず value score を改善することが目的だと、`method.tex` の `hidden.pdf` を用いた図キャプションで説明している。

### 重要な定義・数式

$$
y_{t} \sim \text{Softmax}(W o_{t}), \quad h_{t+1}, o_{t+1} = f_{\rm LM}(h_t, y_t).
$$

**式の意味**: TeX の Definition "Language dynamical system" にある、autoregressive LLM の逐次生成を dynamical system として書いた式である。各時刻で token \(y_t\) をサンプルし、その token を使って次の hidden state と出力側状態へ遷移する。

**記号の定義**:
- \(y_t\) ... 時刻 \(t\) に新しく生成される token
- \(h_t\) ... 過去時刻の key-value pairs を含む hidden state
- \(o_t\) ... token 分布を作る出力側の状態。TeX では last layer の hidden states / logits として扱われる
- \(W\) ... \(o_t\) を vocabulary space \(\mathcal{V}\) 上の確率分布へ写す linear transformation
- \(f_{\rm LM}\) ... LLM の state transition function

**この論文での役割**: LLM を制御対象として扱うための基礎定義である。この対応により、生成中の state に control signal を入れることが representation editing と optimal control の接点になる。

$$
y_{t} \sim \text{Softmax}\left(W (o_{t} + u^o_t)\right), \quad
h_{t+1}, o_{t+1} = f_{\rm LM}(h_t + u^h_t, y_t).
$$

**式の意味**: 上の language dynamical system に control signals \(u_t=\{u_t^h,u_t^o\}\) を加えた controlled language dynamical system である。出力側状態 \(o_t\) と hidden state \(h_t\) の両方に外部から摂動を入れられる形になっている。

**記号の定義**:
- \(u_t^o\) ... \(o_t\) に加える control signal
- \(u_t^h\) ... \(h_t\) に加える control signal
- \(u_t=\{u_t^h,u_t^o\}\) ... 時刻 \(t\) の制御入力
- その他の記号 ... 直前の language dynamical system の式と同じ

**この論文での役割**: Re-Control が「固定ベクトルを足す Static RE」ではなく、生成ステップごとに representation space を動的に perturb する手法であることを表す中心式である。ただし実装では全 state ではなく last layer の \(o_t\) だけに制御を加える。

$$
R\left([\mathbf{x}, \mathbf{y}_t]\right) := \begin{cases}
0 & \text{if } y_{t} \neq \operatorname{EOS} \\
r\left(\left[\mathbf{x}, \mathbf{y}_t\right]\right) & \text{if } y_{t} = \operatorname{EOS},
\end{cases}
\qquad
\argmax_{\{u_t\}_{t=1}^T} \mathbb{E} [R] - \lambda \sum_{t=1}^T ||u_t||_2^2.
$$

**式の意味**: 報酬は生成途中には与えず、EOS に到達した最終応答にだけ reward \(r\) を与える。そのうえで、期待報酬を上げつつ、制御信号の二乗ノルムを小さく保つ制御列 \(\{u_t\}\) を求める。

**記号の定義**:
- \(\mathbf{x}\) ... prompt
- \(\mathbf{y}_t\) ... 時刻 \(t\) までに生成された response
- \([\mathbf{x},\mathbf{y}_t]\) ... prompt と response の連結
- \(R\) ... alignment task の報酬関数
- \(r\) ... final response に対する reward。実験では公開 reward model を使う
- \(\lambda\) ... regularization の hyper-parameter

**この論文での役割**: alignment を「expected reward を最大化する制御問題」として明示する式である。正則化項は reward overoptimization を防ぎ、perturbed LLM の generation quality を保つために導入される。

$$
V(s_t) = \begin{cases}
\mathbb{E}_{s_{t+1}}\left[V(s_{t+1})\right], & \text{if } y_t \neq \operatorname{EOS} \\
r\left(\left[\mathbf{x}, \mathbf{y}_t\right]\right), & \text{if } y_t = \operatorname{EOS}.
\end{cases}
\qquad
\mathcal{L} = \sum_{i} \sum_{m} \sum_{t} \left(V_{\phi}(s^{i,m}_t) - \operatorname{stop-grad}(v^{i,m}_t)\right)^2.
\qquad
v^{i,m}_t =
\begin{cases}
V_{\phi}(s^{i,m}_{t+1}) & \text{if } y^{i,m}_t \neq \operatorname{EOS} \\
r^{i,m},  & \text{if } y^{i,m}_t = \operatorname{EOS}.
\end{cases}
$$

**式の意味**: zero policy \(u_t=0\) の value function を Bellman equation で定義し、その近似 \(V_\phi\) を MSE loss で学習する。途中状態の価値は次状態の価値の期待値、EOS 状態の価値は最終 reward である。

**記号の定義**:
- \(V(s_t)\) ... state \(s_t\) から最終的に得られる reward の期待値
- \(V_\phi\) ... neural network で parameterize された value function
- \(s_t^{i,m}\) ... prompt \(i\)、sample \(m\)、時刻 \(t\) の LLM state
- \(v_t^{i,m}\) ... Bellman target。TeX では \(y_t^{i,m}\neq\operatorname{EOS}\) なら \(V_\phi(s_{t+1}^{i,m})\)、EOS なら \(r^{i,m}\)
- \(\operatorname{stop-grad}(\cdot)\) ... target 側に gradient を流さない操作

**この論文での役割**: Re-Control の学習部分である。full RL を回すのではなく、pre-trained LLM の roll-out から hidden state trajectory と最終 reward を集め、value network だけを学習する設計を支えている。

$$
u_t = u_t + \alpha \nabla_{s_t} V_{\phi}(s_t + u_t).
$$

**式の意味**: test time に、現在の state へ加える control signal \(u_t\) を value function が増える方向に更新する。初期値は \(u_t=0\) で、この更新を \(n\) 回繰り返す。

**記号の定義**:
- \(u_t\) ... 時刻 \(t\) の control signal
- \(\alpha\) ... gradient ascent の step size
- \(n\) ... test-time intervention の更新回数
- \(\nabla_{s_t}V_\phi(s_t+u_t)\) ... state に関する value function の勾配

**この論文での役割**: 学習済み value function を使って、生成時に動的な制御信号を得る手順である。TeX は "Implicit Regularization" として、小さな \(\alpha\) と限られた \(n\) により control signal を小さく保つと説明する。

### 実装 / アルゴリズム上の要点

- **value function の入力**: 実験では last layer の hidden states \(o_t\) に value network を学習し、test time でもこの層だけに control signal を加える。TeX は attention key-value pairs \(h_t\) への拡張を future studies として述べる。
- **value network**: HH-RLHF では Vicuna-7B 用が 3 層、Falcon-7B 用が 2 層で、hidden dimension はどちらも 4096。SHP では Vicuna-7B / Llama3-8B とも 2 層、hidden dimension 4096。
- **学習データ作成**: prompt ごとに \(M\) responses を sample する一般式を出すが、appendix の実験設定では HH-RLHF と SHP とも \(M=1\) と明記されている。
- **学習設定**: Adam、learning rate \(1*10^{-4}\)、batch size 512、fp16、100 epochs。計算環境は NVIDIA A100 80GB、CUDA 12.4、Python 3.12.2、PyTorch 2.2.2。
- **test-time hyperparameters**: validation set 上で coherence + diversity + average reward の和を最大化するように \(\alpha,n\) を選ぶ。HH-RLHF では Vicuna-7B が \(\alpha=0.5,n=30\)、Falcon-7B が \(\alpha=0.2,n=200\)。SHP では Vicuna-7B が \(\alpha=1.0,n=50\)、Llama3-8B が \(\alpha=1.0,n=30\)。

## 実験・結果

- **データセット / ベンチマーク**: 主実験は \(\texttt{HH-RLHF}\) と \(\texttt{Stanford SHP}\)。HH-RLHF は appendix で 161,000 training samples と 8,550 test samples とされ、helpfulness / harmlessness 改善に使われる。SHP は 385,000 collective human preferences、349,000 training、18,400 validation、18,400 test samples とされ、Reddit post と top-level comments の preference から成る。OOD 解析では \(\texttt{HarmfulQA}\) を使い、1,960 harmful questions、9,536 harmless conversations、7,356 harmful conversations と説明される。
- **比較対象 / baseline**: Base、Prompt Engineering、Static RE、Controlled Decoding (CD)、CD prefix、CD prefix + Prompting、Ours、Ours + Prompting。Further Analysis では LoRA-based PPO / DPO とも比較する。CD は reward model と base model の tokenization strategy が同じである必要があり、表では Falcon-7B や Llama3-8B の一部が N/A になっている。
- **指標**: Diversity、Coherence、Average Reward、Win Rate、Inference Time。Diversity は repeated n-grams に基づき、Coherence は prompt と continuation の SimCSE embedding cosine similarity、Average Reward は reward model の平均、Win Rate は GPT-4 judge によって model response が dataset の preferred response より良いと評価された割合である。HH-RLHF の GPT-4 評価では test set から 300 prompts を random sample し、応答提示順を randomize する。
- **reward model**: HH-RLHF では `argsearch/llama-7b-rm-float32`、SHP では `openbmb/UltraRM-13b` を使う。appendix では UltraRM-13B が Anthropic HH-RLHF、Standford SHP、Summarization で訓練されていると説明される。
- **主な結果（Table \(\ref{tab:performance}\), HH-RLHF / Vicuna-7B）**: Base は Diversity 0.816、Coherence 0.568、Average Reward 5.894、Win Rate 57.6、Inference Time 0.60h。Ours は 0.824、0.579、6.214、75.6、0.85h。Ours + Prompting は 0.830、0.577、6.267、80.3、0.93h。CD は Win Rate 72.3 だが Inference Time 47.43h、CD prefix は Win Rate 74.6、32.13h。
- **主な結果（HH-RLHF / Falcon-7B）**: Base は Win Rate 42.3、Average Reward 3.439。Ours は Win Rate 58.0、Average Reward 3.512、Inference Time 1.93h。Ours + Prompting は Win Rate 62.6、Average Reward 4.083、Inference Time 2.00h。CD prefix は Average Reward 4.397 と高いが、Win Rate は 49.6、Inference Time 48.13h。
- **主な結果（SHP / Vicuna-7B）**: Base は Win Rate 40.3、Average Reward -5.68。Ours は Win Rate 58.0、Average Reward -5.38。Ours + Prompting は Win Rate 63.6、Average Reward -4.63。
- **主な結果（SHP / Llama3-8B）**: Base は Win Rate 56.3、Average Reward -4.64。Ours は Win Rate 71.0、Average Reward -4.39。Ours + Prompting は Win Rate 77.0、Average Reward -4.14。
- **著者が主張する貢献**: introduction では、(1) control perspective から LLM alignment の representation editing method を提案、(2) value function を学習し test time に gradient-based optimization で control signal を計算、(3) 既存 test-time alignment methods を上回り strong generalization ability を示す、と列挙している。
- **Further Analysis**: LoRA-based PPO / DPO との比較では、Vicuna-7B on HH-RLHF で Re-Control は "competitive alternative to LoRa-based fine-tuning methods" と主張される。HarmfulQA では HH-RLHF で学習した value function を使い、reward model は OOD で accurate ではないため GPT-4 評価に集中し、Ours + Prompting が Vicuna-7B と Falcon-7B の両方で highest GPT-4 win rate と述べられる。

## 妥当性と限界

- **この主張を支える根拠**: Table \(\ref{tab:performance}\) は 4 設定（HH-RLHF x Vicuna-7B / Falcon-7B、SHP x Vicuna-7B / Llama3-8B）で Ours / Ours + Prompting が GPT-4 Win Rate を改善していることを示す。特に CD は推論時間が大きく、TeX は controlled decoding が entire reward model を繰り返し forward する一方、Re-Control は 2- or 3-layer value function を通すため速いと説明する。
- **評価設計の妥当性**: Average Reward だけでなく Diversity / Coherence / GPT-4 Win Rate / Inference Time を同時に見る設計になっている。Falcon-7B の HH-RLHF では CD prefix が Average Reward 4.397 で最大だが Win Rate は Ours より低く、reward model のスコアだけを最終判断にしていない点は重要である。
- **著者が認めている limitations / future work**: appendix A は、(1) 現状は last layer の hidden space の value function のみで、intermediate layers や low-rank subspace への拡張が future work、(2) current paper は single reward model の目的だけを扱い、multiple conflicting objectives に対する multi-objective optimization が future work、(3) value function 学習は simple one-iteration policy iteration であり、iterations を増やすことや provable convergence を持つ value-function training algorithm が future work、と述べる。
- **読者として注意すべき点**: hyperparameter study では、\(\alpha\) や \(n\) を増やすと reward が改善するだけでなく、coherence と diversity が nearly zero まで落ちる reward overoptimization が観察される。したがって、Re-Control の効果は test-time intervention の強さを validation set で選ぶ手順に依存する。
- **読者として注意すべき点**: GPT-4 judge は helpfulness、harmlessness、relevance、accuracy、insightfulness などで 1-10 点を付ける設計だが、人手評価を追加したという記述は TeX 中には明示されていない。SHP については 1,000 random sampled test prompts で評価したと appendix にある一方、HH-RLHF のように「GPT-4 評価は 300 prompts」と明確に分けた記述は限定的である。
- **Broader Impacts**: appendix B は、value function の学習に "negative goals" を含めると誤用され得るため注意が必要だと述べる。
- **追加で確認したい実験 / 疑問**: 複数層や low-rank subspace への介入、多目的 reward、value function の policy iteration 回数、reward model と GPT-4 judge の独立性、人手評価、長い生成長での安定性は、この TeX だけでは十分には検証されていない。

## 用語メモ

- **Re-Control** ... 本論文の提案手法。hidden state 上の value function を学習し、test time に \(u_t\) を勾配上昇で求める dynamic representation editing。
- **language dynamical system** ... autoregressive LLM を、状態 \(s_t=\{h_t,o_t\}\) と sampled token \(y_t\) によって時間発展する stochastic dynamical system と見たもの。
- **control signal \(u_t\)** ... 生成ステップ \(t\) で hidden state / output state に加える摂動。実験では last layer の \(o_t\) のみに加える。
- **zero policy** ... 何も制御しない policy、すなわち \(u_t=0\)。Re-Control はこの policy の value function を一度だけ学習する。
- **value function \(V_\phi\)** ... state \(s_t\) から最終的に得られる reward の期待値を予測する neural network。実験では 2-3 層 MLP、hidden dimension 4096。
- **Bellman equation** ... 途中状態の価値を次状態の価値の期待値に結び、EOS では final reward に結びつける再帰式。
- **implicit regularization** ... objective には \(\lambda\|u_t\|_2^2\) があるが、test time では小さい step size \(\alpha\) と限定された updates \(n\) によって control signal を小さく保つ、という実装上の正則化。
- **Static RE** ... prompt 後の hidden state から expected reward を予測する linear regression を学習し、その重み方向へ activation space を固定的に shift する baseline。
- **CD / CD prefix** ... CD は reward score を token probabilities に組み合わせる guided decoding。CD prefix は partially generated responses から expected reward を予測する prefix scorer を使う。
- **Win Rate** ... GPT-4 judge が、生成応答を dataset の preferred response より良いと評価した割合。HarmfulQA では reference response がないため base model response より良い割合として測る。
- **reward overoptimization** ... reward を高める方向に強く動かしすぎることで、coherence / diversity が崩れる現象。本論文では Fig. \(\ref{fig:parameter}\) と hyperparameter study で観察される。

## 読む順番の提案

- まず `main.tex` の abstract を読み、問題設定が fine-tuning のコスト、test-time alignment の限界、representation editing の導入であることを押さえる。
- 次に `intro.tex` を読み、contributions の 3 点と、Static RE が固定 perturbation で autoregressive nature を使っていないという動機を確認する。正規ノートの Summary の「問題」「貢献」に対応する。
- 数学的な対応を理解するには `background.tex` の stochastic dynamical system / optimal control を軽く見てから、`method.tex` の Definition "Language dynamical system"、controlled system、objective、Bellman equation、test-time update を順に読む。正規ノートの Re-Control の説明と数式の中心部分に対応する。
- 実験の読み方は `exp.tex` の Experimental Setup、Baselines、Experimental Results を読んだあと、`table/maintable.tex` の Table \(\ref{tab:performance}\) で 4 設定の数値を確認する。正規ノートの Summary「結果」に対応する。
- 限界と実装詳細は `appendix.tex` を読む。特に `appendix:limitations`、HH-RLHF / SHP の value network 設定、test-time \(\alpha,n\)、HarmfulQA の OOD 設定、GPT-4 evaluation prompt が重要である。正規ノートの Critical Thoughts と Notes / Quotes に対応する。
- `main.bbl` は baseline や関連研究の対応を確認するために使う。たとえば Li et al. の Inference-Time Intervention は main.bbl では NeurIPS 2023 として載っており、Static RE の背景を追うときの参照になる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2406.05954v3/`
- 正規ノート: `notes/arXiv-2406.05954v3.md`
