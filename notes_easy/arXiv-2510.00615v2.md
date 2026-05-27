# Acon: Optimizing Context Compression for Long-horizon LLM Agents（長期 horizon LLM エージェントの context compression 最適化）

- arXiv: https://arxiv.org/abs/2510.00615
- 一次ソース: ../papers/arXiv-2510.00615v2/
- 正規ノート: ../notes/arXiv-2510.00615v2.md

---

## 一言で言うと

この論文は、長期 horizon の LLM agent で増え続ける interaction history と latest observation を、タスク成功に必要な情報を落としにくい形に圧縮する **Agent Context Optimization (Acon)** を提案する。著者は AppWorld、OfficeBench、8-objective QA で、主に peak tokens を 26-54% 減らしつつ大きな性能劣化を避け、小型 compressor や小型 agent にも効果が移ると主張している（`text/0_abstract.tex`, `text/1_introduction_v2.tex`）。

## 何を議論する論文か

- **問題設定**: LLM agent は各 step で最新 observation $o_t$ と過去の interaction history $\mathbf{h}_{t-1}$ を読んで action $a_t$ を出す。長期タスクでは actions と observations が蓄積し、context length が事実上 unbounded になるため、inference cost と不要情報による判断劣化が問題になる（`text/1_introduction_v2.tex`, `figures/1_concept.tex`）。
- **対象範囲 / 仮定**: agent の LLM パラメータ $\theta$ と agent prompt $\mathcal{P}_{\sf agent}$ は固定し、compression 用 LLM $f(\cdot;\phi,\mathcal{P})$ の guideline prompt $\mathcal{P}$ を自然言語空間で最適化する。環境は POMDP $\mathcal{E}=\langle\mathcal{S},\mathcal{A},\mathcal{O},\mathcal{T},\mathcal{R}\rangle$ として定式化され、transition function $\mathcal{T}(s,a)\rightarrow(s',o)$ は deterministic とされる（`text/3_method_v3.tex`）。
- **既存研究との差分**: 既存の context compression は document/RAG compression、dialogue memory summarization、KV-cache compression に大別されるが、著者はこれらが long-horizon agent の heterogeneous context には不十分だと位置づける。agent 向けの Mind2Web、ContextualizeWeb、openhands-condenser、SWE-agent 系も、naive prompting または narrow domain に限られると述べる（`text/2_related_works_v2.tex`, `main.bbl`）。
- **この論文で答えたい問い**: 1) context を圧縮して token efficiency を上げても task performance を保てるか、2) optimized compressor を小型モデルへ distill できるか、3) 小型 LM agent が long context の悪影響を受ける場面で Acon が性能を上げるか、を実験で検証する（`text/4_experiments.tex`）。

## 背景と前提

- **LLM agent の context は単なる会話履歴ではない**。この論文では、agent context は environment の "world model" として働くと説明される。具体的には causal relations、evolving states、preconditions、future decision cues が混ざるため、一般的な要約で「読みやすく短くする」だけでは不十分になる（`text/3_method_v3.tex`）。
- **history compression と observation compression は別物**。history compression は過去の $o_0,a_0,\ldots,o_{t-1},a_{t-1}$ を短くする。一方、observation compression は現在返ってきた長い observation $o_t$ を、history $\mathbf{h}_{t-1}$ も見ながら圧縮する。両方とも閾値を超えたときだけ compressor を呼ぶ。
- **baseline の役割**。No Compression は上限参照、FIFO は古い interaction を落とす単純法、Retrieval は current query に近い過去 interaction を embedding search で取る方法、LLMLingua は extractive compression、Prompting は general compression instruction による naive LLM compression である（`text/4_experiments.tex`, `text/999_appendix.tex`）。
- **prompt optimization 系研究との関係**。Acon は OPRO、TextGrad、DSPy、APO などの「LLM feedback を使って自然言語 prompt を更新する」流れに属する。ただし、更新対象は agent の解答 prompt ではなく、agent context を圧縮する guideline である（`text/3_method_v3.tex`, `main.bbl`）。
- **評価で見る cost は複数ある**。本文の主張で中心になるのは peak tokens であり、これは trajectory 中の単一 sequence の最大 token 数である。Dependency は action generation が過去 token にどれだけ依存したかの累積量で、API cost は cached input tokens も考慮した別分析である（`text/999_appendix.tex`, `tables/11_12_one_row.tex`）。

## 提案手法

### コアアイデア

Acon の中心は、compressor LLM のパラメータを RL で直接更新するのではなく、自然言語で書かれた compression guideline $\mathcal{P}$ を最適化することである。まず training set で「No Compression では成功するが、compressed context では失敗する」trajectory pair を集める。この contrastive subset $\mathcal{D}_{\sf cont}$ は、圧縮によって何か重要な情報が落ちた可能性を示す。

optimizer LLM は、full context $\mathcal{H}$ と compressed context $\mathcal{H}'$ を比べ、失敗の原因を自然言語 feedback として出す。その feedback をまとめて guideline $\mathcal{P}^{(0)}$ から $\mathcal{P}^{(1)}$ へ更新する。この段階を **utility maximization step (UT)** と呼ぶ。次に、compressed context でも成功した task だけを使って「実際に使われた情報」と「冗長な情報」を分析し、より短い guideline へ refine する。この段階を **compression maximization step (CO)** と呼ぶ（`text/3_method_v3.tex`, `text/3_algorithm.tex`, `figures/3_method.tex`）。

最適化後の compressor は gpt-4.1 teacher compressor として使われるが、推論時の overhead を減らすため、teacher の圧縮入出力 $(\mathbf{x},\mathbf{y})$ を使って Qwen3-14B、Qwen3-8B、Phi-4 などへ SeqKD + LoRA で distill する（`text/3_method_v3.tex`, `text/4_experiments.tex`, `figures/5_distilled_history_compressor.tex`）。

### 重要な定義・数式

$$
\mathcal{M}(o_t, \mathbf{h}_{t-1}; \theta, \mathcal{P}_{\sf agent}) \mapsto a_t
$$

**式の意味**: agent LLM が、現在の observation と過去の interaction history を条件に次の action を生成する、という問題設定の基本式である（`text/3_method_v3.tex`, Eq. `eq:agent`）。

**記号の定義**:
- $\mathcal{M}$ ... action を生成する LLM agent
- $o_t$ ... step $t$ の latest observation
- $\mathbf{h}_{t-1}$ ... $t-1$ までの interaction history、すなわち $(o_0,a_0,o_1,a_1,\ldots,o_{t-1},a_{t-1})$
- $\theta$ ... 事前学習済み LLM の固定パラメータ
- $\mathcal{P}_{\sf agent}$ ... environment description、tools、output format、few-shot examples などを含む agent prompt
- $a_t$ ... step $t$ で生成される action

**この論文での役割**: Acon は $\mathcal{M}$ や $\theta$ を直接変えるのではなく、この式に渡される $o_t$ と $\mathbf{h}_{t-1}$ を圧縮して、action generation の cost と品質を変える。

$$
C(\mathcal{H}) = \sum\nolimits_{t=1}^{T}\mathcal{C}(\mathbf{h}_{t-1}, o_t)
$$

**式の意味**: 1 つの task trajectory 全体で、各 step の dynamic context を encode する cost を足し合わせた total cost を定義する（`text/3_method_v3.tex`, Eq. `eqn:cost`）。

**記号の定義**:
- $C(\mathcal{H})$ ... task 完了までの context cost の総和
- $\mathcal{H}$ ... $\{\mathbf{h}_{t-1},o_t\}_{t=1}^{T}$、各 step の history と observation の列
- $T$ ... trajectory の総 step 数
- $\mathcal{C}(\mathbf{h}_{t-1},o_t)$ ... step $t$ の per-step cost。本文では transformer が $n$ input tokens を decode するときの $\mathcal{O}(n)$ computational cost の例が挙げられる

**この論文での役割**: context が長くなるほど $C$ が増えるため、Acon の圧縮は単に文章を短くする処理ではなく、この cost を下げるための介入として定式化される。

$$
\begin{aligned}
\mathbf{h}_t' &=
f(\mathbf{h}_t;\phi,\mathcal{P}_{\sf hist}) \;\;\text{if } |\mathbf{h}_t| > T_{\sf hist}, \quad
\mathbf{h}_t \;\;\text{otherwise}, \\
o_t' &=
f(o_t, \mathbf{h}_{t-1}; \phi, \mathcal{P}_{\sf obs}) \;\;\text{if } |o_t| > T_{\sf obs}, \quad
o_t \;\;\text{otherwise}.
\end{aligned}
$$

**式の意味**: history または latest observation が閾値を超えたときだけ LLM compressor $f$ で圧縮し、短ければそのまま残す、という selective compression の定義である（`text/3_method_v3.tex`, Eq. `eqn:history_comp`, `eqn:obs_comp`）。

**記号の定義**:
- $f(\cdot;\phi,\mathcal{P})$ ... compression を行う LLM
- $\phi$ ... compressor LLM のパラメータ
- $\mathcal{P}_{\sf hist}$ ... history compression 用 guideline
- $\mathcal{P}_{\sf obs}$ ... observation compression 用 guideline
- $T_{\sf hist}$ ... history compression の閾値
- $T_{\sf obs}$ ... observation compression の閾値
- $\mathbf{h}_t'$、$o_t'$ ... 圧縮後、または未圧縮のまま使われる context

**この論文での役割**: Acon は history と observation を統一的に扱うが、入力と失われやすい情報が違うため guideline は分ける。history compression の実装では latest action-observation pair を保持し、必要時だけ compressor を呼ぶ（`text/999_appendix.tex`）。

$$
\max_{\psi}\;
\underbrace{\mathbb{E}\!\left[\,\mathcal{R}\!\big(s_T(\psi)\big)\,\right]}_{\text{maximize}}
\;-\;
\lambda\,
\underbrace{\mathbb{E}\!\left[\,C\!\big(\mathcal{H}'(\psi)\big)\,\right]}_{\text{minimize}},
\qquad \lambda \ge 0
$$

**式の意味**: compressor の設定 $\psi$ を、task reward を上げつつ compressed context の cost を下げるように選ぶ、という Acon の目的関数である（`text/3_method_v3.tex`, Eq. `eq:optim`）。

**記号の定義**:
- $\psi \triangleq (\phi,\mathcal{P})$ ... compressor のパラメータと compression guideline
- $\mathbb{E}[\cdot]$ ... tasks に関する期待値
- $\mathcal{R}(s_T(\psi))$ ... compressed context を使った trajectory の terminal state に対する reward
- $\mathcal{H}'(\psi)$ ... $\psi$ によって得られる compressed context sequence
- $C(\mathcal{H}'(\psi))$ ... compressed context の total cost
- $\lambda$ ... reward と cost の trade-off を調整する非負の multiplier

**この論文での役割**: UT は主に reward 項を改善し、CO は主に cost 項を下げる、という二段階最適化の説明に使われる。ただし実際には reward が sparse、gold compression がない、cost が discrete であるため、著者は RL ではなく natural language prompt optimization を使う。

$$
\mathcal{P}^{(1)}
= \text{LLM}(\text{Update Instruction}, \mathcal{P}^{(0)}, \textstyle \Vert_{i=1}^n \text{Feedback}_i)
$$

**式の意味**: contrastive trajectory から得た複数の natural language feedback を連結し、optimizer LLM に渡して compression guideline を更新する式である（`text/3_method_v3.tex`, Eq. `eqn:prompt_opt`）。

**記号の定義**:
- $\mathcal{P}^{(0)}$ ... 更新前の compression guideline
- $\mathcal{P}^{(1)}$ ... UT 後の updated guideline
- $\text{Feedback}_i$ ... full context では成功し compressed context では失敗した task から得た、失敗原因に関する自然言語 feedback
- $\Vert$ ... feedback の concatenation
- $\text{Update Instruction}$ ... guideline 更新用 prompt

**この論文での役割**: Acon の「gradient-free」性を具体化する更新式である。数値勾配ではなく、失敗原因の自然言語分析を batch textual gradient のように扱い、さらに複数候補を training subset で評価して選ぶ。

### 実装 / アルゴリズム上の要点

- **baseline trajectory の収集**: training set $\mathcal{D}_{\sf train}$ で No Compression の trajectory $\mathcal{H}$ と成功ラベルを集める。Algorithm 1 では baseline success がある index 集合 $\mathcal{I}^{+}$ を作る。
- **UT step**: 現在の guideline で compressed trajectory $\mathcal{H}'$ を作り、$r_i^{\sf base}=1$ かつ $r_i(\mathcal{P})=0$ の task から $\mathcal{D}_{\sf cont}$ を作る。optimizer LLM は full history と compressed history の差分を見て、missing critical facts、lost variables、API misuse、inefficient looping などを JSON で分析する prompt が用意されている（`prompts/03_prompt_optimizer_analysis_prompt.tex`）。
- **candidate selection**: prompt update 時には 5 candidate prompts を sample し、training subset 上で best-performing candidate を選ぶ。default optimizer は `o3`（`text/999_appendix.tex`）。
- **CO step**: 圧縮ありで成功した task $\mathcal{D}_{\sf succ}$ だけを使い、どの情報が実際に必要だったか、どの冗長 pattern を削れるかを analyzer LLM に出させる。AppWorld の CO 後 prompt では、長い log、未使用 credential、table borders、meta prose などを削り、必要な tokens、ids、emails、amounts、lists、paths などは残すよう明示される（`prompts/01_appworld_history_optimized_length.tex`）。
- **compression thresholds**: history は AppWorld / OfficeBench で $T_{\text{hist}}=4096$、8-objective QA で $2048$。observation は AppWorld で $T_{\text{obs}}=1024$、OfficeBench で $512$、8-objective QA で $400$。history compression では、Acon と全 baseline が latest action-observation pair を保持する（`text/999_appendix.tex`）。
- **distillation**: teacher compressor は UT 後の optimized guideline を使う gpt-4.1 で、student は Qwen3-14B、Qwen3-8B、Phi-4 など。LoRA rank 16、$\alpha=32$、learning rate $10^{-4}$、3 epochs、batch size 4、max sequence length 10,000、AdamW、5% warmup、weight decay 0.01、1 A100 80GB GPU が使われる（`text/4_experiments.tex`, `text/999_appendix.tex`, `figures/5_distilled_history_compressor.tex`）。

## 実験・結果

- **データセット / ベンチマーク**: AppWorld、OfficeBench、8-objective QA の 3 つ。AppWorld は 9 everyday applications と約 100 simulated users、457 APIs を持ち、90 training tasks と 168 test-normal tasks を使う。OfficeBench は Word、Excel、PDF、Calendar、Email、Shell、Calculator などの office automation task で、text-related tasks に限定し、92 train / 95 test に 1:1 split する。8-objective QA は NaturalQuestions の質問を 8 個束ね、2018 Wikipedia 上の BM25 retriever を使う 100 train / 100 test task である（`text/999_appendix.tex`）。
- **比較対象 / baseline**: No Compression、FIFO、Retrieval、LLMLingua、Prompting。FIFO は last 5 interaction turns、Retrieval は 4 interaction turns plus last turn、LLMLingua は keep rate 30%、Retrieval embedding は OpenAI `text-embedding-3-large` を使う（`text/4_experiments.tex`, `text/999_appendix.tex`）。
- **指標**: AppWorld は task completion score、OfficeBench は benchmark-defined accuracy functions、8-objective QA は EM と F1。効率指標は Steps、Peak Tokens、Dependency。Peak Tokens は system prompts を除く trajectory 中の最大 sequence token 数、Dependency は LightThinker / MEM1 に従う cumulative dependency である（`text/4_experiments.tex`, `text/999_appendix.tex`）。
- **主な結果**: AppWorld test-normal 168 tasks では、gpt-4.1 agent / gpt-4.1 compressor の No Compression が Acc. 56.0、Peak 9.93k、Dep. 5.96。History Acon UT+CO は Acc. 56.5、Peak 7.33k、Dep. 4.69 で、Peak は約 26% 減る。FIFO 45.8、Retrieval 27.4、LLMLingua 39.3、Prompting 43.5 より高い（`tables/1_appworld_gpt-4.1.tex`）。
- **主な結果**: AppWorld hard 63 tasks では No Compression 39.7、history FIFO 15.9、Retrieval 7.9、LLMLingua 15.9、Prompting 23.8、Acon UT 28.6、Acon UT+CO 30.2。Observation Acon UT+CO は hard 31.8 である（`tables/1_appworld_gpt-4.1.tex`）。
- **主な結果**: OfficeBench 95 test tasks では No Compression が Acc. 76.84、Peak 7.27k。History Acon UT は Acc. 74.74、Peak 4.93k、History Acon UT+CO は Acc. 72.63、Peak 4.54k。Level 3 では No Compression 54.84 に対し History Acon UT は 61.29 と高いが、UT+CO は 51.61 である（`tables/3_4_officebench_qa.tex`, `tables/4_officebench_main.tex`）。
- **主な結果**: 8-objective QA では No Compression が EM 0.366、F1 0.488、Peak 10.35k、Dep. 3.32。History Acon UT は EM 0.373、F1 0.494、Peak 4.71k、Dep. 1.57。Peak は約 54.5% 減る（`tables/3_4_officebench_qa.tex`）。
- **主な結果**: distillation について、著者は optimized compressors を小型モデルへ distill しても gpt-4.1 compressor の accuracy の 95% 超を保つと述べる（`text/0_abstract.tex`, `text/4_experiments.tex`, `figures/5_distilled_history_compressor.tex`）。AppWorld の追加表では、history Acon Qwen3-14B compressor が Acc. 50.0、Peak 6.83k、observation Acon Qwen3-14B compressor が Acc. 56.5、Peak 7.57k である（`tables/5_appworld_distilled_optimizer.tex`）。
- **主な結果**: 小型 agent では、Qwen3-14B agent が AppWorld で 26.8% から 33.9%、8-objective QA で EM 0.158 から 0.197 に改善したと本文が述べる。Introduction では小型 LM の改善として AppWorld 32%、OfficeBench 20%、Multi-objective QA 46% が挙げられる（`text/1_introduction_v2.tex`, `text/4_experiments.tex`, `figures/6_distilled_agent_scatter.tex`）。
- **著者が主張する貢献**: 1) history と observation の両方を扱う agent context compression framework、2) failure-driven かつ task-aware な gradient-free compression guideline optimization、3) optimized compressor の distillation、4) 3 benchmark で peak tokens 26-54% 削減と小型 LM agent の 20-46% 改善である（`text/1_introduction_v2.tex`）。

## 妥当性と限界

- **この主張を支える根拠**: AppWorld、OfficeBench、8-objective QA という性質の異なる long-horizon benchmarks で、同じ枠組みを history / observation compression の両方に適用している。主実験は gpt-4.1 agent / gpt-4.1 compressor で揃え、baseline も FIFO、Retrieval、LLMLingua、Prompting を並べている（`tables/1_appworld_gpt-4.1.tex`, `tables/3_4_officebench_qa.tex`）。
- **この主張を支える根拠**: prompt optimizer ablation では、AppWorld history compression で default の `o3` + task contrastive feedback が Acc. 51.2、contrastive feedback なしが 50.6、`gpt-4.1` optimizer が 47.6、`gpt-5` optimizer が 50.6 である。著者の「contrastive feedback が有効」という主張はこの表に支えられるが、`o3` での差は +0.6 pt と大きくはない（`tables/11_12_one_row.tex`）。
- **この主張を支える根拠**: threshold ablation では、threshold が小さすぎると compression calls が増えて accuracy が落ち、大きすぎると accuracy は保ちやすいが cost が残ると説明される。著者は moderate values、すなわち history 4096 / observation 1024 を AppWorld の良い trade-off とする（`text/4_experiments.tex`, `figures/4_compression_frequency.tex`）。
- **著者が認めている limitations / future work**: history compression は total cost を増やす場合がある。理由は、compressed history で難しい task の step 数が増えうることと、Transformer の KV-cache を壊して compressed histories の再計算を強いることにある。Observation compression はこの overhead を一部緩和するが、generative compression 自体の latency は残る。future work として KV-cache-level compression / eviction strategies が挙げられる（`text/999_appendix.tex`）。
- **著者が認めている limitations / future work**: 主な実験は budget constraints により GPT models に寄っており、Gemini、Claude、DeepSeek-R1、Qwen3-235B などへの一般化は未検証である（`text/999_appendix.tex`）。
- **読者として注意すべき点**: 本文 `text/4_experiments.tex` は 8-objective QA の history compression について "dependency by 54.5% and 61.5%" と書くが、`tables/3_4_officebench_qa.tex` の History Acon UT は Dep. 3.32 から 1.57 なので削減率は約 52.7%。61.5% は Observation Acon UT の Dep. 1.28 に近い。数値を使うときは表を優先して確認した方がよい。
- **読者として注意すべき点**: 「peak tokens が減る」ことと「API dollar cost が減る」ことは同じではない。著者自身も API cost analysis で、observation compression は input compression により cost を下げる一方、history compression は KV-cache overhead により rarely helps と述べる（`text/4_experiments.tex`, `tables/11_12_one_row.tex`）。
- **読者として注意すべき点**: History + Observation compression を同時に使うと、AppWorld 追加表では Peak は 5.85k や 5.90k まで下がるが、Acc. は 45.8 や 44.6 まで落ちる。つまり、2 種類の圧縮を単純に足せば最良になるわけではない（`tables/1_appworld_gpt-4.1_additional.tex`）。
- **追加で確認したい実験 / 疑問**: optimizer LLM (`o3`) を使う guideline optimization 自体の総 cost や latency は、主表では直接比較されていない。実運用では、offline optimization cost、online compressor cost、agent cost を分けて測る必要がある。
- **追加で確認したい実験 / 疑問**: OfficeBench は official split がないため著者が 92/95 に random split している。ambiguous tasks の除去と testbeds の非共有は確認したと書かれているが、公式 split ではない点は読む側で留意する必要がある（`text/999_appendix.tex`）。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Agent Context Optimization (Acon)** ... long-horizon LLM agent の interaction histories と environment observations を、task-aware な guideline に従って圧縮する framework。
- **interaction history $\mathbf{h}_{t-1}$** ... $o_0,a_0,o_1,a_1,\ldots,o_{t-1},a_{t-1}$ の列。単なるログではなく、後続の action generation に必要な state、IDs、API format、途中結果を含む。
- **latest observation $o_t$** ... 現在 step で environment から返る observation。長い API schema、Web/page content、document content などが入りうる。
- **history compression** ... $\mathbf{h}_t$ が $T_{\sf hist}$ を超えたとき、過去の interaction 全体を summary に置き換える処理。latest action-observation pair は保持される。
- **observation compression** ... $o_t$ が $T_{\sf obs}$ を超えたとき、現在 observation を history も参照しながら短くする処理。圧縮後の $o_t'$ は agent 入力になり、history にも保存される。
- **compression guideline $\mathcal{P}$** ... compressor LLM に渡す自然言語 prompt。Acon が最適化する主対象であり、$\mathcal{P}_{\sf hist}$ と $\mathcal{P}_{\sf obs}$ がある。
- **contrastive task feedback** ... No Compression では成功し compressed context では失敗した trajectory pair を比べ、圧縮で失われた情報や歪んだ情報を自然言語で特定する feedback。
- **UT (utility maximization step)** ... contrastive failures から guideline を更新し、主に task reward を回復する段階。
- **CO (compression maximization step)** ... compressed context でも成功した task から冗長部分を見つけ、より短い guideline に refine する段階。
- **Peak Tokens** ... system prompts を除き、trajectory 中の単一 sequence で観測される最大 token 数。本文の 26-54% 削減の中心指標。
- **Dependency** ... 各 step の input/output tokens から計算される、action generation の cumulative dependency。式は `text/999_appendix.tex` にある。
- **KV-cache overhead** ... history compression で過去文脈を summary に置き換えると、以前の KV-cache をそのまま使いにくくなり、再計算 cost が生じる問題。
- **SeqKD + LoRA distillation** ... teacher compressor の出力を student model に cross-entropy で真似させ、LoRA で小型 compressor を作る手順。
- **AppWorld** ... 9 everyday applications、457 APIs、約 100 simulated users を含む benchmark。Acon の主 benchmark。
- **OfficeBench** ... Word、Excel、Email など複数 office apps をまたぐ automation benchmark。この論文では text-related tasks に絞る。
- **8-objective QA** ... 8 個の NaturalQuestions 由来質問を 1 task に束ね、search tool を使って consolidated answer set を出す benchmark。

## 読む順番の提案

- まず `text/0_abstract.tex` と `text/1_introduction_v2.tex` を読み、著者が問題を「unbounded context」と「heterogeneous agent context」として捉えている点を押さえる。正規ノートの Summary 冒頭の問題設定に対応する。
- 次に `figures/1_concept.tex` と `figures/3_method.tex` を見る。前者は context が蓄積して peak tokens が増える動機、後者は「No Compression 成功 vs Compression 失敗」から feedback を作る Acon の流れを示す。
- 手法は `text/3_method_v3.tex` の Eq. `eq:agent`, `eqn:cost`, `eqn:history_comp`, `eqn:obs_comp`, `eq:optim`, `eqn:prompt_opt` を順に読む。正規ノートの Method / objective / UT+CO の記述と対応する。
- 実験は `tables/1_appworld_gpt-4.1.tex`、`tables/3_4_officebench_qa.tex`、`tables/11_12_one_row.tex` を優先する。数値は本文の自然文より表を基準にする。特に 8-objective QA の dependency 削減率は本文と表の整合を確認する。
- 実装や限界は `text/999_appendix.tex` の Experimental Setup Details と Limitations & Future Work を読む。threshold、LoRA 設定、API cost、KV-cache overhead、model coverage limitation がまとまっている。
- 正規ノート `notes/arXiv-2510.00615v2.md` では Summary と Critical Thoughts がこの easy note の「実験・結果」「妥当性と限界」に対応する。数式と表番号を確認しながら読むと、正規ノートの評価コメントがどの TeX 根拠に依存しているか追いやすい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2510.00615v2/`
- 正規ノート: `notes/arXiv-2510.00615v2.md`
