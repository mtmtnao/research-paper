# Improving Factuality and Reasoning in Language Models through Multiagent Debate（黒箱 LLM の推論・事実性を multi-agent debate で改善する prompting 研究）

- arXiv: https://arxiv.org/abs/2305.14325
- 一次ソース: ../papers/arXiv-2305.14325v1/
- 正規ノート: ../notes/arXiv-2305.14325v1.md

---

## 一言で言うと

複数の LLM インスタンスを agent として同じ問題に答えさせ、他 agent の応答を context として渡しながら複数 round の debate を行うことで、単一 agent の生成より reasoning と factual validity を改善できるかを調べる論文。著者は、3 agents・2 rounds を主な設定として、Arithmetic / GSM8K / Chess Move Prediction / Biographies / MMLU / Chess Move Validity の 6 タスクで Multi-Agent (Debate) が比較手法を上回ると報告している（tables/reason.tex, tables/accuracy.tex）。

## 何を議論する論文か

- **問題設定**: LLM は chain of reasoning で不自然な飛躍をしたり、事実生成で hallucination を起こしたりする。既存の prompting、verification、self-consistency、scratchpads などは主に単一 model instance に対する改善であり、この論文は複数 instance の相互批判で同じ問題を改善できるかを問う（text/introduction.tex）。
- **対象範囲 / 仮定**: 生成結果だけを得られる black-box LLM を対象にする。likelihood や gradient などの model-internal information は使わない。実験は Appendix で `gpt-3.5-turbo-0301` を使ったと明記され、異種モデル実験だけ chatGPT と Bard の組み合わせを扱う（text/appendix.tex, text/experiments.tex）。
- **既存研究との差分**: Chain-of-Thought、self-reflection / Self-Refine / Reflexion、多数決型の self-consistency などは単一 agent の推論や複数サンプルの集約に寄る。一方、本論文では他 agent の回答を LLM 自身に読ませ、critique と更新を複数 round 繰り返す点が差分になる。Related Work では、Irving et al. の AI safety via debate は人間が議論を評価する設定であり、本論文とは異なると位置づけている（text/related.tex, main.bbl）。
- **この論文で答えたい問い**: 実験節の冒頭で、multiagent debate が reasoning をどの程度改善するか、factual validity をどの程度改善するか、どの design choices が性能改善に効くか、という 3 問を掲げている（text/experiments.tex）。

## 背景と前提

- **LLM の生成と hallucination**: Introduction は、LLM が大量のインターネット上の text corpus で訓練されるため、抽出された自然言語の品質や正確さが保証されないことを背景に置く。そのため、事実を自信ありげに誤る、あるいは reasoning chain で implausible jumps をする問題がある。
- **agent の意味**: この論文での agent は、人間の参加者ではなく、同じまたは異なる language model instance のこと。主な実験では同じ chatGPT-based language model の複数コピーを使う。
- **debate の意味**: 各 agent が初期回答を作り、その後、他 agent の応答を context として読んで自分の回答を更新する手続き。論文は convergence が一般には保証されない multi-agent game として見られると述べつつ、経験的には複数 round 後に単一の共有回答へ収束しやすいと報告する（text/method.tex）。
- **baseline との関係**: reasoning では Single Agent、Single Agent (Reflection)、Multi-Agent (Majority)、Multi-Agent (Debate) を比較する。factuality では個別応答が比較しにくいとして Multi-Agent (Majority) を外し、Single Agent、Single Agent (Reflection)、Multi-Agent (Debate) を比較する（text/experiments.tex）。
- **prompt の読み方**: Abstract と Introduction では同じ procedure / prompt templates を使うと書かれるが、Appendix の tables/prompt_settings.tex にはタスク別の Starting prompt と Debate prompt が載る。したがって、全タスクで一字一句同じ prompt というより、同じ debate の型を各タスク用 template に入れて使う、という理解が安全。

## 提案手法

### コアアイデア

本論文の multiagent debate は、複数の LLM agent が候補回答を並列に生成し、その後に他 agent の回答を読んで自分の回答を更新する生成手続きである。各 response は「別の thought process」または「別の information source」として扱われ、agent は他 agent の responses を検証し、自分の response を refine する役割を持つ（text/method.tex）。

具体的には、まず各 agent に同じ query を独立に解かせる。次に、他 agent の responses を連結して context として渡し、consensus prompt によって更新回答を生成させる。この consensus prompt を、更新済み responses を使ってさらに複数 round 繰り返す。図としては figText/debate_overview.tex が procedure の概観、figText/prompt_debate_duration.tex が短い consensus prompt と長い consensus prompt の例を示す。

著者は、この方法が retrieval や prompt engineering、zero-shot Chain-of-Thought と直交すると述べ、実験では debate と zero-shot Chain-of-Thought を組み合わせている（text/introduction.tex, text/method.tex, figText/prompt_ablation.tex）。

### 重要な定義・数式

TeX 中に、multiagent debate の目的関数、確率モデル、更新式のような中核的な明示式は少ない。手法本体は数式ではなく、agent の初期生成、他 agent responses の context 化、consensus prompt による更新、round の反復として定義されている。以下の式は method §2.1 の導入例であり、手法の最適化式ではない。

$$
0.5 \times 3 \times 4 = 64
$$

**式の意味**: 3, 4, 5 の三角形を直角三角形と見なし、底辺と高さから面積を計算する例として TeX に書かれている。算術上は 6 になるが、TeX では `64` と記されているため、ここでは原文の表記をそのまま示す。

**記号の定義**:
- $0.5$ ... 三角形の面積公式における $1/2$
- $3, 4$ ... 直角をはさむ辺として扱われる長さ
- $64$ ... TeX 上に書かれた計算結果

**この論文での役割**: 複数の解法で同じ問題を解き、答えが一致すれば信頼が上がり、一致しなければ reasoning を cross-examine する、という multi-threaded reasoning の導入例である。multiagent debate の formal definition ではない。

$$
0.5 \times 3 \times 4 \times \sin(\theta)
$$

**式の意味**: 同じ三角形の面積を、角度 $\theta$ と $\sin(\theta)$ を使って別経路で求める例として示される。Law of Cosines で $\theta$ を推定し、その角度を使って面積を出す、という説明に対応する（text/method.tex）。

**記号の定義**:
- $\theta$ ... 三角形内の角度として導入される量
- $\sin(\theta)$ ... 角度 $\theta$ の sine
- $3, 4$ ... 面積計算に使われる 2 辺の長さ

**この論文での役割**: 1 つの問題に対して複数の reasoning path を持ち、それらを照合するという比喩的な役割を持つ。LLM agent では、各 agent の response がこの複数 path / 複数 source に相当する。

### 実装 / アルゴリズム上の要点

- step1: 複数 agent を用意し、同じ query に対して各 agent が独立に initial response を生成する。
- step2: ある agent を更新するとき、他 agent の responses を context として渡す。本文では基本的に concatenation と説明される。
- step3: consensus prompt により、他 agent の responses を追加情報として使い、更新回答を生成させる。
- step4: 更新後 responses を使って同じ debate procedure を複数 round 繰り返す。
- step5: 実験の主要設定では、計算コストのため 3 agents・2 rounds を中心に評価する（tables/reason.tex の caption, text/experiments.tex）。
- step6: agent 数が増えて context length error が出る場合、全 agent responses を chatGPT で summarize してから各 agent に渡す。これは 5 agents 以上を扱うために使われ、figText/summarize.tex では performance も改善すると説明される。
- step7: consensus の速さは prompt によって変わる。short prompt は consensus を早め、longer / more stubborn な prompt は収束を遅くするが最終精度を高める、と著者は報告する（text/method.tex, text/experiments.tex, figText/convergence_analysis.tex, figText/consensus.tex）。

## 実験・結果

- **データセット / ベンチマーク**:
  - `Arithmetic`: 本文では addition, multiplication, subtraction を含む 6 個の two-digit numbers の式と説明される。一方、Appendix では各 task に 0 から 30 の random integers を 6 個生成し、100 generated arithmetic tasks で評価したと書かれている（text/experiments.tex, text/appendix.tex）。
  - `GSM8K`: grade school mathematical reasoning tasks。Appendix では 100 grade school math problems を使い、boxed answer から最終回答を抽出して accuracy を測る。
  - `Chess Move Prediction`: grand-master games の最初の 14 moves を PGN notation で与え、次の best move を予測させる。Appendix では `https://www.pgnmentor.com/players/Adams.zip` の games、white to move at turn 14、Stockfish search depth 20、300 selected chess games と説明される。
  - `Biographies`: 524 well-known computer scientists の ground truth bullet point biographies を構成し、人物ごとの bullet point biography を生成させる新規 factuality task。
  - `MMLU`: exams で学習・評価される factual knowledge questions への応答を測る既存データセットとして使われる。Appendix では各 subject area に random に分布する 100 selected MMLU questions で評価したと書かれる。
  - `Chess Move Validity`: BIG-Bench Chess-State Tracking Benchmark の hardest reported task `synthetic_short` を使い、100 selected chess validity tasks で valid destination square を答えさせる。
- **比較対象 / baseline**:
  - reasoning: Single Agent、Single Agent (Reflection)、Multi-Agent (Majority)、Multi-Agent (Debate)。
  - factuality: Single Agent、Single Agent (Reflection)、Multi-Agent (Debate)。Multi-Agent (Majority) は、個別応答が直接比較しにくいとして除外される。
- **指標**:
  - Arithmetic / GSM8K: final answer accuracy。
  - Chess Move Prediction: predicted move を実行した後の Stockfish 推定 relative pawn score、表では Chess ($\Delta$PS)。
  - Biographies: generated biography が ground truth bullet とどの程度一致するか。Appendix では chatGPT に yes / no / uncertain で consistency を判定させ、uncertain を無視すると説明される。
  - MMLU: correct multiple-choice answer を選べた accuracy。
  - Chess Move Validity: generated answer が valid answers の 1 つなら correct。
- **主な結果**:
  - Table 1 / `tbl:reasoning`: Arithmetic は Single Agent 67.0 ± 4.7、Reflection 72.1 ± 4.5、Majority 69.0 ± 4.6、Debate 81.8 ± 2.3。Grade School Math は 77.0 ± 4.2、75.0 ± 4.3、81.0 ± 3.9、85.0 ± 3.5。Chess ($\Delta$PS) は 91.4 ± 10.6、102.1 ± 11.9、102.2 ± 6.2、122.9 ± 7.6。
  - Table 2 / `tbl:accuracy`: Biographies は Single Agent 66.0 ± 2.2、Reflection 68.3 ± 2.9、Debate 73.8 ± 2.3。MMLU は 63.9 ± 4.8、57.7 ± 5.0、71.1 ± 4.6。Chess Move Validity は 29.3 ± 2.6、38.8 ± 2.9、45.2 ± 2.9。
  - agent 数の分析では、2 rounds に固定して agent 数を増やすと Arithmetic の performance が単調に上がると報告される。round 数の分析では、3 agents に固定して debate length を伸ばすと Arithmetic performance が単調に上がるが、4 rounds を超える追加 round は 4 rounds と類似の最終性能になると述べる（text/experiments.tex, figText/debate_agents.tex）。
  - different initialization prompts として professor / doctor / mathematician の persona を MMLU で使うと、71.1 から 74.2 に改善したと報告される（text/experiments.tex）。
  - chatGPT と Bard の異種モデル debate では、20 GSM8K math problems に対して Bard が 11 問、chatGPT が 14 問、joint multi-agent debate が 17 問を解いたと報告される（text/experiments.tex, figText/debate_between_models.tex）。
- **著者が主張する貢献**:
  - multi-agent debate process による factual correctness と reasoning accuracy の改善手法を提示した。
  - contemporary language models が苦手とする factual correctness benchmark として biographies task を導入した。
  - 6 つの reasoning / factual accuracy tasks で、agent 数、debate rounds、prompt、summarization、persona、異種モデル混合を分析した（text/introduction.tex, text/experiments.tex）。

## 妥当性と限界

- **この主張を支える根拠**:
  - reasoning と factuality の両方で、同じ starting prompt と language model を使って baseline と Debate を比較すると説明される（text/experiments.tex）。
  - 表の主要結果では、6 タスクすべてで Multi-Agent (Debate) が Single Agent を上回る。reasoning では Majority と Reflection も含む比較があり、factuality では Reflection が MMLU で 63.9 から 57.7 に下がる一方、Debate は 71.1 まで上がる。
  - qualitative examples では、全 agent が初期に間違っていても debate 中の critique により正解へ到達する場合があると著者は述べる（text/experiments.tex, figText/math_overview.tex, figText/debate_between_models.tex）。
  - factuality では、不確実な事実について agent 間で回答が分かれ、debate によってより accurate な consensus answer に変わると報告する。ただし figText/uncertainty_generation.tex は、必ず factual に正しいとは限らないとも明記している。
- **著者が認めている limitations / future work**:
  - multiagent debate は複数 generation と debate procedure を必要とするため、他の prompting techniques より computationally expensive である（text/discussion.tex）。
  - debate が長くなると、現行 LLM は debate input 全体を十分処理できず、最近の generations に偏って注目することがある。
  - debates は通常 single final answer に収束するが、その answer が正しいとは限らない。誤答でも model は正しく一貫していると自信を持って affirm することがある。
  - 著者は、debate で得た追加データを distill して base model の self-improvement に戻す可能性、longer-context / improved language models / early debate summarization による改善可能性を discussion で述べる。
- **読者として注意すべき点**:
  - convergence は method 節で一般には保証されないと明記される。経験的に収束しやすいことと、正解に収束することは別である。
  - Biographies の評価は chatGPT による consistency 判定に依存し、Appendix 自身も、generated facts に ground truth bullet で捕捉されない誤情報が含まれうると述べる。
  - factuality では Majority baseline が除外されているため、reasoning の表と同じ比較構造ではない。
  - Arithmetic の task 説明は、本文の six two-digit numbers と Appendix の integers between 0 and 30 が完全には一致しない。
  - Table 1 / 2 では、同一 token 予算や同一計算予算で他手法と比較したとは TeX 中に明示されていない。
- **追加で確認したい実験 / 疑問**:
  - どの割合で「全 agent 初期不正解から正解へ到達」するのかは、TeX 中では定量表として示されていない。
  - Biographies 以外の open-ended factual generation、より低頻度の人物や最新情報で同じ傾向が出るかは TeX 中には示されていない。
  - chatGPT × Bard 実験は 20 GSM8K problems であり、異種モデル混合の一般性を主張するには追加タスクとサンプル数が必要に見える。
  - 著者が示唆する persuasion のしやすさを confidence proxy とする考えは興味深いが、calibration 指標としての定量評価は TeX 中には明示されていない。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **multiagent debate**: 複数の language model instances が個別回答を作り、他 agent の responses を読んで critique / update を繰り返し、最終的に共通回答へ向かう procedure。
- **agent**: この論文では LLM の instance。主実験では同じ chatGPT-based model の複数コピー、異種実験では chatGPT と Bard。
- **consensus prompt**: 他 agent の responses を追加情報として渡し、更新回答を作らせる prompt。short / long の違いで consensus speed と accuracy が変わると分析される。
- **round**: 各 agent が他 agent の更新済み responses を見て自分の response を更新する debate の反復単位。主実験は 2 rounds。
- **black-box access**: likelihood や gradient ではなく、language model generations のみを使える設定。一般的な public model serving interfaces で使えることを著者は利点とする。
- **Single Agent (Reflection)**: 生成後に同じ model に自分の response を critique / refine させる baseline。factuality では一部で性能が下がる。
- **Multi-Agent (Majority)**: 複数 agent の生成結果から多数決を取る baseline。reasoning では比較されるが、factuality では応答が直接比較しにくいとして外される。
- **GSM8K**: grade school math word problems の benchmark。main.bbl では Cobbe et al. 2021 の Training verifiers to solve math word problems と対応する。
- **MMLU**: Massive Multitask Language Understanding。本文では factual knowledge questions として使われ、main.bbl では Hendrycks et al. 2020 と対応する。
- **Chess ($\Delta$PS)**: Chess Move Prediction の評価列。suggested move 後の Stockfish 推定 pawn score advantage を報告する。
- **Biographies**: 著者らが構成した 524 well-known computer scientists の bullet biography factuality task。評価は ground truth bullets と generated bullets の一致に基づく。
- **ease of persuasion**: 不確実な事実では agent が意見を変えやすく、確信のある事実では変えにくいという観察から、factual confidence の評価に使える可能性があると著者が述べる概念（text/experiments.tex）。
- **summarization**: agent 数が多いと context が長くなるため、他 agent responses を直接 concatenation する代わりに chatGPT で summarize して渡す設計。figText/summarize.tex では context length を減らし performance を改善すると説明される。
- **persona / different initialization prompts**: agent ごとに professor / doctor / mathematician のような role を与える設定。MMLU で 71.1 から 74.2 に改善したと報告される。

## 読む順番の提案

- まず `text/abstract.tex` と `text/introduction.tex` を読み、問題設定が「単一 model instance の改善」から「複数 model instances の相互批判」へ移る点を押さえる。正規ノートの Summary の「問題」「手法」に対応する。
- 次に `text/method.tex` と `figText/debate_overview.tex` を読み、初期回答、他 agent responses の context 化、consensus prompt、複数 round の反復を確認する。正規ノートの Takeaway にある consensus prompt / stubbornness の理解につながる。
- その後 `tables/reason.tex` と `tables/accuracy.tex` を見て、6 タスクの数値を先に把握する。正規ノートの Summary の表値と一致するか確認しやすい。
- 評価設計は `text/experiments.tex` の Tasks / Baselines / Results と `text/appendix.tex` の Evaluation Details を合わせて読む。特に Biographies の chatGPT 判定、Chess の Stockfish depth 20、MMLU / GSM8K の 100 selected problems は Appendix 側にある。
- design choices は `figText/debate_agents.tex`, `figText/convergence_analysis.tex`, `figText/summarize.tex`, `figText/debate_between_models.tex` を本文の Analysis subsection と一緒に読む。正規ノートの Takeaway の agent 数・round 数・summarization・persona・Bard 混合の記述に対応する。
- 最後に `text/discussion.tex` を読み、計算コスト、長い debate の context 処理、誤答への収束という限界を確認する。正規ノートの Critical Thoughts は、この限界と TeX に明示されていない追加疑問を分けて読むとよい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2305.14325v1/`
- 正規ノート: `notes/arXiv-2305.14325v1.md`
