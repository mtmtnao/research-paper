# Reflexion: Language Agents with Verbal Reinforcement Learning（LLM エージェントの重み更新なし verbal reinforcement learning）

- arXiv: https://arxiv.org/abs/2303.11366
- 一次ソース: ../papers/arXiv-2303.11366v4/
- 正規ノート: ../notes/arXiv-2303.11366v4.md

---

## 一言で言うと

Reflexion は、LLM エージェントを fine-tuning せず、失敗した軌跡と評価信号を自然言語の self-reflection に変換して episodic memory に保存し、次の試行の文脈として使う枠組みである。著者はこれを verbal reinforcement と呼び、ALFWorld、HotPotQA、HumanEval/MBPP/LeetcodeHardGym で ReAct や CoT などの baseline を改善すると主張する。

## 何を議論する論文か

- **問題設定**: ReAct、SayCan、Toolformer、HuggingGPT、generative agents、WebGPT のような LLM ベースの goal-driven agent は、外部環境で行動できる一方、trial-and-error から素早く学ぶ仕組みが弱い。従来の reinforcement learning は training samples と fine-tuning のコストが大きく、in-context examples だけでは長い trajectory のどこが悪かったかという credit assignment problem を扱いにくい。
- **対象範囲 / 仮定**: 対象は、LLM が text と action を生成し、環境・ヒューリスティック・LLM・unit tests などから評価信号を得られるタスクである。Actor は CoT や ReAct として実装され、LLM 本体の重みは更新しない。改善は memory に保存された self-reflection を次 trial の入力に加えることで起こす。
- **既存研究との差分**: Self-Refine や prompt optimization は主に single-generation task 向けで、著者は Reflexion が hidden constraints、decision making、binary reward、memory を同時に扱う点を表で比較している。Programming では AlphaCode、CodeT、Self-Debugging、CodeRL と比較し、Reflexion は test execution、debugging execution、self-generated tests、multiple languages、self-reflection を組み合わせる手法として位置づけられている（`main.tex` Related work の表）。
- **この論文で答えたい問い**: LLM の重みを更新せず、失敗から作った自然言語の「semantic gradient」を memory に保存するだけで、decision-making、reasoning、programming の性能を trial をまたいで改善できるか。

## 背景と前提

- **LLM agent**: この論文では、LLM が単に回答文を出すだけでなく、API、ゲーム環境、compiler、interpreter など外部環境に対する action も生成する主体を指す。
- **Policy と memory**: 従来の policy は主にモデルパラメータで決まるが、Reflexion では policy を LLM の Actor と memory の組として扱う。ここが「重み更新なしで policy を変える」という主張の中心である。
- **Credit assignment**: 失敗した trajectory の中で、どの action が後続の失敗を引き起こしたかを見つける問題である。著者は、scalar reward より自然言語の self-reflection のほうが「targeted changes in actions」を表しやすいと述べる。
- **Actor baseline**: Reasoning では Chain-of-Thought (CoT)、decision-making では ReAct、HotPotQA の search あり設定でも ReAct を使う。CoT (GT) は ground truth context $C_{gt}$ を与えて reasoning だけを切り出す設定である。
- **Evaluator の種類**: HotPotQA では exact match (EM) grading、ALFWorld では LLM による binary classification と hand-written heuristic、programming では self-generated unit tests と compiler/interpreter feedback を使う。
- **Programming の比較対象**: HumanEval と MBPP は自然言語仕様から function body を生成する benchmark で、Rust 版は MultiPL-E により Python benchmark を変換している。LeetcodeHardGym は著者が導入した benchmark で、GPT-4 の pre-training cutoff date として書かれた October 8, 2022 より後に公開された Leetcode hard-rated questions 40 問からなる。

## 提案手法

### コアアイデア

Reflexion は、Actor $M_a$、Evaluator $M_e$、Self-Reflection model $M_{sr}$ の 3 要素からなる。Actor は現在の状態観測と memory を条件に text/action を生成する。Evaluator は生成された trajectory を評価し、binary success/fail や scalar reward、あるいは unit-test feedback のような信号を返す。Self-Reflection model は、その評価信号、trajectory、既存 memory を使って、次 trial で役立つ自然言語の self-reflection を生成する。

重要なのは、LLM の weights は更新せず、更新されるのは外部の memory buffer である点である。著者は、self-reflective feedback が agent に改善方向を与えるため「`semantic' gradient signal」として働くと説明している。ただし、これは通常の勾配降下の数学的勾配ではなく、言語による改善指示を指す表現である。

### 重要な定義・数式

$$
\pi_{\theta}(a_i | s_i), \quad \theta = \{M_a, mem\}
$$

**式の意味**: Algorithm 1 に出てくる policy の初期化で、次の action $a_i$ は状態 $s_i$ に対する policy $\pi_{\theta}$ から生成される。Reflexion では $\theta$ を Actor $M_a$ と memory $mem$ の組として置く。

**記号の定義**:
- $\pi_{\theta}$ ... Actor が状態から action / generation を選ぶ policy
- $a_i$ ... $i$ 番目の action
- $s_i$ ... $i$ 番目の状態
- $M_a$ ... text と action を生成する Actor model
- $mem$ ... Self-Reflection model の出力を保存する memory

**この論文での役割**: Reflexion の中心的な再定義である。重み更新をしない代わりに、$mem$ を変えることで同じ Actor の次 trial のふるまいを変える。

$$
\tau_t = {[a_0, o_0, \dots a_i, o_i]}, \quad r_t = M_e(\tau_0)
$$

**式の意味**: Algorithm 1 では trial $t$ の trajectory $\tau_t$ を action と observation の列として生成し、Evaluator がそれを評価する。本文の first trial 説明では評価式が $r_t = M_e(\tau_0)$ と書かれているが、周辺文脈では各 trial の trajectory を Evaluator が採点する処理を説明している。

**記号の定義**:
- $\tau_t$ ... trial $t$ で生成された action と observation の trajectory
- $a_i$ ... trajectory 内の action
- $o_i$ ... action 後に環境から返る observation
- $r_t$ ... trial $t$ の scalar reward
- $M_e$ ... trajectory を評価する Evaluator model

**この論文での役割**: Reflexion は直接 policy gradient を計算せず、Evaluator の sparse/binary/scalar feedback を次の self-reflection の入力にする。ここが「失敗の軌跡を言語フィードバックへ増幅する」処理の入口である。

$$
mem \gets [sr_0], \quad \text{Append } sr_t \text{ to } mem
$$

**式の意味**: Algorithm 1 では初回の self-reflection $sr_0$ を memory に入れ、その後の trial でも $sr_t$ を $mem$ に追加する。本文では $sr_t$ を trial $t$ の verbal experience feedback と説明している。

**記号の定義**:
- $sr_0$ ... 初回 trial 後に生成される self-reflection
- $sr_t$ ... trial $t$ 後に生成される self-reflection
- $mem$ ... self-reflection を蓄積する episodic / long-term memory
- $\Omega$ ... 保存する experience 数の上限。本文では max context LLM limitations のため usually set to 1-3 と書かれている

**この論文での役割**: memory が次 trial の Actor 入力に入るため、失敗の自然言語要約が次の action choice や answer generation を制約する。ALFWorld と HotPotQA では memory size 3、programming では max memory limit 1 experience とされる。

$$
T = \{t_0, t_1, \dots, t_n\}
$$

**式の意味**: Programming 設定で、LLM が生成した unit tests から構文的に有効なものを抽出し、$n$ 個をサンプリングして test suite $T$ を作る。本文では $n$ を maximum of 6 unit tests に設定すると書かれている。

**記号の定義**:
- $T$ ... 内部評価に使う self-generated unit test suite
- $t_0, t_1, \dots, t_n$ ... 個々の unit test
- $n$ ... test suite に含める test 数。最大 6

**この論文での役割**: Programming では Evaluator の質が性能を左右する。HumanEval Python では false positive rate が低く Reflexion が 91.0 pass@1 に達する一方、MBPP Python では self-generated tests の false positive が高く、Reflexion が GPT-4 baseline を下回る。

### 実装 / アルゴリズム上の要点

- step1: $M_a$, $M_e$, $M_{sr}$ を初期化し、policy を $\theta=\{M_a, mem\}$ として置く。
- step2: Actor が環境と相互作用して trajectory $\tau_t$ を作る。ALFWorld と HotPotQA ReAct では thought/action/observation の列、programming では function implementation と実行結果が主な軌跡になる。
- step3: Evaluator が $\tau_t$ を評価する。HotPotQA は EM、ALFWorld は LLM binary classification と heuristic、programming は self-generated unit tests と compiler/interpreter feedback を用いる。
- step4: Self-Reflection model が trajectory、reward、memory を見て natural language feedback を生成する。
- step5: 生成された $sr_t$ を memory に追加し、環境を reset して次 trial に入る。本文説明では Evaluator が正しいと判断するまで loop するとされ、実験では task ごとに max trials や連続失敗停止条件が置かれている。
- step6: Memory は context 長制約のため上限を持つ。ALFWorld では last 3 self-reflections、HotPotQA では 3 experiences、programming では 1 experience と書かれている。

## 実験・結果

- **データセット / ベンチマーク**: ALFWorld 134 environments、HotPotQA 100 questions、HumanEval、MBPP、LeetcodeHardGym を用いる。HotPotQA 自体は 113k question-and-answer pairs の Wikipedia-based dataset と説明される。LeetcodeHardGym は 40 Leetcode hard-rated questions in 19 programming languages である。
- **比較対象 / baseline**: ALFWorld は ReAct、HotPotQA は ReAct-only、CoT-only、CoT (GT)-only、および episodic memory (EPM) ablation、programming は GPT-4 single code generation sample、CodeT + GPT-3.5、CodeT + Codex などが表に出る。Related work では Self-Refine、Beam search、AlphaCode、CodeT、Self-Debugging、CodeRL との機能比較も行う。
- **指標**: ALFWorld は solved tasks の累積割合と失敗理由、HotPotQA は exact match による accuracy、programming は pass@1 accuracy と unit test 評価の TP/FN/FP/TN を使う。
- **主な結果**: ALFWorld では ReAct + Reflexion が simple heuristic を使って 134 tasks 中 130 tasks を完了し、12 iterative learning steps で ReAct baseline を absolute 22% 上回る。ReAct-only は trials 6-7 の間で性能増加が止まると書かれている。
- **主な結果**: HotPotQA では Reflexion が 100 questions で ReAct/CoT 系 baseline を上回る。baseline は temperature 0.7 で retry しても first trial で失敗した task を subsequent trials で解けなかったとされる。CoT (GT) は ground truth context を持っても 39% の questions を誤り、Reflexion により accuracy が 14% 改善する。EPM ablation では self-reflection が episodic memory learning advantage の上に 8% absolute boost を与える。
- **主な結果**: Programming の Table `tbl:programming:success` では、HumanEval (PY) は GPT-4 80.1 に対して Reflexion 91.0、HumanEval (RS) は 60.0 に対して 68.0、MBPP (PY) は GPT-4 80.1 に対して Reflexion 77.1、MBPP (RS) は 70.9 に対して 75.4、Leetcode Hard (PY) は 7.5 に対して 15.0 である。MBPP Python だけは Reflexion が GPT-4 baseline を下回る。
- **主な結果**: Table `tbl:programming:failures` では、HumanEval (PY) は Base 0.80 / Reflexion 0.91 / TP 0.99 / FN 0.40 / FP 0.01 / TN 0.60、MBPP (PY) は Base 0.80 / Reflexion 0.77 / TP 0.84 / FN 0.59 / FP 0.16 / TN 0.41、HumanEval (RS) は Base 0.60 / Reflexion 0.68 / TP 0.87 / FN 0.37 / FP 0.13 / TN 0.63、MBPP (RS) は Base 0.71 / Reflexion 0.75 / TP 0.84 / FN 0.51 / FP 0.16 / TN 0.49 である。
- **主な結果**: Programming ablation Table `tbl:programming:ablation` は HumanEval Rust 50 hardest problems で、Base model 0.60、Test generation omission 0.52、Self-reflection omission 0.60、Reflexion 0.68 を示す。
- **追加モデル評価**: Appendix `othermodels.tex` では HumanEval Python の starchat-beta が Baseline 0.26 / Reflexion 0.26 で改善しない。一方、HotPotQA 100 questions では CoT (GT) + text-davinci-003 が 0.60 から 0.77、gpt-3.5-turbo が 0.57 から 0.71、gpt-4 が 0.68 から 0.80、ReAct + text-davinci-003 が 0.30 から 0.55、gpt-3.5-turbo が 0.26 から 0.38、gpt-4 が 0.39 から 0.51 に改善する。
- **著者が主張する貢献**: (1) policy を agent memory encoding と LLM parameters の組として parameterize する verbal reinforcement の paradigm、(2) LLM の self-reflection が handful of trials で complex tasks を学ぶのに有用であること、(3) LeetcodeHardGym の導入、(4) 複数タスクで strong baselines を改善し code generation benchmarks で state-of-the-art results を達成したこと。

## 妥当性と限界

- **この主張を支える根拠**: 3 種類のタスク、すなわち sequential decision-making、single-step / search-based reasoning、programming で同じ Actor/Evaluator/Self-Reflection/memory の枠組みを使い、いずれも baseline と比較して改善を示している。特に ALFWorld では長い trajectory の early mistake を self-reflection が「self-hints」に蒸留するという分析、HotPotQA では EPM だけでなく first-person self-reflection を追加した ablation、programming では test generation と self-reflection の両方を外す ablation が主張を支える。
- **この主張を支える根拠**: Programming では成功例だけでなく、MBPP Python の失敗分析を行い、false positive test execution rate が MBPP Python 16.3%、HumanEval Python 1.4% であることを理由に性能差を説明している。これは Evaluator の質が Reflexion の上限を決めるという読み方につながる。
- **著者が認めている limitations / future work**: Reflexion は natural language による policy optimization だが、non-optimal local minima に陥り得る。長期記憶は maximum capacity を持つ sliding window に限定され、future work として vector embedding databases や traditional SQL databases への拡張を挙げている。Code generation では test-driven development が non-deterministic generator functions、APIs と相互作用する impure functions、hardware specifications に依存する outputs、parallel / concurrent behavior を扱いにくい。
- **著者が認めている limitations / future work**: Appendix の WebShop Limitation では、100 environments の two-shot ReAct + Reflexion agent を 4 trials 後に停止し、ReAct を有意に上回らず、helpful self-reflections も生成できなかったと述べる。著者は、WebShop のように diversity and exploration が非常に必要なタスクでは Reflexion が local minima を抜けにくいと結論づけている。
- **読者として注意すべき点**: Reflexion は「評価器が失敗を正しく見つけ、それを LLM が有用な反省に変換できる」ことに強く依存する。ALFWorld の heuristic は同一 action/response が 3 cycles 超、または 30 actions 超という手作りの停止条件であり、HotPotQA の EM は意味的に同じ別表現を誤りにする可能性がある。これらは TeX の評価設定から分かる注意点である。
- **読者として注意すべき点**: Appendix `othermodels.tex` の starchat-beta 結果から、著者は self-corrections を指定する能力を stronger, larger models の emergent quality と書く。ただし TeX 中で検証されている programming の小モデル例は starchat-beta の HumanEval Python であり、モデルサイズ、訓練データ、instruction following のどれが主要因かまでは切り分けられていない。
- **追加で確認したい実験 / 疑問**: 同じ token / API call 予算で Reflexion と sampling、beam search、CodeT などを比較した cost-accuracy は TeX 中には示されていない。Memory size $\Omega$ の網羅的 sweep も TeX 中には示されていない。Evaluator の false positive を下げたとき MBPP Python が GPT-4 baseline を超えるかも追加確認したい点である。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Reflexion**: LLM weights を更新せず、task feedback signals を verbal self-reflection に変換して memory に保存し、次 trial の Actor 入力に戻す枠組み。
- **verbal reinforcement**: reward を直接 gradient update に使うのではなく、自然言語の feedback として LLM に与える強化の形。著者は「semantic gradient signal」とも表現する。
- **Actor $M_a$**: text と actions を生成するモデル。実験では CoT や ReAct が Actor の役割を担う。
- **Evaluator $M_e$**: Actor の出力 trajectory を採点するモデルまたは関数。EM、LLM binary classification、hand-written heuristic、self-generated unit tests などがある。
- **Self-Reflection model $M_{sr}$**: sparse reward、trajectory、memory から、次 trial のための具体的な verbal self-reflection を作る LLM。
- **trajectory $\tau_t$**: trial $t$ における action と observation の列。Algorithm 1 では $\tau_t={[a_0,o_0,\dots,a_i,o_i]}$ と書かれる。
- **episodic memory / long-term memory**: Self-Reflection model の出力を保存する buffer。短期記憶は trajectory history、長期記憶は self-reflection と説明される。
- **$\Omega$**: memory に保存する experience 数の上限。通常 1-3 と書かれる。
- **CoT (GT)**: Chain-of-Thought に ground truth context $C_{gt}$ を与え、検索や行動選択ではなく long context 上の reasoning を見る設定。
- **EPM ablation**: HotPotQA で、self-reflection ではなく most recent trajectory を episodic memory として含める ablation。self-reflection は EPM の利得の上に 8% absolute boost を与えるとされる。
- **pass@1**: Programming benchmark で、最初の提出が正しい割合。Table `tbl:programming:success` の主指標。
- **TP / FN / FP / TN**: 内部 unit tests と真の solution pass/fail の関係。FP は tests pass だが solution fail で、Reflexion では誤った提出を早期に通してしまうため特に問題になる。
- **LeetcodeHardGym**: 著者が導入した code-generation RL gym environment。40 hard-level Leetcode questions、19 programming languages、October 8, 2022 より後に公開された問題からなる。
- **WebShop Limitation**: Appendix で示される失敗例。WebShop では search query の曖昧性や多様な探索が必要で、Reflexion は ReAct を有意に上回らなかった。

## 読む順番の提案

- まず `main.tex` の Abstract と Introduction を読み、問題設定を「LLM agent は trial-and-error から素早く学べない」「self-reflective feedback が semantic gradient として働く」という形で押さえる。正規ノートでは `Summary（著者の主張）` の最初の 2 bullet に対応する。
- 次に `main.tex` Section 3 `Reflexion: reinforcement via verbal reflection` と Algorithm 1 を読む。Actor $M_a$、Evaluator $M_e$、Self-Reflection $M_{sr}$、memory $mem$、$\Omega$ の関係を確認する。正規ノートでは `Takeaway` の「policy を LLM 重み + テキスト記憶と再定義」に対応する。
- 実験は ALFWorld、HotPotQA、Programming の順に読む。ALFWorld では heuristic 条件（3 cycles / 30 actions）と 130/134、HotPotQA では CoT / ReAct / CoT (GT) と EPM ablation、Programming では Table `tbl:programming:success`、`tbl:programming:failures`、`tbl:programming:ablation` を必ず照合する。
- Appendix は必要箇所だけ読む。`additional_information/othermodels.tex` はモデルサイズ・モデル種類による改善差、`decisionmaking.tex` は ALFWorld の反省例と WebShop limitation、`programming.tex` は code generation prompt 形式、`reasoning.tex` は HotPotQA の ReAct / CoT / CoT (GT) 例に対応する。
- 最後に Limitations と WebShop Limitation を読み、正規ノートの `Critical Thoughts（評価・疑問）` と照らす。ここで、著者の限界認識と読者側の追加疑問を分けて読むと、主張と評価が混ざりにくい。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2303.11366v4/`
- 正規ノート: `notes/arXiv-2303.11366v4.md`
