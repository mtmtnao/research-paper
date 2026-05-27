# Self-Refine: Iterative Refinement with Self-Feedback（同一 LLM によるテスト時の反復自己改善）

- arXiv: https://arxiv.org/abs/2303.17651
- 一次ソース: ../papers/arXiv-2303.17651v2/
- 正規ノート: ../notes/arXiv-2303.17651v2.md

---

## 一言で言うと

Self-Refine は、単一の LLM $\mathcal{M}$ に「初期生成」「自分の出力への自然言語 feedback」「feedback に基づく refinement」を反復させる、追加学習なしの test-time 手法である。著者は 7 タスクと GPT-3.5 / ChatGPT / GPT-4 で、同じ LLM の one-step generation より平均で約 20% absolute 改善すると主張している（`sections/100_abstract.tex`, `tables/main_results.tex`）。

## 何を議論する論文か

- **問題設定**: LLM は coherent な出力を生成できるが、対話応答の複数品質やコード可読性のような「多面的」「定義しにくい」要件を 1 回の生成で満たせないことがある。論文は、候補出力を改善された候補へ反復的に写す iterative refinement を、追加の supervised refiner や reward model なしで行えるかを問う。
- **対象範囲 / 仮定**: 対象は自然言語生成・コード生成・数学推論を含む 7 タスクで、実験の主な base LLM は GPT-3.5 (`text-davinci-003`), ChatGPT (`gpt-3.5-turbo`), GPT-4 である。手法は few-shot prompt $\{p_{\text{gen}}, p_{\text{fb}}, p_{\text{refine}}\}$ を前提にし、モデルが prompt 形式に従えることを仮定する。著者自身も Limitations で「sufficient few-shot modeling or instruction-following abilities」が必要と書いている。
- **既存研究との差分**: 既存の refinement には、ドメイン固有データで refiner を訓練する方法、外部 reward / human annotations に依存する方法、compiler や Wikipedia edits のような task-specific feedback を使う方法がある。Self-Refine は、同一 LLM が自分の出力に自然言語 feedback を出し、その feedback で自分の出力を直す点を差分としている（`tab:related_work_short_summary` / `sections/700_related.tex`）。
- **この論文で答えたい問い**: 強い LLM は、自分の初期出力の欠点を自然言語で具体的に指摘し、その feedback を使って自分の出力を改善できるのか。さらに、その改善は単に複数サンプルを出す効果ではなく、feedback と refinement の反復によるものなのか。

## 背景と前提

- **few-shot prompting / in-context learning**: 論文では、各 module を追加学習ではなく few-shot prompt で実装する。$p_{\text{gen}}$ は input-output pairs、$p_{\text{fb}}$ は input-output-feedback triples、$p_{\text{refine}}$ は input-output-feedback-refined quadruples を含む（`sections/300_method-alt.tex`）。
- **test-time refinement**: モデル重みを更新せず、推論時に出力 $y_t$ と feedback $fb_t$ の履歴を prompt に連結して次の出力を得る。したがって、訓練データを増やす話ではなく、同じ model API の呼び出し方を変える話である。
- **自然言語 feedback**: この論文での feedback は、単なる scalar reward ではなく、出力のどこをどう直すべきかを自然言語で述べるもの。著者は "actionable" を「改善に結びつく具体的行動を含む」、"specific" を「変更すべき具体的な phrase を特定する」と説明している。
- **baseline との関係**: 主比較は「同じ base LLM に 1 回だけ生成させる Direct / Base」と「同じ base LLM に Self-Refine を適用する」比較である。別モデルを追加した ensemble ではない。
- **評価の注意**: Dialogue Response, Sentiment Reversal, Acronym Generation などは preference 評価を使う。GPT-4-pref は人間評価の proxy として用いられ、著者は human-pref との相関を Sentiment 82%, Acronym 68%, Dialogue 71% と報告している（`sections/500_results.tex`）。

## 提案手法

### コアアイデア

Self-Refine は、同一の LLM $\mathcal{M}$ を generator, feedback provider, refiner として使う。まず入力 $x$ から初期出力 $y_0$ を生成し、次に $y_t$ に対する feedback $fb_t$ を同じ $\mathcal{M}$ で生成し、最後に $x$、過去出力、過去 feedback を prompt に連結して $y_{t+1}$ を得る。これを最大反復回数または task-specific stop condition まで繰り返す。

論文の重要な設計は、refinement 時に直近の $y_t, fb_t$ だけでなく、履歴 $y_0, fb_0, \dots, y_t, fb_t$ を保持することである。著者はこれによりモデルが過去の mistakes から学び、繰り返しを避けられると説明している（Eq. (4), Algorithm 1）。

### 重要な定義・数式

#### 初期生成

$$
y_0 = \mathcal{M}\left(p_{\text{gen}} \| x\right)
$$

**式の意味**: 入力 $x$ と generation 用 prompt $p_{\text{gen}}$ を連結し、同一 LLM $\mathcal{M}$ から初期出力 $y_0$ を得る式である。TeX では Eq. (1) / `eq:gen` として置かれている。

**記号の定義**:
- $x$ ... 解くべき入力列。例として dialogue context, Python program, math word problem など。
- $p_{\text{gen}}$ ... 初期生成用の task-specific few-shot prompt。
- $\mathcal{M}$ ... base LLM。主実験では GPT-3.5, ChatGPT, GPT-4。
- $\|$ ... 文字列または prompt 部分の concatenation。
- $y_0$ ... 最初の draft / candidate output。

**この論文での役割**: `tab:results` の Base はこの one-step generation に対応する。Self-Refine の改善量は、まずこの $y_0$ と反復後の出力を比べる形で評価される。

#### 自己 feedback

$$
fb_t = \mathcal{M}\left(p_{\text{fb}} \| x \| y_t\right)
$$

**式の意味**: 同じ LLM $\mathcal{M}$ に、feedback 用 prompt、元入力、現在の出力を渡して、自然言語 feedback $fb_t$ を生成する。TeX では Eq. (2) / `eq:feedback`。

**記号の定義**:
- $t$ ... feedback-refine loop の反復 index。
- $y_t$ ... 反復 $t$ 時点の出力。
- $p_{\text{fb}}$ ... feedback 生成用 prompt。input-output-feedback triples を含む。
- $fb_t$ ... $y_t$ に対する feedback。論文では actionable かつ specific であることを重視する。

**この論文での役割**: Self-Refine の中心的主張は、この $fb_t$ が外部の人間や reward model ではなく、同じ LLM から得られる点にある。`tab:ablation_feedback` の ablation は、この feedback の質が性能に効くことを検証している。

#### 直近 feedback に基づく refinement

$$
y_{t+1} = \mathcal{M}\left(p_{\text{refine}} \| x \| y_t \| fb_t\right)
$$

**式の意味**: refine 用 prompt、元入力、現在の出力、現在の feedback を連結し、次の出力 $y_{t+1}$ を得る式である。TeX では Eq. (3) / `eq:refine`。

**記号の定義**:
- $p_{\text{refine}}$ ... feedback を使って出力を改善するための few-shot prompt。
- $y_{t+1}$ ... feedback を取り込んだ次の candidate output。
- $x, y_t, fb_t, \mathcal{M}$ ... 上と同じ。

**この論文での役割**: feedback が単なる採点で終わらず、実際に出力を変更する module である。Code Optimization では nested loops の遅さを指摘する feedback を受けて、より速い実装に変える例が `fig:analysis_example` に示されている。

#### 履歴つき refinement

$$
y_{t+1} = \mathcal{M}\left(p_{\text{refine}} \| x \| y_0 \| fb_0 \| ... \| y_t \| fb_t\right)
$$

**式の意味**: 実際の instantiation では、直近だけでなく過去の全 output-feedback pairs を prompt に入れて refine する。TeX では Eq. (4) / `eqn:refine2`。

**記号の定義**:
- $y_0, \dots, y_t$ ... 初期出力から現在までの候補出力列。
- $fb_0, \dots, fb_t$ ... 各出力に対して生成された feedback の列。
- $p_{\text{refine}}, x, \mathcal{M}$ ... 上と同じ。

**この論文での役割**: Algorithm 1 の refine step に対応し、反復型手法としての Self-Refine を定義する式である。反復が増えるほど prompt が長くなるため、計算コストと context 長は読者が注意すべき実装上の点になる。

### 実装 / アルゴリズム上の要点

- step1: 入力 $x$、model $\mathcal{M}$、3 つの prompts $\{p_{\text{gen}}, p_{\text{fb}}, p_{\text{refine}}\}$、stop condition $\mathrm{stop}(\cdot)$ を用意する。
- step2: $y_0=\mathcal{M}(p_{\text{gen}}\|x)$ で初期出力を作る。
- step3: 各反復で $fb_t=\mathcal{M}(p_{\text{fb}}\|x\|y_t)$ を生成する。
- step4: $\mathrm{stop}(fb_t,t)$ が真なら break する。停止条件は指定 timestep、または feedback から抽出される scalar stop score など task-specific に決める。
- step5: 停止しない場合は履歴つき prompt で $y_{t+1}$ を生成する。
- step6: 最後の refinement $y_t$ を出力する。ただし Acronym Generation のような multi-aspect feedback タスクでは、品質が単調増加しないため、feedback が生成する数値 score に基づき全 iteration から出力を選ぶ説明が付録にある。
- 実験では feedback-refine iterations は「desired output quality or task-specific criterion」に達するまで、最大 4 iterations と書かれている（`sections/400_tasks.tex`）。Code Readability 付録では budget constraints により $N=5$ iterations の実験も記載されている。
- 本文は "We use greedy decoding with a temperature of 0.7" と書く。greedy と temperature 0.7 の語は厳密には同時に使いにくいが、ここでは TeX の表現どおりに記録する。

## 実験・結果

- **データセット / ベンチマーク**: 7 タスクで評価する。内訳は Sentiment Reversal（Zhang et al., 1000 review passages）、Dialogue Response Generation（Mehri and Eskenazi, `tab:task_descriptions` では 372 conversations、`task_details/responsegen.tex` では automatic eval 342 instances / human eval 100 instances。別途 `sections/appendix/ab_eval.tex` の表キャプションには 150 examples for each dataset とあり、human A/B のサンプル数は TeX 内で記述ゆれがある）、Code Optimization（PIE, 1000 programs）、Code Readability Improvement（CodeNet, 300 programs、human subset 60）、Math Reasoning（GSM8K, 1319 questions）、Acronym Generation（250 acronyms、本論文で導入）、Constrained Generation / CommonGen-Hard（200 samples、本論文で導入、20-30 keyword constraints）。
- **比較対象 / baseline**: 主 baseline は同じ LLM の Base / Direct generation。base LLM は GPT-3.5 (`text-davinci-003`), ChatGPT (`gpt-3.5-turbo`), GPT-4。code-based tasks では Codex (`code-davinci-002`) も扱う。
- **指標**: GSM8K は solve rate、Code Optimization は percentage of programs optimized、Constrained Generation は coverage %。Dialogue Response, Code Readability, Sentiment Reversal, Acronym Generation は human-pref を用い、さらに GPT-4-pref を proxy として使う。Code Readability では GPT-4 に「文脈上適切に命名された変数の割合」を計算させる記述がある。
- **主な結果**: `tab:results` では 21 個の model-task cells のうち、GPT-3.5 の GSM8K だけ 64.1 -> 64.1 で同点、他は Base から +Self-Refine で改善している。Abstract は平均約 20% absolute improvement と述べる。
- **GPT-4 の代表値**: `tab:results` では Dialogue Response 25.4 -> 74.6 (+49.2)、Sentiment Reversal 3.8 -> 36.2 (+32.4)、Constrained Generation 15.0 -> 45.0 (+30.0)、Code Readability 27.4 -> 56.2 (+28.8)、Acronym Generation 30.4 -> 56.0 (+25.6)、Code Optimization 27.3 -> 36.0 (+8.7)、GSM8K 92.9 -> 93.1 (+0.2)。
- **feedback 品質の ablation**: `tab:ablation_feedback` では、specific/actionable feedback が generic feedback や no feedback より高い。Code Optimization は 27.5 vs 26.0 vs 24.8、Sentiment Reversal は 43.2 vs 31.2 vs 0、Acronym Generation は 56.4 vs 54.0 vs 48.0。
- **iteration の効果**: `fig:iter_score_improvements` では平均スコアが iteration とともに上がる。Code Optimization は $y_0=22.0$ から $y_3=28.8$、Sentiment Reversal は 33.9 から 36.8、Constrained Generation は 29.0 から 49.7。Constrained Generation の差分は +11.3, +6.4, +3.0 で、初期反復の寄与が大きい。
- **数学推論での限界**: GSM8K の改善は小さい。本文は、ChatGPT feedback の 94% が "everything looks good" だったと述べ、自己 feedback が誤り検出に失敗することを理由に挙げる。付録の Oracle Feedback では GPT-3.5 64.06 -> 68.9 (+4.8)、ChatGPT 74.8 -> 76.2 (+1.4)、GPT-4 92.9 -> 93.8 (+0.7) と外部正誤情報で改善が増える。
- **著者が主張する貢献**: 同一 LLM の test-time iterative self-feedback/refinement フレームワーク、追加 training data / RL 不要の簡潔な実装、7 タスクでの有効性、Acronym Generation と CommonGen-Hard の導入、feedback quality・iteration・sampling・model strength に関する分析である。

## 妥当性と限界

- **この主張を支える根拠**: 同じ base LLM の Direct と +Self-Refine を比較しているため、改善は別モデル追加ではなく feedback-refine loop の効果として観察される。`tab:ablation_feedback` は feedback の有無と質を分離し、`fig:iter_score_improvements` は反復回数の効果を示す。さらに 70 samples（35 success / 35 failure cases）の qualitative analysis で、失敗時の 33% は feedback が誤り位置を間違え、61% は不適切な修正案、6% は refiner が良い feedback を誤実装したと報告している。
- **著者が認めている limitations / future work**: strong few-shot / instruction-following ability が必要である。実験は GPT-3.5, ChatGPT, GPT-4, Codex という closed-source models に依存し、pretraining corpus, model sizes, biases が完全には分からない。英語データセットのみで評価している。有害テキスト生成への悪用を明示的には防いでいない。
- **読者として注意すべき点**: 評価の一部は GPT-4-pref に依存し、人間評価との相関は高いが完全ではない。Self-Refine は反復ごとに model calls と prompt 長が増える。TeX 中に明示された範囲では、同じ token budget での self-consistency などとの厳密な Pareto 比較は主実験ではない。`sections/600_analysis.tex` には $k=4$ samples との 1 vs. $k$ 比較があるが、全ての代替推論戦略を代表するわけではない。
- **追加で確認したい実験 / 疑問**: stop score の精度、早すぎる停止と遅すぎる停止の定量評価は本文中では見当たらない。数学や factuality のように自己誤り検出が難しいタスクでは、compiler, unit test, calculator など外部検証器との hybrid を、純 self-feedback と同 token 予算で比較したい。弱い / open model では Vicuna-13B が required format や refinement に失敗しており、prompt engineering でどこまで改善するかも未解決である。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **Self-Refine**: 同一 LLM で initial generation, feedback, refine を反復する手法名。TeX の macro は `\ours`。
- **$\mathcal{M}$**: 生成・feedback・refinement をすべて担う同一 base LLM。
- **$p_{\text{gen}}$**: 初期出力を作るための few-shot prompt。task-specific input-output pairs を含む。
- **$p_{\text{fb}}$**: 現在の出力への feedback を作るための prompt。input-output-feedback triples を含む。
- **$p_{\text{refine}}$**: feedback を使って次の出力を作る prompt。input-output-feedback-refined quadruples を含む。
- **$fb_t$**: 反復 $t$ における自己 feedback。単なる点数ではなく、自然言語で改善箇所と改善方法を書く。
- **actionable feedback**: 改善につながる具体的な action を含む feedback。Code Optimization の例では、遅い for loop を formula に変えるような指示。
- **specific feedback**: 変更すべき具体的 phrase や構造を特定する feedback。
- **stop condition**: Algorithm 1 の $\mathrm{stop}(fb_t,t)$。最大 timestep または feedback 内の stopping indicator から決まる。
- **Base / Direct**: feedback-refine loop を使わず、同じ LLM に 1 回だけ出力させる比較基準。
- **GPT-4-pref**: GPT-4 を評価者として使う preference metric。Sentiment, Acronym, Dialogue で human-pref との相関が報告される。
- **CommonGen-Hard / Constrained Generation**: CommonGen の難化版として本論文で導入されたタスク。通常の 3-5 concepts ではなく 20-30 concepts を文に含める。
- **Oracle Feedback**: GSM8K 付録で使う外部正誤情報つき feedback。純粋な self-feedback ではなく、現在解が誤りかどうかの情報で refinement を進める。
- **Vicuna-13B**: 弱い / smaller model の検証対象。初期生成はできても feedback/refinement prompt 追従に失敗しやすい例として使われる。

## 読む順番の提案

- まず正規ノート `notes/arXiv-2303.17651v2.md` の Summary と Takeaway を読み、論文の主張を「同一 LLM」「追加学習なし」「actionable/specific feedback」「GSM8K の失敗」の 4 点で把握する。
- 次に `sections/100_abstract.tex` と `sections/200_intro.tex` を読む。ここで、問題設定が「複雑な要件を 1 回で満たせない LLM 出力の refinement」であること、既存法が training data / reward / human annotations に寄りがちなことを確認する。
- 手法は `sections/300_method-alt.tex` の Eq. (1)-(4) と Algorithm 1 を先に読む。特に Eq. (4) の履歴つき prompt が、正規ノートの「過去の全 $(y_i, fb_i)$ 履歴を concat する設計」に対応する。
- 実験は `sections/400_tasks.tex`, `sections/500_results.tex`, `tables/main_results.tex` を読む。タスク名・指標・主数値は `tab:results` に集約されている。
- 妥当性を見るには `sections/600_analysis.tex` の `tab:ablation_feedback`（feedback ablation）、`fig:iter_score_improvements`（iteration）、Qualitative Analysis、$k=4$ sampling 比較を読む。正規ノートの Critical Thoughts はここに対応する。
- 限界と比較は `sections/700_related.tex`, `sections/800_conclusion.tex`, `sections/appendix/vicuna_results.tex`, `sections/appendix/additional_analysis.tex` を読む。`tab:related_work_short_summary` は関連研究との差分、Vicuna 付録は instruction-following 前提の弱さを確認する場所である。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2303.17651v2/`
- 正規ノート: `notes/arXiv-2303.17651v2.md`
