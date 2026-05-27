# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning（SFT 前置きなしの強化学習で LLM の推論能力を引き出し、多段 post-training と蒸留に接続する研究）

- arXiv: https://arxiv.org/abs/2501.12948
- 一次ソース: ../papers/arXiv-2501.12948v2/
- 正規ノート: ../notes/arXiv-2501.12948v2.md

---

## 一言で言うと

DeepSeek-V3-Base に対して、人間が書いた推論軌跡による SFT を前置きせず、最終答えを検証する rule-based reward と GRPO だけで DeepSeek-R1-Zero を訓練し、self-reflection・verification・長い CoT が現れることを示す論文である。さらに、読みやすさ・一般タスク・安全性を改善する DeepSeek-R1 の多段 pipeline と、R1 出力 800K サンプルを使う小モデルへの distillation を評価している。

## 何を議論する論文か

- **問題設定**: LLM の数学、コード、STEM、論理のような reasoning capability を、どれだけ人間アノテーションに依存せずに伸ばせるかを扱う。Introduction では、CoT prompting や post-training の成功が "human-annotated demonstrations" に強く依存し、モデル探索を人間の推論様式に制約する可能性があると位置づける。
- **対象範囲 / 仮定**: 主な成功条件は、問題に ground-truth answer やテストケースがあり、正誤を機械的に検証できることである。数学では boxed answer などの final answer matching、コードでは compiler と predefined test cases を使う。Appendix の Data Recipe では数学証明は correctness 判定が難しいため除外されている。
- **既存研究との差分**: DeepSeek-R1-Zero は "bypass the conventional supervised fine-tuning (SFT) phase before RL training" し、outcome-based RL を base model に直接適用する。PPO ではなく value model を不要にする GRPO を使い、reasoning tasks には neural reward model を使わない。
- **この論文で答えたい問い**: 1. 十分大きい base model と信頼できる verifier があれば、SFT なしの pure RL だけで高度な reasoning behaviors が現れるか。2. その能力を読みやすく、一般タスクにも強い最終モデル DeepSeek-R1 にどう統合するか。3. 大モデルで得た reasoning trajectory を小モデルへ蒸留できるか。

## 背景と前提

- **LLM / CoT / SFT / RL**: CoT は最終答えの前に intermediate reasoning steps を出す prompting または学習様式である。SFT は curated input-output pairs を模倣する post-training、RL は reward signal を最大化するように policy を更新する段階として説明されている。
- **DeepSeek-V3-Base と DeepSeek-V3**: Appendix Background では、DeepSeek-V3-Base は 671B total parameters、37B activated per token の MoE、14.8T tokens で pre-trained と説明される。本論文では DeepSeek-R1-Zero と DeepSeek-R1 は DeepSeek-V3-Base の上に訓練され、DeepSeek-V3 は instructed model として baseline や data pipeline に使われる。
- **verifiable tasks**: DeepSeek-R1-Zero の reward は math, coding, logical reasoning domains で正誤を検証できることに依存する。Appendix Discussion の "The importance of verifiers" でも、rule-based RMs と ground-truth に対する LLM judging が reward hacking を抑える鍵だとされる。
- **GRPO と PPO の関係**: GRPO は PPO の value model を省き、同じ question に対する group outputs の reward を mean/std で正規化して advantage を作る。Appendix の "A Comparison of GRPO and PPO" では、PPO は GAE の $\lambda$ 調整と value model の計算負荷が必要で、GRPO は大規模モデルで実用的だと説明される。
- **baseline との関係**: 実験の主な比較対象は DeepSeek-V3、Claude-3.5-Sonnet-1022、GPT-4o-0513、OpenAI-o1-mini、OpenAI-o1-1217 である。蒸留では GPT-4o、Claude-3.5-Sonnet、QwQ-32B-Preview、Qwen2.5-32B-Zero も比較される。

## 提案手法

### コアアイデア

DeepSeek-R1-Zero は、DeepSeek-V3-Base に「まず reasoning process を `<think>...</think>` に書き、最後に `<answer>...</answer>` を出す」という最小限の template だけを与え、content-specific biases を避けたまま GRPO で訓練する。reward は最終答えの accuracy reward と format reward の和であり、reasoning process の内容そのものは明示的に教師しない。

DeepSeek-R1 は、R1-Zero の弱点である poor readability と language mixing、一般タスクの弱さを補うための multi-stage pipeline である。具体的には、数千件の cold-start long CoT による SFT、language consistency reward を加えた first RL、rejection sampling で作った reasoning/non-reasoning data による SFT、helpfulness/harmlessness reward model も使う second RL という流れで訓練される（Figure `fig:r1-pipeline`）。

蒸留では、DeepSeek-R1 が生成した約 800K supervised samples を使い、Qwen2.5-Math、Qwen2.5、Llama 系の小モデルを SFT する。論文は、この段階では RL を入れず、distillation だけの効果を示すことを目的にしている。

### 重要な定義・数式

$$
\begin{aligned}
\mathcal{J}_{GRPO}(\theta)
&= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)]} \\
&\frac{1}{G}\sum_{i=1}^G \left(
\min \left(
\frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)} A_i,
\text{clip} \left(
\frac{\pi_\theta(o_i |q)}{\pi_{\theta_{old}}(o_i |q)}, 1 - \epsilon, 1 + \epsilon
\right) A_i
\right)
- \beta \mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right)
\right), \\
\mathbb{D}_{KL}\left(\pi_{\theta} || \pi_{ref}\right)
&= \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)}
- \log\frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1
\end{aligned}
$$

**式の意味**: GRPO の目的関数であり、old policy から同じ question に対する複数 outputs をサンプルし、advantage が高い output の確率を増やすように policy $\pi_\theta$ を更新する。PPO 型の clipping と reference policy への KL penalty が入る（Equation `eq:GRPO-obj` と直後の KL 式）。

**記号の定義**:
- $q$ ... question
- $P(Q)$ ... question の分布
- $o_i$ ... $i$ 番目の sampled output
- $G$ ... group size。本文の訓練設定では各 question につき 16 outputs
- $\pi_\theta$ ... 更新対象の policy model
- $\pi_{\theta_{old}}$ ... rollout を生成する old policy
- $\pi_{ref}$ ... reference policy。訓練では 400 steps ごとに最新 policy に置換
- $\mathbb{D}_{KL}$ ... policy $\pi_\theta$ と reference policy $\pi_{ref}$ のずれを罰する KL divergence 項
- $A_i$ ... output $o_i$ の advantage
- $\epsilon$ ... GRPO clip ratio。R1 first RL では $\epsilon=10$
- $\beta$ ... KL coefficient。R1-Zero と R1 first RL では 0.001

**この論文での役割**: DeepSeek-R1-Zero と DeepSeek-R1 の RL 更新そのものを定義する中核式である。value model を使わないため、大規模 MoE での reasoning RL を実用的にする根拠として Appendix の PPO 比較にもつながる。

$$
A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \cdots, r_G\})}{\mathrm{std}(\{r_1, r_2, \cdots, r_G\})}
$$

**式の意味**: 同じ question から得た $G$ 個の outputs の reward を group 内で正規化し、各 output が相対的にどれだけ良いかを advantage として表す。

**記号の定義**:
- $A_i$ ... $i$ 番目の output の group-relative advantage
- $r_i$ ... $i$ 番目の output の reward
- $\{r_1,\ldots,r_G\}$ ... 同じ question に対する group outputs の reward
- $\mathrm{mean}$ ... group rewards の平均
- $\mathrm{std}$ ... group rewards の標準偏差

**この論文での役割**: GRPO が value model なしで動く理由を表す式である。PPO の GAE/value model ではなく、同一問題内の相対評価で advantage を作る。

$$
Reward_\text{rule} = Reward_\text{acc} + Reward_\text{format}
$$

**式の意味**: DeepSeek-R1-Zero の rule-based reward は、正答性とフォーマット遵守の和である。本文では accuracy reward と format reward は同じ重みで結合される。

**記号の定義**:
- $Reward_\text{rule}$ ... reasoning task に対する rule-based reward
- $Reward_\text{acc}$ ... final answer が正しいかを評価する reward。数学では answer matching、コードでは compiler と test cases
- $Reward_\text{format}$ ... `<think>` と `</think>` など、指定された形式を守ったかを評価する reward

**この論文での役割**: R1-Zero の「人間の reasoning trajectory を教えず、正しい最終結果と形式だけで訓練する」という設計を表す。reasoning tasks で neural reward model を避ける判断とも結びつく。

$$
Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}
$$

**式の意味**: CoT の中で target language の words が全 words に占める割合を reward として与える。language mixing を減らすための補助 reward である。

**記号の定義**:
- $Reward_{language}$ ... language consistency reward
- $Num(Words_{target})$ ... target language に属する words の数
- $Num(Words)$ ... CoT 全体の words の数

**この論文での役割**: DeepSeek-R1 の first RL stage で、R1-Zero に見られた English/Chinese mixing を抑えるために使われる。Appendix の ablation では、LC reward が language consistency を保つ一方、coding benchmark で slight degradation があるとされる。

$$
\begin{align}
Reward &= Reward_{\text{reasoning}} + Reward_{\text{general}} + Reward_{\text{language}}\\
\text{where, } Reward_{\text{reasoning}} &= Reward_{\text{rule}}\\
Reward_{\text{general}} &= Reward_{\text{reward\_model}} + Reward_{\text{format}}
\end{align}
$$

**式の意味**: DeepSeek-R1 の second RL stage における batch reward の構成である。reasoning data は rule-based reward、general data は reward model と format reward、さらに language reward を足す。

**記号の定義**:
- $Reward$ ... second RL stage の最終 reward
- $Reward_{\text{reasoning}}$ ... math, code, logic などの reasoning data に対する reward
- $Reward_{\text{rule}}$ ... rule-based reward
- $Reward_{\text{general}}$ ... helpfulness/harmlessness など general data に対する reward
- $Reward_{\text{reward\_model}}$ ... helpful reward model または safety reward model の score
- $Reward_{\text{format}}$ ... general data 側の formatting reward
- $Reward_{\text{language}}$ ... language consistency reward

**この論文での役割**: R1 が reasoning だけでなく helpfulness と harmlessness も扱うようにする接続点である。論文は、preference reward を使う訓練を長く続けると reward hacking が起きるため、general instruction data と preference-based rewards は final 400 steps のみに入れると述べる。

### 実装 / アルゴリズム上の要点

- step1: **DeepSeek-R1-Zero** は DeepSeek-V3-Base から開始する。template は Table `tab:r0_template` の `<think>...</think><answer>...</answer>` 形式だけで、reasoning content への制約は入れない。
- step2: R1-Zero の GRPO 設定は learning rate 3e-6、KL coefficient 0.001、rollout temperature 1、各 question 16 outputs、training step あたり 32 unique questions で batch size 512。max length は 8.2k step までは 32,768 tokens、その後 65,536 tokens。全体は 10,400 steps、1.6 training epochs。
- step3: **DeepSeek-R1** は cold-start SFT から始める。Appendix Cold Start では、数千件の high-quality diverse reasoning prompts について R1-Zero で複数 reasoning trajectories を生成し、正答・readable format・repetition/language-mixing filtering を通した後、DeepSeek-V3 で reasoning と summary を整える。
- step4: R1 first RL は R1-Zero とほぼ同じ設定だが、language consistency reward を追加し、GRPO clip ratio $\epsilon=10$ を使う。
- step5: rejection sampling で reasoning-related training samples 約 600K を集め、DeepSeek-V3 SFT data の一部を含む non-reasoning data 約 200K を加える。Table `tab:800k` の合計は 804,745 samples、平均 1.0 rounds、平均 5355.3 tokens。
- step6: R1 second RL は 1,700 steps。temperature は 0.7 に下げる。general instruction data と preference-based rewards は final 400 steps のみに入れ、reward hacking を避ける。
- step7: 蒸留では 800K data で各 base model を 2-3 epochs SFT する。Table `tab:distill_config` の base は Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-14B、Qwen2.5-32B、Llama-3.1-8B、Llama-3.3-70B-Instruct。初期 learning rate は順に $1\times10^{-4}$、$8\times10^{-5}$、$7\times10^{-5}$、$6\times10^{-5}$、$5\times10^{-5}$、$2\times10^{-5}$。
- step8: 訓練コストは Table `tab:cost` で、R1-Zero 101K H800 GPU hours、SFT data creation 5K、R1 41K、total 147K。H800 を \$2/GPU hour と仮定して total \$294K。

## 実験・結果

- **データセット / ベンチマーク**: MMLU、MMLU-Redux、MMLU-Pro、C-Eval、CMMLU、IFEval、FRAMES、GPQA Diamond、SimpleQA、C-SimpleQA、SWE-Bench Verified、Aider、LiveCodeBench (2024-08 -- 2025-01)、Codeforces、CNMO 2024、AIME 2024 を使う。distilled models では AIME 2024、MATH-500、GPQA Diamond、Codeforces、LiveCodeBench を代表結果として報告する。
- **比較対象 / baseline**: Table `tab:main` では Claude-3.5-Sonnet-1022、GPT-4o-0513、DeepSeek-V3、OpenAI-o1-mini、OpenAI-o1-1217、DeepSeek-R1 を比較する。OpenAI-o1-1217 は API access が mainland China で難しいため official reports に基づくと書かれている。蒸留比較では GPT-4o-0513、Claude-3.5-Sonnet-1022、QwQ-32B-Preview、Qwen2.5-32B-Zero も使う。
- **指標**: MMLU 系は EM、DROP は 3-shot F1、IF-Eval は Prompt Strict、GPQA/AIME/MATH/CNMO/LiveCodeBench は Pass@1、AlpacaEval2.0 は LC-winrate、ArenaHard は GPT-4-1106 judge、SWE Verified は Resolved、Aider-Polyglot は Acc.、Codeforces は Percentile と Rating。評価は max generation length 32,768 tokens、temperature 0.6、top-p 0.95 で $k$ samples を生成し、AIME/GPQA は $k=64$、MATH/Codeforces は $k=16$、LCB は $k=8$ を使う。
- **主な結果**: R1-Zero は AIME 2024 Pass@1 が initial 15.6% から 77.9% へ上昇し、self-consistency decoding で 86.7% になる（Figure `fig:r1-zero`）。R1-Zero は MATH-500 95.9、CNMO 2024 88.1、GPQA Diamond 75.8、LiveCodeBench 50.0、Codeforces rating 1444 / percentile 80.4、MMLU 88.8（Table `tab:stage_r1`, Table `tab:v3_full_compare`）。
- **主な結果**: DeepSeek-R1 は AIME 2024 79.8、MATH-500 97.3、CNMO 2024 78.8、LiveCodeBench 65.9、Codeforces rating 2029 / percentile 96.3、SWE Verified 49.2、Aider-Polyglot 53.3、MMLU 90.8、MMLU-Pro 84.0、IF-Eval 83.3、FRAMES 82.5、AlpacaEval2.0 87.6、ArenaHard 92.3（Table `tab:main`）。
- **主な結果**: OpenAI-o1-1217 との比較では、AIME 2024 は 79.2 vs R1 79.8、MATH-500 は 96.4 vs 97.3、LiveCodeBench は 63.4 vs 65.9、Codeforces rating は 2061 vs 2029、SWE Verified は 48.9 vs 49.2、Aider-Polyglot は 61.7 vs 53.3（Table `tab:main`）。
- **主な結果**: 蒸留では DeepSeek-R1-Distill-Qwen-32B が AIME 72.6、MATH 94.3、GPQA Diamond 62.1、LiveCodeBench 57.2、Codeforces rating 1691。DeepSeek-R1-Distill-Llama-70B は AIME 70.0、cons@64 86.7、MATH 94.5、GPQA 65.2、LiveCodeBench 57.5、Codeforces rating 1633（Table `tab:distill`）。
- **主な結果**: Distillation vs RL では、QwQ-32B-Preview が AIME 50.0 / MATH 90.6 / GPQA 54.5 / LCB 41.9、Qwen2.5-32B-Zero が 47.0 / 91.6 / 55.0 / 40.2、DeepSeek-R1-Distill-Qwen-32B が 72.6 / 94.3 / 62.1 / 57.2（Table `tab:distill_vs_rl`）。著者は小モデルでは large-scale RL より蒸留が経済的かつ有効だと結論づける。
- **主な結果**: AIME 2025 では R1 が 11.3/15、すなわち 75% solve rate、OpenAI o1-1217 が 12.0/15、すなわち 80% と報告される。AMC 12 2024 は R1 が 143.7/150、USAMO Index は 256.7（Table `tab:math_2025_eval`）。
- **主な結果**: 安全性では、Table `tab:safety_eval` で DeepSeek R1 は risk control ありの Average Score 95.0、括弧内の pure model は 85.9。HarmBench は risk control あり 89.3、pure model 35.0。DeepSeek R1 (hide cot) は Average Score 96.0、pure model 89.7。
- **著者が主張する貢献**: SFT なしの pure RL でも large-scale model が self-verification や reflection を獲得しうること、読みやすさと一般能力を戻す multi-stage pipeline、GRPO と rule-based reward の具体的設定、R1/R1-Zero/distilled models の公開、小モデルで distillation が直接 RL を上回る実証、PRM と MCTS の失敗知見の共有である。

## 妥当性と限界

- **この主張を支える根拠**: R1-Zero の AIME 2024 15.6% から 77.9% への上昇、response length の増加、Table `tab:aha_moment` の "Wait, wait. Wait. That's an aha moment..."、Appendix Self-Evolution の reflective words が training start と比べ 5- to 7-fold に増える観察が、RL による self-evolution の主要根拠である。
- **この主張を支える根拠**: Table `tab:stage_r1` は R1-Zero、Dev1、Dev2、Dev3、R1 を同じ benchmark 群で並べ、cold-start SFT が instruction-following を上げる一方で AIME が 77.9 から 59.0 に下がり、reasoning-oriented RL と後段 SFT/RL で回復する流れを示す。
- **この主張を支える根拠**: Table `tab:v3_full_compare` は同じ DeepSeek-V3-Base を共有する DeepSeek-V3 と R1 系列を比較し、LiveCodeBench 36.2 vs 65.9、AIME 39.2 vs 79.8、MATH-500 90.2 vs 97.3 のように、reasoning benchmarks で post-training 差が大きいことを示す。
- **この主張を支える根拠**: Appendix Decontamination は DeepSeek-V3 base の knowledge cutoff を July 2024 とし、evaluation questions/reference solutions と 10-gram matching する pre-training texts と post-training data を除去したと説明する。数学だけで約 six million potential pre-training texts を除去した一方、n-gram method は paraphrase of testset を防げないとも明記する。
- **著者が認めている limitations / future work**: DeepSeek-R1 は structure output と tool use が既存モデルより弱く、search engines や calculators を使えない。token efficiency には overthinking が残る。Chinese/English 以外では language mixing が起きうる。few-shot prompting は consistently degrades performance とされ、zero-shot setting が推奨される。software engineering tasks は long evaluation times のため大規模 RL が十分適用されていない。
- **著者が認めている limitations / future work**: pure RL は reliable reward signals に依存する。writing など reliable reward model を作りにくい task では model-based reward が exploitation を受けやすく、policy model が reward model を hack する可能性がある。論文では human annotation による supervised data と数百 step の RL に留めている。
- **読者として注意すべき点**: "The importance of base checkpoint" で、7B dense model と 16B MoE model は AIME で meaningful improvements を示さず、長い response が repetition になったと報告される。したがって「どのサイズでも pure RL が効く」という主張ではなく、十分強い base model と verifier がある条件での結果として読む必要がある。
- **読者として注意すべき点**: R1-Zero の "aha moment" は定性的な例と reflective words の頻度変化で示されるが、"wait" を増やすこと自体が精度を上げたという介入実験は TeX 中には明示されていない。
- **読者として注意すべき点**: 安全性は risk control system の有無で大きく変わる。Table `tab:safety_jailbreak` では DeepSeek-R1 pure model は jailbreak unsafe ratio 85.9 だが、risk control system ありでは 4.3、rejected ratio は 87.3 になる。open-source model を local deployment する場合、論文自身が comparable risk control measures を勧めている。
- **追加で確認したい実験 / 疑問**: cold-start data の量や様式を変えた ablation、base model capacity の閾値、R1-Distill models に RL を追加した場合の伸び、non-Chinese/English での language reward 設計、PRM を RL reward ではなく reranker として使った場合の Pass@1 改善、tool-use RL の追加効果を確認したい。

## 用語メモ

一般的な辞書的定義ではなく、この論文での使われ方を中心に書く。

- **DeepSeek-V3-Base** ... R1-Zero と R1 の出発点になる base model。671B total parameters、37B activated per token の MoE。
- **DeepSeek-R1-Zero** ... SFT を前置きせず、rule-based reward と GRPO だけで訓練した reasoning model。poor readability と language mixing が課題。
- **DeepSeek-R1** ... cold-start SFT、RL、rejection sampling SFT、second RL を組み合わせた最終モデル。reasoning capability と一般タスク・preference alignment の両立を狙う。
- **GRPO** ... Group Relative Policy Optimization。同じ question への複数 outputs の reward を group 内で正規化し、value model なしで advantage を作る RL algorithm。
- **rule-based reward** ... final answer matching、compiler/test cases、format checker など、事前定義ルールで計算する reward。reasoning tasks では neural reward model を使わない。
- **accuracy reward** ... 数学なら reference answer と一致するか、コードなら hidden/predefined tests を通るかを評価する reward。
- **format reward** ... 本文では特に reasoning process を `<think>` と `</think>` で囲む formatting requirement を守ることへの reward。
- **language consistency reward** ... CoT 内の target language words の割合。R1 で language mixing を減らすために使われる。
- **CoT** ... Chain-of-Thought。論文では長い reasoning process を出してから final answer を出す形式で、self-reflection、verification、alternative approaches を含みうる。
- **cold-start data** ... R1 の初期 SFT に使う、human-readable で conversational な long CoT data。R1-Zero の raw CoT をそのまま使うのではなく、人手確認や DeepSeek-V3 による refinement を含む。
- **rejection sampling** ... 複数 responses を生成し、正答・可読性・format などを満たすものだけを SFT data として残す手順。
- **Reward Model (RM)** ... general data の helpfulness/harmlessness を評価する model。helpful RM は pairwise、safety RM は point-wise で訓練される。
- **Reward Hacking** ... reward function の欠陥や偏りを policy model が利用し、高 reward だが人間意図に合わない output を出す現象。論文では helpful RM 使用時に観察され、preference reward を final 400 steps に制限する理由になっている。
- **PRM** ... Process Reward Model。途中ステップを採点する reward model。論文では fine-grain step 定義、intermediate correctness 判定、reward hacking のため large-scale RL では採用しない。
- **MCTS** ... Monte Carlo Tree Search。token generation では search space が指数的に大きく、value model の訓練が難しいため、self-search による iterative improvement は困難だったと報告される。
- **Pass@1 / cons@64** ... Pass@1 は sampled responses の正答率平均として報告される。cons@64 は 64 samples の majority vote による consensus 結果。
- **MoE** ... Mixture-of-Experts。DeepSeek-V3-Base/R1 は total parameters は 671B だが、各 token で activated されるのは 37B。
- **distillation** ... DeepSeek-R1 が作った high-quality outputs を supervised data として、小さい base model に SFT する手法。本論文の distillation 実験では RL stage を入れない。

## 読む順番の提案

- まず `sn-article.tex` の Abstract と Introduction を読む。ここで "human-annotated demonstrations" への依存、SFT を前置きしない理由、R1-Zero と R1 の関係を押さえる。正規ノートでは Summary の「問題」「手法」に対応する。
- 次に `sn-article.tex` の Section `DeepSeek-R1-Zero` を読む。Equation `eq:GRPO-obj`、advantage 式、Table `tab:r0_template`、reward 式、Figure `fig:r1-zero`、Table `tab:aha_moment` を優先する。正規ノートでは R1-Zero の training details と "aha moment" に対応する。
- その後 `sn-article.tex` の Section `DeepSeek-R1` と Figure `fig:r1-pipeline` を読む。Model-based Rewards、first RL、second RL の reward 構成を追うと、R1-Zero から R1 へ何を補ったかが分かる。正規ノートでは multi-stage pipeline の箇条書きに対応する。
- 実験はまず Table `tab:stage_r1` と Table `tab:main` を見る。R1-Zero、Dev1、Dev2、Dev3、R1 の役割分担と、OpenAI-o1-1217 など baseline との差が把握できる。次に Table `tab:distill` と `tab:distill_vs_rl` を見ると、distillation が小モデルで強いという主張につながる。
- Appendix は `Data Recipe`、`Hyper-Parameters`、`Training Cost`、`Self-Evolution`、`More Analysis`、`Discussion` の順に読むとよい。特に `sec:conta` はデータ汚染、`sec:attempt` は PRM/MCTS の失敗理由に対応する。limitations は `sn-article.tex` の Conclusion, Limitation, and Future Work（`sec:limit`）で確認する。
- 安全性を読む場合は `Ethics and Safety Statement` の後で Appendix `DeepSeek-R1 Safety Report`、Table `tab:safety_eval`、`tab:safety_taxnomy`、`tab:safety_jailbreak` を見る。risk control system の有無で数字が変わる点に注意する。
- `00README.json` では top-level source は `sn-article.tex` とされている。この展開には `main.bbl` は見当たらないため、citation key から文献タイトルを展開しない。`appendix.tex` 末尾に `\bibliography{sn-bibliography}` があることだけ確認できる。

## もとの論文・正規ノート

- 論文 TeX: `papers/arXiv-2501.12948v2/`
- 正規ノート: `notes/arXiv-2501.12948v2.md`
